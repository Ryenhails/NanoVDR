"""Teacher target precomputation.

The teacher runs exactly once, before any student training, and its outputs are
written to disk. Everything downstream reads those cached targets, which is
what makes the two towers independent: neither one ever needs the teacher in
memory, and both can train at the same time on the same cache.

The cache format is deliberately dull:

    <name>.h5
      doc_emb  (N, D) float16   teacher embeddings, row-aligned with the dataset
      status   (N,)   uint8     1 where encoding succeeded, 0 where it failed

Storing float16 halves the cache and costs nothing measurable, since the
targets are L2-normalised before use anyway.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
import torch

__all__ = ["TeacherEmbedder", "cache_targets", "load_targets"]

DEFAULT_TEACHER = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_QUERY_INSTRUCTION = "Find a document image that matches the given query."


class TeacherEmbedder:
    """Thin wrapper over the released Qwen3-VL embedder.

    The upstream model ships its own encoder class rather than a standard
    ``AutoModel`` head, so it is fetched from the snapshot and imported. Kept
    behind this interface so a different teacher only has to satisfy
    ``encode_images`` / ``encode_queries``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEACHER,
        dtype: torch.dtype = torch.float16,
        attn_implementation: str = "flash_attention_2",
    ):
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(model_name)
        sys.path.insert(0, local_dir)
        from scripts.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

        t0 = time.time()
        self.model = Qwen3VLEmbedder(
            model_name_or_path=model_name,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.name = model_name
        print(f"  teacher {model_name} loaded in {time.time() - t0:.1f}s", flush=True)

    @property
    def embed_dim(self) -> int:
        return int(getattr(self.model, "embed_dim", 4096))

    @torch.no_grad()
    def encode_images(self, images: Sequence) -> np.ndarray:
        """(B,) PIL images -> (B, D) float16."""
        embs = self.model.process([{"image": im} for im in images])
        return embs.cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def encode_queries(
        self,
        queries: Sequence[str],
        instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    ) -> np.ndarray:
        """(B,) strings -> (B, D) float16.

        The instruction must match the one the query tower prepends at training
        and inference time, otherwise the student is chasing a target its input
        never described.
        """
        items = [{"text": q, "instruction": instruction} if instruction else {"text": q} for q in queries]
        embs = self.model.process(items)
        return embs.cpu().numpy().astype(np.float16)


def cache_targets(
    teacher: TeacherEmbedder,
    items: Iterable,
    out_path: str | Path,
    total: int,
    kind: str = "image",
    batch_size: int = 4,
    embed_dim: Optional[int] = None,
    instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    flush_every: int = 200,
) -> Path:
    """Stream ``items`` through the teacher and write an HDF5 target cache.

    Rows that fail to encode are left at zero with ``status = 0`` rather than
    dropped, so the cache stays row-aligned with the source dataset and a later
    run can fill the gaps.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim = embed_dim or teacher.embed_dim

    encode = teacher.encode_images if kind == "image" else (
        lambda batch: teacher.encode_queries(batch, instruction=instruction)
    )

    t0, done, failed = time.time(), 0, 0
    with h5py.File(out_path, "w") as f:
        emb_ds = f.create_dataset("doc_emb", shape=(total, dim), dtype=np.float16)
        status_ds = f.create_dataset("status", shape=(total,), dtype=np.uint8)
        f.attrs["teacher"] = teacher.name
        f.attrs["kind"] = kind
        if kind == "query" and instruction:
            f.attrs["instruction"] = instruction

        batch, idxs = [], []
        for i, item in enumerate(items):
            batch.append(item)
            idxs.append(i)
            if len(batch) < batch_size and i + 1 < total:
                continue
            try:
                emb_ds[idxs[0] : idxs[-1] + 1] = encode(batch)
                status_ds[idxs[0] : idxs[-1] + 1] = 1
            except Exception as exc:  # keep going; a bad page should not kill a 12-hour job
                failed += len(batch)
                print(f"  [warn] rows {idxs[0]}-{idxs[-1]} failed: {type(exc).__name__}", flush=True)
            done += len(batch)
            batch, idxs = [], []
            if done % flush_every < batch_size:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (total - done) / max(rate, 1e-6) / 60
                print(f"  {done}/{total}  {rate:.1f}/s  eta {eta:.0f} min", flush=True)

    print(f"done: {done - failed}/{total} encoded, {failed} failed, {out_path}", flush=True)
    return out_path


def load_targets(
    path: str | Path,
    embed_dim: Optional[int] = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Read a target cache. Returns ``(targets, valid_indices)``.

    Targets keep their original row indices so they line up with the source
    dataset; ``valid_indices`` lists the rows that actually encoded.
    """
    with h5py.File(path, "r") as f:
        emb = f["doc_emb"][:, :embed_dim] if embed_dim else f["doc_emb"][:]
        targets = torch.from_numpy(emb.astype(np.float32))
        valid = np.nonzero(f["status"][:] > 0)[0]
    return targets, valid


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache teacher targets for a dataset.")
    ap.add_argument("--data", required=True, help="Arrow dataset directory")
    ap.add_argument("--out", required=True, help="Output .h5 path")
    ap.add_argument("--kind", choices=["image", "query"], default="image")
    ap.add_argument("--column", default=None, help="Column to encode (default: image/query)")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=None)
    ap.add_argument("--instruction", default=DEFAULT_QUERY_INSTRUCTION)
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    from datasets import load_from_disk

    ds = load_from_disk(args.data)
    col = args.column or ("image" if args.kind == "image" else "query")
    if col not in ds.column_names:
        raise SystemExit(f"column {col!r} not in dataset; have {ds.column_names}")

    teacher = TeacherEmbedder(args.teacher, attn_implementation=args.attn)
    cache_targets(
        teacher,
        (ds[i][col] for i in range(len(ds))),
        args.out,
        total=len(ds),
        kind=args.kind,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        instruction=args.instruction if args.kind == "query" else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
