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
    ap.add_argument("--kind", choices=["image", "query", "query_tokens"], default="image")
    ap.add_argument("--column", default=None, help="Column to encode (default: image/query)")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER,
                    help="a Qwen3-VL hub id, or a multi-vector key: "
                         + ", ".join(sorted(MULTIVECTOR_TEACHERS)))
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

    if args.kind == "query_tokens":
        mv = MultiVectorTeacher(args.teacher)
        cache_multivector_targets(
            mv,
            [ds[i][col] for i in range(len(ds))],
            args.out,
            batch_size=args.batch_size,
        )
        return 0

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


# ==========================================================================
#  Multi-vector (late-interaction) teachers
# ==========================================================================
# ColNanoVDR distils from ColPali-style teachers, which emit a *set* of token
# vectors per query instead of one pooled vector. Everything below is the
# multi-vector counterpart of the single-vector caching path above: a uniform
# wrapper over the three loading conventions these teachers ship with, and a
# ragged target cache that a query token set actually fits in.

MULTIVECTOR_TEACHERS: dict[str, tuple[str, int, str]] = {
    # key            -> (hub id, embedding width, backend)
    "colqwen35": ("athrael-soju/colqwen3.5-4.5B-v3", 320, "colpali"),
    "vultron":   ("vultr/VultronRetrieverCore-Qwen3.5-4.5B", 320, "colpali"),
    "tomoro8b":  ("TomoroAI/tomoro-colqwen3-embed-8b", 320, "colqwen3"),
    "colvec4b":  ("webAI-Official/webAI-ColVec1.1-4b", 640, "colvec"),
    "colvec8b":  ("webAI-Official/webAI-ColVec1.1-8b", 640, "colvec"),
}


class MultiVectorTeacher:
    """Uniform ``encode_queries`` over the late-interaction teachers.

    The three backends differ in how they load and in what they return:

    ``colpali``   ``colpali_engine``'s ``ColQwen3_5``; the projection is
                  already L2-normalised, and padding comes back as zero rows.
    ``colqwen3``  remote code on the Hub; the processor exposes
                  ``process_texts`` rather than ``process_queries``.
    ``colvec``    remote code returning *raw* hidden states, so the caller has
                  to normalise, and padding is identified by the attention
                  mask rather than by zero rows.

    Normalising here rather than in the loss is deliberate: the objective
    assumes both measures already sit on the unit sphere, and a teacher that
    quietly did not would degrade silently instead of failing.
    """

    def __init__(self, name: str = "colqwen35", device: str = "cuda"):
        if name not in MULTIVECTOR_TEACHERS:
            raise KeyError(
                f"Unknown multi-vector teacher {name!r}. "
                f"Available: {sorted(MULTIVECTOR_TEACHERS)}"
            )
        self.name = name
        self.model_id, self.embed_dim, self.backend = MULTIVECTOR_TEACHERS[name]

        if self.backend == "colpali":
            from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

            self.processor = ColQwen3_5Processor.from_pretrained(self.model_id)
            self.model = None
            for attn in ("sdpa", "eager"):
                try:
                    self.model = ColQwen3_5.from_pretrained(
                        self.model_id, dtype=torch.bfloat16,
                        attn_implementation=attn, device_map=device,
                    ).eval()
                    break
                except Exception as exc:
                    print(f"  attn={attn} failed: {type(exc).__name__}", flush=True)
            if self.model is None:
                raise RuntimeError(f"could not load {self.model_id}")
        else:
            from transformers import AutoModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map=device,
            ).eval()

        self.device = next(self.model.parameters()).device
        self.normalize = self.backend == "colvec"

    def _process(self, queries: Sequence[str]) -> dict:
        if self.backend == "colqwen3":
            return self.processor.process_texts(list(queries))
        if self.backend == "colvec":
            return self.processor.process_queries(list(queries))
        return self.processor.process_queries(queries=list(queries))

    @torch.no_grad()
    def encode_queries(self, queries: Sequence[str], batch_size: int = 32) -> list[np.ndarray]:
        """Encode queries into a list of ``(K_i, D)`` float16 token sets."""
        out: list[np.ndarray] = []
        for start in range(0, len(queries), batch_size):
            inputs = self._process(queries[start : start + batch_size])
            inputs = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
            res = self.model(**inputs)
            hidden = res if isinstance(res, torch.Tensor) else res[0]
            attn = inputs.get("attention_mask") if self.backend == "colvec" else None
            out.extend(_unpad_tokens(hidden, attn, self.normalize))
        return out


def _unpad_tokens(hidden, attn_mask, normalize: bool) -> list[np.ndarray]:
    """``(B, L, D)`` padded -> a list of ``(K_i, D)`` float16 real-token blocks."""
    out = []
    for i in range(hidden.shape[0]):
        v = hidden[i]
        if attn_mask is not None:
            v = v[attn_mask[i].bool()]
        v = v[v.float().norm(dim=-1) > 1e-6]
        if normalize:
            v = v / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).to(v.dtype)
        out.append(v.cpu().float().numpy().astype(np.float16))
    return out


def cache_multivector_targets(
    teacher: MultiVectorTeacher,
    queries: Sequence[str],
    out_path: str | Path,
    batch_size: int = 32,
    flush_every: int = 2000,
) -> Path:
    """Write a ragged query-token target cache.

    Token sets have different lengths, so they are stored concatenated with an
    offset index rather than padded into a rectangle:

        tokens   (T, D) float16   every token of every query, end to end
        offsets  (N+1,) int64     query i owns tokens[offsets[i]:offsets[i+1]]
        status   (N,)   uint8     1 where the query encoded

    Padding a 1.5M-query cache to the longest query would multiply it several
    times over, and the loss consumes a mask anyway.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(queries)

    blocks: list[np.ndarray] = []
    status = np.zeros(total, dtype=np.uint8)
    lengths = np.zeros(total, dtype=np.int64)
    t0 = time.time()

    for start in range(0, total, batch_size):
        chunk = list(queries[start : start + batch_size])
        try:
            embs = teacher.encode_queries(chunk, batch_size=len(chunk))
            for j, e in enumerate(embs):
                blocks.append(e)
                lengths[start + j] = e.shape[0]
                status[start + j] = 1
        except Exception as exc:
            print(f"  [warn] rows {start}-{start+len(chunk)-1} failed: "
                  f"{type(exc).__name__}", flush=True)
            for j in range(len(chunk)):
                blocks.append(np.zeros((0, teacher.embed_dim), dtype=np.float16))
        done = start + len(chunk)
        if done % flush_every < batch_size:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{total}  {rate:.1f}/s  "
                  f"eta {(total-done)/max(rate,1e-6)/60:.0f} min", flush=True)

    offsets = np.zeros(total + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = (np.concatenate(blocks, axis=0) if blocks
              else np.zeros((0, teacher.embed_dim), dtype=np.float16))

    with h5py.File(out_path, "w") as f:
        f.create_dataset("tokens", data=tokens, dtype=np.float16)
        f.create_dataset("offsets", data=offsets, dtype=np.int64)
        f.create_dataset("status", data=status, dtype=np.uint8)
        f.attrs["teacher"] = teacher.model_id
        f.attrs["kind"] = "query_tokens"
        f.attrs["embed_dim"] = teacher.embed_dim

    print(f"done: {int(status.sum())}/{total} encoded, "
          f"{tokens.shape[0]} tokens, {out_path}", flush=True)
    return out_path


def load_multivector_targets(
    path: str | Path,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Read a ragged target cache. Returns ``(tokens, offsets, valid_indices)``."""
    with h5py.File(path, "r") as f:
        tokens = torch.from_numpy(f["tokens"][:].astype(np.float32))
        offsets = f["offsets"][:]
        valid = np.nonzero(f["status"][:] > 0)[0]
    return tokens, offsets, valid
