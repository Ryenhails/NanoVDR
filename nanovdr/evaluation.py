"""ViDoRe evaluation and deployment profiling.

Two things a retriever has to be judged on, measured under one protocol:
retrieval quality, and what it costs to run. Reporting the first without the
second is how a model that needs a 256 GB index gets compared to one that
needs 16.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .scoring import score as score_reprs

__all__ = ["ndcg_at_k", "evaluate_retrieval", "profile_model", "VIDORE"]

VIDORE = {
    "v1": {
        "arxivqa": "vidore/arxivqa_test_subsampled",
        "docvqa": "vidore/docvqa_test_subsampled",
        "infovqa": "vidore/infovqa_test_subsampled",
        "tabfquad": "vidore/tabfquad_test_subsampled",
        "tatdqa": "vidore/tatdqa_test",
        "shiftproject": "vidore/shiftproject_test",
        "syntheticDocQA_ai": "vidore/syntheticDocQA_artificial_intelligence_test",
        "syntheticDocQA_energy": "vidore/syntheticDocQA_energy_test",
        "syntheticDocQA_govt": "vidore/syntheticDocQA_government_reports_test",
        "syntheticDocQA_health": "vidore/syntheticDocQA_healthcare_industry_test",
    },
    "v2": {
        "esg_reports": "vidore/esg_reports_v2",
        "biomedical_lectures": "vidore/biomedical_lectures_v2",
        "economics_reports": "vidore/economics_reports_v2",
        "esg_reports_human": "vidore/esg_reports_human_labeled_v2",
    },
    "v3": {
        "finance_en": "vidore/vidore_v3_finance_en",
        "finance_fr": "vidore/vidore_v3_finance_fr",
        "cs": "vidore/vidore_v3_computer_science",
        "hr": "vidore/vidore_v3_hr",
        "energy": "vidore/vidore_v3_energy",
        "industrial": "vidore/vidore_v3_industrial",
        "pharma": "vidore/vidore_v3_pharmaceuticals",
        "physics": "vidore/vidore_v3_physics",
    },
}


def ndcg_at_k(scores: np.ndarray, qrels: Sequence[Sequence[int]], k: int = 5) -> float:
    """Mean NDCG@k with binary relevance. ``scores`` is (Q, N)."""
    out = []
    for q, rel in enumerate(qrels):
        if not len(rel):
            continue
        rel_set = set(rel)
        top = np.argsort(-scores[q])[:k]
        dcg = sum((1.0 if d in rel_set else 0.0) / np.log2(i + 2) for i, d in enumerate(top))
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        out.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(out)) if out else float("nan")


@torch.no_grad()
def evaluate_retrieval(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    qrels: Sequence[Sequence[int]],
    k: int = 5,
) -> float:
    """NDCG@k for already-encoded sides. Accepts a (Q, K, D) query tensor for
    late interaction, or (Q, D) for a dot product."""
    from .align import Repr

    q = (
        Repr(vec=query_emb)
        if query_emb.dim() == 2
        else Repr(tokens=query_emb, mask=torch.ones(query_emb.shape[:2], dtype=torch.bool))
    )
    s = score_reprs(q, Repr(vec=doc_emb)).cpu().numpy()
    return ndcg_at_k(s, qrels, k=k)


@torch.no_grad()
def profile_model(
    doc_tower,
    pages: Sequence,
    batch_size: int = 8,
    warmup: int = 3,
    embed_dim: Optional[int] = None,
    dtype_bytes: int = 4,
) -> dict:
    """Measure indexing throughput, peak VRAM and index footprint.

    Reported per model so that quality and cost sit in the same table. Warmup
    passes are excluded and the CUDA cache is reset first, otherwise the first
    batch's allocator behaviour dominates the peak-memory number.
    """
    device = next(doc_tower.parameters()).device
    for _ in range(warmup):
        doc_tower.encode(pages[:batch_size], batch_size=batch_size, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    emb = doc_tower.encode(pages, batch_size=batch_size, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    dim = embed_dim or emb.shape[-1]
    return {
        "pages": len(pages),
        "seconds": round(elapsed, 3),
        "pages_per_second": round(len(pages) / max(elapsed, 1e-9), 2),
        "peak_vram_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None
        ),
        "embed_dim": int(dim),
        "index_gb_per_million": round(dim * dtype_bytes * 1e6 / 1e9, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a document tower on ViDoRe.")
    ap.add_argument("--doc-model", required=True, help="Hub id or local path")
    ap.add_argument("--teacher-cache", required=True,
                    help="Directory of cached teacher query embeddings and qrels")
    ap.add_argument("--benchmarks", nargs="+", default=["v1", "v2", "v3"], choices=list(VIDORE))
    ap.add_argument("--datasets", nargs="*", default=None, help="Restrict to these dataset keys")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import h5py
    from datasets import get_dataset_config_names, load_dataset

    from .towers import DocTower

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tower = DocTower.from_pretrained(args.doc_model).to(device).eval()
    cache = Path(args.teacher_cache)
    results: dict[str, dict] = {}

    for bench in args.benchmarks:
        per_dataset = {}
        for name, hub_id in VIDORE[bench].items():
            if args.datasets and name not in args.datasets:
                continue
            corpus_h5 = cache / f"teacher_8b_{name}_corpus.h5"
            queries_npy = cache / f"teacher_8b_{name}_queries.npy"
            meta_json = cache / "qrels_meta" / f"{name}.json"
            if not (corpus_h5.exists() and queries_npy.exists() and meta_json.exists()):
                print(f"  skip {name}: teacher cache incomplete", flush=True)
                continue

            with h5py.File(corpus_h5, "r") as f:
                t_docs = np.asarray(f["doc_emb"][:], dtype=np.float32)
            t_queries = np.load(queries_npy).astype(np.float32)
            meta = json.loads(meta_json.read_text())
            qid2i = {q: i for i, q in enumerate(meta["qids"])}
            cid2i = {c: i for i, c in enumerate(meta["cids"])}
            qrels = [[] for _ in meta["qids"]]
            for r in meta["qrels"]:
                if r.get("score", 1) > 0:
                    qrels[qid2i[r["query-id"]]].append(cid2i[r["corpus-id"]])

            # v1 ships one flat table; v2 and v3 split corpus/queries/qrels into
            # separate configs, and loading those without naming one either fails
            # or silently hands back the query table instead of the pages.
            configs = get_dataset_config_names(hub_id)
            ds = load_dataset(hub_id, "corpus" if "corpus" in configs else None,
                              split="test")
            col = "image" if "image" in ds.column_names else ds.column_names[0]
            seen, pages = set(), []
            for i in range(len(ds)):
                row = ds[i]
                key = row.get("image_filename") or i
                if key in seen:
                    continue
                seen.add(key)
                pages.append(row[col])
                if len(pages) >= t_docs.shape[0]:
                    break

            s_docs = tower.encode(pages, batch_size=args.batch_size, device=device)
            tq = torch.from_numpy(t_queries / np.linalg.norm(t_queries, axis=1, keepdims=True))
            td = torch.from_numpy(t_docs / np.linalg.norm(t_docs, axis=1, keepdims=True))

            student = evaluate_retrieval(tq, s_docs, qrels, k=args.k) * 100
            teacher = evaluate_retrieval(tq, td, qrels, k=args.k) * 100
            per_dataset[name] = {
                "student_ndcg": round(student, 2),
                "teacher_ndcg": round(teacher, 2),
                "retention": round(student / teacher * 100, 1) if teacher else None,
            }
            print(f"  {bench}/{name:22s} student {student:5.2f}  teacher {teacher:5.2f}", flush=True)

        if per_dataset:
            avg_s = float(np.mean([v["student_ndcg"] for v in per_dataset.values()]))
            avg_t = float(np.mean([v["teacher_ndcg"] for v in per_dataset.values()]))
            results[bench] = {
                "per_dataset": per_dataset,
                "student_avg": round(avg_s, 2),
                "teacher_avg": round(avg_t, 2),
                "retention": round(avg_s / avg_t * 100, 1) if avg_t else None,
            }
            print(f"{bench} avg: student {avg_s:.2f}  teacher {avg_t:.2f}  "
                  f"retention {results[bench]['retention']}%", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.doc_model, "results": results}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
