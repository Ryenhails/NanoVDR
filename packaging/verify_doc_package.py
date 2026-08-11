"""Environment check for the packaged doc tower.

Loads the package the way a user would -- from a directory, through
``AutoModel`` with ``trust_remote_code`` -- and checks it against the cached 8B
teacher embeddings for a real ViDoRe corpus:

  1. cosine(student page, teacher page) on the same images
  2. NDCG@5 with teacher queries against student pages (doc-side isolation),
     next to the teacher x teacher ceiling on the same data

A correctly packaged model reproduces the training-time alignment (~0.8 cosine)
and retains most of the teacher's retrieval quality. A broken one shows cosine
near zero.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

# Directory holding the cached teacher evaluation embeddings, laid out as
#   teacher_8b_<dataset>_corpus.h5   dataset "doc_emb"  (N, D) float16
#   teacher_8b_<dataset>_queries.npy                    (Q, D)
#   qrels_meta/<dataset>.json        {"qids", "cids", "qrels"}
# Override with NANOVDR_EVAL_CACHE.
EVAL_CACHE = Path(os.environ.get("NANOVDR_EVAL_CACHE", "./teacher_cache/eval"))
VIDORE = {
    "arxivqa": "vidore/arxivqa_test_subsampled",
    "docvqa": "vidore/docvqa_test_subsampled",
    "tabfquad": "vidore/tabfquad_test_subsampled",
}


def ndcg_at_k(scores: np.ndarray, qrels: list[list[int]], k: int = 5) -> float:
    """scores: (Q, D). qrels[q] = list of relevant doc indices."""
    out = []
    for q, rel in enumerate(qrels):
        if not rel:
            continue
        top = np.argsort(-scores[q])[:k]
        gains = [1.0 if d in set(rel) else 0.0 for d in top]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
        out.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(out)) if out else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--dataset", default="arxivqa", choices=sorted(VIDORE))
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    from transformers import AutoImageProcessor, AutoModel

    print(f"loading package from {args.package} (trust_remote_code=True) ...")
    model = AutoModel.from_pretrained(args.package, trust_remote_code=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.package, trust_remote_code=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {n_par/1e6:.2f}M parameters, embed_dim={model.cfg.embed_dim}, "
          f"tiles<={model.cfg.tile_max_num} (+thumbnail, {model.cfg.tile_max_total} max)")

    name = args.dataset
    with h5py.File(EVAL_CACHE / f"teacher_8b_{name}_corpus.h5", "r") as h:
        t_docs = np.asarray(h["doc_emb"][:], dtype=np.float32)
    t_queries = np.load(EVAL_CACHE / f"teacher_8b_{name}_queries.npy").astype(np.float32)
    meta = json.loads((EVAL_CACHE / "qrels_meta" / f"{name}.json").read_text())
    print(f"cache: {t_docs.shape[0]} docs, {t_queries.shape[0]} queries, dim {t_docs.shape[1]}")

    from datasets import load_dataset

    ds = load_dataset(VIDORE[name], split="test")
    img_col = "image" if "image" in ds.column_names else ds.column_names[0]

    # corpus order in the cache follows the deduplicated page order used at eval
    n_docs = t_docs.shape[0] if not args.limit else min(args.limit, t_docs.shape[0])
    seen, pages, keep = {}, [], []
    for i in range(len(ds)):
        row = ds[i]
        key = row.get("image_filename") or i
        if key in seen:
            continue
        seen[key] = len(pages)
        pages.append(row[img_col])
        keep.append(len(pages) - 1)
        if len(pages) >= n_docs:
            break
    print(f"encoding {len(pages)} pages ...")

    s_docs = model.encode(pages, processor, batch_size=args.batch_size, device=device).numpy()

    # ---- check 1: per-page alignment with the teacher
    t_sub = t_docs[: len(pages)]
    t_sub = t_sub / np.linalg.norm(t_sub, axis=1, keepdims=True)
    cos = (s_docs * t_sub).sum(1)
    print(f"\ncosine(student page, teacher page): mean {cos.mean():.4f}  "
          f"min {cos.min():.4f}  max {cos.max():.4f}")

    # ---- check 2: retrieval, teacher queries against student pages
    if len(pages) == t_docs.shape[0]:
        qid2i = {q: i for i, q in enumerate(meta["qids"])}
        cid2i = {c: i for i, c in enumerate(meta["cids"])}
        qrels = [[] for _ in meta["qids"]]
        for r in meta["qrels"]:
            if r.get("score", 1) > 0:
                qrels[qid2i[r["query-id"]]].append(cid2i[r["corpus-id"]])
        tq = t_queries / np.linalg.norm(t_queries, axis=1, keepdims=True)
        ndcg_tt = ndcg_at_k(tq @ t_sub.T, qrels)
        ndcg_ts = ndcg_at_k(tq @ s_docs.T, qrels)
        print(f"NDCG@5  teacher x teacher : {ndcg_tt*100:.2f}")
        print(f"NDCG@5  teacher x student : {ndcg_ts*100:.2f}   "
              f"(retention {ndcg_ts/ndcg_tt*100:.1f}%)")
    else:
        print("(skipping NDCG: encoded a subset, not the full corpus)")

    ok = cos.mean() > 0.6
    print(f"\n{'PASS' if ok else 'FAIL'}: packaged model reproduces teacher alignment"
          if ok else "\nFAIL: alignment far below training level — packaging is wrong")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
