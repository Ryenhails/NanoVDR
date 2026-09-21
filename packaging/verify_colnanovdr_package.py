"""Check that a packaged ColNanoVDR tower scores exactly like the checkpoint.

Packaging rebuilds the forward pass out of sentence-transformers modules, so the
only claim worth making is that the rebuilt path is the *same function*. This
encodes the same queries twice -- once through ``QueryTower``, once through the
packaged ``MultiVectorEncoder`` -- and compares element-wise.

Both sides are run on CPU in float32. A mismatch here means the package would
retrieve differently from the checkpoint the numbers were measured on, so a
failure is a hard stop, not a warning.
"""

from __future__ import annotations

import argparse
import sys

import torch

QUERIES = [
    "What was the revenue growth in Q3 2024?",
    "Quel est le chiffre d'affaires de l'exercice precedent ?",
    "energy",
    "Explain the relationship between the reported emissions intensity and the "
    "production volume across the three plants described in the appendix.",
    "Wie hoch war die Eigenkapitalquote zum Jahresende?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="QueryTower.save_pretrained directory")
    ap.add_argument("--pkg", required=True, help="packaged directory or hub id")
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    from nanovdr import QueryTower

    tower = QueryTower.from_pretrained(args.ckpt).eval()
    reference = tower.encode(QUERIES, device="cpu")

    from sentence_transformers import MultiVectorEncoder

    model = MultiVectorEncoder(args.pkg, trust_remote_code=True, device="cpu")
    got = model.encode_query(QUERIES, convert_to_numpy=False)

    worst = 0.0
    bad = []
    for i, (a, b) in enumerate(zip(reference, got)):
        b = b.float().cpu()
        if tuple(a.shape) != tuple(b.shape):
            bad.append((i, tuple(a.shape), tuple(b.shape)))
            continue
        worst = max(worst, (a - b).abs().max().item())

    for i, sa, sb in bad:
        print(f"  query {i}: shape {sa} vs {sb}")
    norms = [float(t.norm(dim=-1).sum()) for t in got]
    print(f"token-norm sums (1.0 means the weights are folded in): "
          f"{', '.join(f'{n:.4f}' for n in norms)}")
    print(f"worst element-wise difference: {worst:.3e}  tolerance {args.tol:.0e}")
    ok = not bad and worst <= args.tol
    print("RESULT:", "EQUIVALENT" if ok else "NOT EQUIVALENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
