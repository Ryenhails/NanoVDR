"""Convert a trained multi-vector query tower into a sentence-transformers package.

ColNanoVDR ships in sentence-transformers' ``MultiVectorEncoder`` layout so that
``MultiVectorEncoder(repo_id, trust_remote_code=True)`` just works. Four modules:

    Transformer                text backbone
    Dense                      bias-free projection into the teacher's width
    Normalize                  per-token L2
    ColNanoVDRWeighting        scoring mask + learned per-token weights

The ordering is the one part worth explaining. ``Dense`` pins its output to
``mv_embeddings`` rather than overwriting ``token_embeddings``, so the
*pre-projection* hidden states survive to the last module -- the weight head reads
those, exactly as it does during training. The final module then folds the softmax
weights into the normalised vectors and publishes the result as
``token_embeddings``, which is what the encoder returns.

No instruction prefix is baked in. The instruction is applied when the teacher
targets are cached; the student is trained on raw query text.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

MODULE_SRC = Path(__file__).with_name("colnanovdr_weighting.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="QueryTower.save_pretrained directory (head.pt + nanovdr_query.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-length", type=int, default=512)
    args = ap.parse_args()

    ckpt, out = Path(args.ckpt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((ckpt / "nanovdr_query.json").read_text())
    if cfg.get("geometry") != "multi":
        raise SystemExit(f"this builder packages multi-vector towers, got {cfg.get('geometry')!r}")
    embed_dim = int(cfg["embed_dim"])
    learn_weights = bool(cfg.get("learn_weights", False))

    head = torch.load(ckpt / "head.pt", map_location="cpu", weights_only=True)
    proj_w = head["proj.weight"].float()
    if "proj.bias" in head:
        raise SystemExit("the multi-vector projection must be bias-free; this checkpoint has a bias")
    if proj_w.shape[0] != embed_dim:
        raise SystemExit(f"projection emits {proj_w.shape[0]}d but config says {embed_dim}d")
    print(f"embed_dim={embed_dim} learn_weights={learn_weights} proj={tuple(proj_w.shape)}")

    from sentence_transformers import MultiVectorEncoder, models
    from sentence_transformers.models import Dense, Normalize
    from transformers import AutoTokenizer

    from colnanovdr_weighting import ColNanoVDRWeighting  # noqa: E402

    word = models.Transformer(str(ckpt), max_seq_length=args.max_seq_length)
    hidden_dim = word.get_word_embedding_dimension()

    dense = Dense(
        in_features=hidden_dim,
        out_features=embed_dim,
        bias=False,
        activation_function=torch.nn.Identity(),
        module_input_name="token_embeddings",
        module_output_name="mv_embeddings",
    )
    dense.linear.weight.data.copy_(proj_w)
    norm = Normalize(module_input_name="mv_embeddings", module_output_name="mv_embeddings")

    tok = AutoTokenizer.from_pretrained(str(ckpt))
    special = sorted({i for i in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
                      if i is not None})
    weighting = ColNanoVDRWeighting(
        hidden_dim=hidden_dim,
        learn_weights=learn_weights,
        special_token_ids=special,
    )
    if learn_weights:
        weighting.weight_head.weight.data.copy_(head["weight_head.weight"].float())
        weighting.weight_head.bias.data.copy_(head["weight_head.bias"].float())

    model = MultiVectorEncoder(modules=[word, dense, norm, weighting],
                               similarity_fn_name="meanmaxsim")
    model.save(str(out))

    shutil.copy(MODULE_SRC, out / MODULE_SRC.name)
    mods = json.loads((out / "modules.json").read_text())
    for m in mods:
        if m["type"].endswith("ColNanoVDRWeighting"):
            m["type"] = "colnanovdr_weighting.ColNanoVDRWeighting"
    (out / "modules.json").write_text(json.dumps(mods, indent=2))

    print(f"\nwrote {out}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {str(f.relative_to(out)):44s} {f.stat().st_size/1e6:8.2f} MB")
    print("\nVerify before publishing:")
    print(f"  python packaging/verify_colnanovdr_package.py --ckpt {ckpt} --pkg {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
