"""Convert a trained query-tower checkpoint into a sentence-transformers package.

The query towers ship in sentence-transformers layout so that
``SentenceTransformer(repo_id)`` just works, which is how every existing
NanoVDR query checkpoint is consumed. The tower is three modules: the text
backbone, mean pooling, and the linear projection into the teacher's space.

No instruction prefix is baked in. The instruction is applied when the teacher
targets are cached; the student is trained on raw query text and prepending
anything at inference moves its input off-distribution.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="directory with query_model.pt + config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-length", type=int, default=512)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((ckpt / "config.json").read_text())
    backbone = cfg.get("query_backbone", "distilbert/distilbert-base-uncased")
    embed_dim = int(cfg["embed_dim"])
    pool_type = cfg.get("pool_type") or "mean"
    if pool_type != "mean":
        raise SystemExit(f"only mean pooling is supported by this layout, got {pool_type!r}")
    print(f"backbone={backbone} embed_dim={embed_dim} pool={pool_type}")

    sd = torch.load(ckpt / "query_model.pt", map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    backbone_sd = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
    proj_w = sd["proj.weight"].float()
    proj_b = sd["proj.bias"].float()
    if proj_w.shape[0] != embed_dim:
        raise SystemExit(f"projection emits {proj_w.shape[0]}d but config says {embed_dim}d")
    print(f"  {len(backbone_sd)} backbone tensors, projection {tuple(proj_w.shape)}")

    from sentence_transformers import SentenceTransformer, models
    from transformers import AutoModel, AutoTokenizer

    hf = AutoModel.from_pretrained(backbone)
    missing, unexpected = hf.load_state_dict(backbone_sd, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected tensors for {backbone}: {unexpected[:6]}")
    if missing:
        print(f"  note: {len(missing)} backbone tensors kept from the pretrained init "
              f"(e.g. {missing[:3]}) -- expected for heads the tower does not train")

    tmp = out / "_backbone"
    hf.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(backbone).save_pretrained(tmp)

    word = models.Transformer(str(tmp), max_seq_length=args.max_seq_length)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    dense = models.Dense(
        in_features=word.get_word_embedding_dimension(),
        out_features=embed_dim,
        bias=True,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight.data.copy_(proj_w)
    dense.linear.bias.data.copy_(proj_b)
    norm = models.Normalize()

    st = SentenceTransformer(modules=[word, pool, dense, norm])
    st.save(str(out))
    shutil.rmtree(tmp, ignore_errors=True)

    emb = st.encode(["what was the revenue growth in Q3 2024?"], convert_to_numpy=True)
    print(f"  smoke test: {emb.shape}, norm {float((emb ** 2).sum() ** 0.5):.4f}")
    if emb.shape[-1] != embed_dim:
        raise SystemExit("packaged model does not emit the configured width")

    print(f"\nwrote {out}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {str(f.relative_to(out)):44s} {f.stat().st_size/1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
