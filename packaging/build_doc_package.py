"""Convert the trained doc-tower checkpoint into a Hub-ready package.

Reads the raw training checkpoint (a plain state dict), rebuilds the model from
the standalone modeling file, verifies every tensor lands where it should, and
writes config + safetensors + preprocessor into an upload directory.

Weights are stored in float32 so the package loads anywhere; cast to bfloat16
at load time on Ampere or newer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_nanovdr_doc import NanoVDRDocConfig, NanoVDRDocModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="directory with doc_model.pt + config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-max-num", type=int, default=6)
    ap.add_argument("--tile-max-total", type=int, default=7)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_cfg = json.loads((ckpt_dir / "config.json").read_text())
    print(f"training config: embed_dim={train_cfg['embed_dim']} "
          f"pool={train_cfg['pool_type']} tiles<={train_cfg['tile_max_num']} "
          f"total={train_cfg['tile_max_total']}")

    cfg = NanoVDRDocConfig(
        visual_encoder_name=train_cfg["doc_visual_encoder"],
        text_backbone_name=train_cfg["doc_text_backbone"],
        embed_dim=train_cfg["embed_dim"],
        pool_type=train_cfg["pool_type"],
        pixel_shuffle_r=train_cfg.get("pixel_shuffle_r", 1),
        tile_min_num=train_cfg.get("tile_min_num", 1),
        tile_max_num=train_cfg.get("tile_max_num", args.tile_max_num),
        tile_max_total=train_cfg.get("tile_max_total", args.tile_max_total),
        tile_use_thumbnail=train_cfg.get("tile_use_thumbnail", True),
    )
    cfg.auto_map = {
        "AutoConfig": "modeling_nanovdr_doc.NanoVDRDocConfig",
        "AutoModel": "modeling_nanovdr_doc.NanoVDRDocModel",
    }

    print("building skeleton from config (no pretrained sub-weights downloaded) ...")
    model = NanoVDRDocModel(cfg)

    sd = torch.load(ckpt_dir / "doc_model.pt", map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k: v.float() for k, v in sd.items()}          # bf16 visual -> fp32, lossless

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  MISSING ({len(missing)}):    {missing[:8]}")
        print(f"  UNEXPECTED ({len(unexpected)}): {unexpected[:8]}")
        raise SystemExit("state dict does not match the model definition — refusing to package.")
    n = sum(p.numel() for p in model.parameters())
    print(f"  loaded cleanly: {len(sd)} tensors, {n/1e6:.2f}M parameters")

    model = model.float().eval()
    model.save_pretrained(out, safe_serialization=True)
    shutil.copy(HERE / "modeling_nanovdr_doc.py", out / "modeling_nanovdr_doc.py")

    # image processor: the visual encoder's own, so preprocessing matches training
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(cfg.visual_encoder_name, trust_remote_code=True)
    proc.save_pretrained(out)

    print(f"\nwrote {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:34s} {f.stat().st_size/1e6:9.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
