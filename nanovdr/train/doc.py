"""Train the document tower.

    torchrun --nproc_per_node=2 -m nanovdr.train.doc --config configs/doc/hires_8b.yaml
"""

from __future__ import annotations

import argparse

from transformers import AutoImageProcessor

from ..align import get_align_loss
from ..config import load_config
from ..data import DocTargetDataset, SourceSpec, doc_collate
from ..towers import DocTower
from ..towers.modeling_doc import NanoVDRDocConfig
from .engine import TrainConfig, train


def build_sources(cfg: dict) -> list[SourceSpec]:
    return [SourceSpec(**s) for s in cfg["sources"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    model_cfg = NanoVDRDocConfig(
        visual_encoder_name=cfg.get("visual_encoder", "OpenGVLab/InternViT-300M-448px-V2_5"),
        text_backbone_name=cfg.get("text_backbone", "answerdotai/ModernBERT-base"),
        embed_dim=cfg["embed_dim"],
        pool_type=cfg.get("pool_type", "mean"),
        tile_min_num=cfg.get("tile_min_num", 1),
        tile_max_num=cfg.get("tile_max_num", 6),
        tile_max_total=cfg.get("tile_max_total", 7),
        tile_use_thumbnail=cfg.get("tile_use_thumbnail", True),
        use_flash_attn=cfg.get("use_flash_attn", True),
        text_attn_implementation=cfg.get("text_attn_implementation", "sdpa"),
    )
    processor = AutoImageProcessor.from_pretrained(model_cfg.visual_encoder_name, trust_remote_code=True)
    tower = DocTower.from_config(model_cfg, processor)

    dataset = DocTargetDataset(
        build_sources(cfg),
        processor=processor,
        embed_dim=cfg["embed_dim"],
        tile_min_num=model_cfg.tile_min_num,
        tile_max_num=model_cfg.tile_max_num,
        tile_max_total=model_cfg.tile_max_total,
        tile_use_thumbnail=model_cfg.tile_use_thumbnail,
        image_size=model_cfg.image_size,
    )
    val = None
    if cfg.get("val_sources"):
        val = DocTargetDataset(
            [SourceSpec(**s) for s in cfg["val_sources"]],
            processor=processor,
            embed_dim=cfg["embed_dim"],
            tile_min_num=model_cfg.tile_min_num,
            tile_max_num=model_cfg.tile_max_num,
            tile_max_total=model_cfg.tile_max_total,
            tile_use_thumbnail=model_cfg.tile_use_thumbnail,
            image_size=model_cfg.image_size,
            verbose=False,
        )

    loss_fn = get_align_loss(cfg.get("align", "cosine"), geometry="single", **cfg.get("align_kwargs", {}))
    tcfg = TrainConfig(**{k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__})

    train(
        tower,
        dataset,
        doc_collate,
        loss_fn,
        tcfg,
        forward_fn=lambda t, b: t(pixel_values=b["pixel_values"], tile_mask=b["tile_mask"]),
        param_groups=tower.get_param_groups(tcfg.peak_lr, cfg.get("visual_lr_scale", 0.033)),
        val_dataset=val,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
