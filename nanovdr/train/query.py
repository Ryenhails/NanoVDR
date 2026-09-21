"""Train the query tower, single-vector or multi-vector.

    python -m nanovdr.train.query --config configs/query/distilbert_8b.yaml
    python -m nanovdr.train.query --config configs/query/ettin150m_colqwen35_otw.yaml

The geometry in the config decides everything downstream: which head the tower
carries, which objective is legal, which target cache is read, and how a batch
is collated. A single-vector run pairs fixed-width targets; a multi-vector run
pairs ragged teacher token sets.
"""

from __future__ import annotations

import argparse

from ..align import get_align_loss
from ..config import load_config
from ..data import (
    QueryTargetDataset,
    QueryTokenTargetDataset,
    SourceSpec,
    make_query_collate,
    make_query_token_collate,
)
from ..towers import QueryTower
from .engine import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    geometry = cfg.get("geometry", "single")
    tower = QueryTower(
        backbone=cfg.get("backbone", "distilbert/distilbert-base-uncased"),
        embed_dim=cfg["embed_dim"],
        geometry=geometry,
        pool_type=cfg.get("pool_type", "mean"),
        instruction=cfg.get("instruction"),
        learn_weights=cfg.get("learn_weights", False),
        max_length=cfg.get("max_length", 512),
    )

    if geometry == "multi":
        def build(specs, verbose=True):
            return QueryTokenTargetDataset([SourceSpec(**s) for s in specs], verbose=verbose)
        collate = make_query_token_collate(tower, cfg.get("max_length", 512))
    else:
        def build(specs, verbose=True):
            return QueryTargetDataset(
                [SourceSpec(**s) for s in specs], embed_dim=cfg["embed_dim"], verbose=verbose
            )
        collate = make_query_collate(tower, cfg.get("max_length", 512))

    dataset = build(cfg["sources"])
    val = build(cfg["val_sources"], verbose=False) if cfg.get("val_sources") else None

    loss_fn = get_align_loss(
        cfg.get("align", "cosine" if geometry == "single" else "ot"),
        geometry=geometry,
        **cfg.get("align_kwargs", {}),
    )
    tcfg = TrainConfig(**{k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__})

    train(
        tower,
        dataset,
        collate,
        loss_fn,
        tcfg,
        forward_fn=lambda t, b: t(input_ids=b["input_ids"], attention_mask=b["attention_mask"]),
        val_dataset=val,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
