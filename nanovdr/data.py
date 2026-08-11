"""Datasets pairing raw inputs with cached teacher targets.

Both towers train against the same shape of thing: a source Arrow dataset and
an HDF5 file of teacher embeddings whose rows line up with it. Everything that
differs between the towers (tiling and image preprocessing on one side,
tokenisation on the other) is confined to the collate function.

A mixture is a base dataset plus zero or more supplements, each with its own
target cache and an integer upsample factor. Supplements are concatenated
rather than sampled with per-domain weights: weighting a small domain equally
against a large base collapsed retrieval quality in our experiments, while
plain concatenation with a modest upsample is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .teacher import load_targets
from .tiling import dynamic_tile

__all__ = ["SourceSpec", "DocTargetDataset", "QueryTargetDataset", "doc_collate", "make_query_collate"]


@dataclass
class SourceSpec:
    """One component of a training mixture."""

    path: str
    targets: str
    upsample: int = 1
    column: Optional[str] = None


@dataclass
class _Part:
    ds: Any
    targets: torch.Tensor
    valid: np.ndarray
    column: str
    upsample: int = 1


def _load_parts(sources: Sequence[SourceSpec], embed_dim: int, default_column: str, verbose: bool) -> list[_Part]:
    from datasets import Dataset as HFDataset

    parts: list[_Part] = []
    for spec in sources:
        ds = HFDataset.load_from_disk(spec.path)
        targets, valid = load_targets(spec.targets, embed_dim=embed_dim)
        column = spec.column or default_column
        if column not in ds.column_names:
            raise ValueError(f"{spec.path}: column {column!r} not found; have {ds.column_names}")
        if len(ds) != targets.shape[0]:
            raise ValueError(
                f"{spec.path}: {len(ds)} rows but {targets.shape[0]} cached targets. "
                "The target cache must be row-aligned with its dataset."
            )
        parts.append(_Part(ds, targets, valid, column, max(1, int(spec.upsample))))
        if verbose:
            print(f"  {spec.path}: {len(valid)}/{len(ds)} usable x{spec.upsample}", flush=True)
    return parts


class _MixtureDataset(Dataset):
    """Index bookkeeping shared by both towers."""

    def __init__(self, sources: Sequence[SourceSpec], embed_dim: int, default_column: str, verbose: bool = True):
        if not sources:
            raise ValueError("Need at least one source.")
        self.embed_dim = embed_dim
        self.parts = _load_parts(sources, embed_dim, default_column, verbose)
        self.index: list[tuple[int, int]] = []
        for pi, part in enumerate(self.parts):
            for _ in range(part.upsample):
                self.index.extend((pi, int(row)) for row in part.valid)
        if verbose:
            print(f"  mixture: {len(self.index)} samples from {len(self.parts)} source(s)", flush=True)

    def __len__(self) -> int:
        return len(self.index)

    def _lookup(self, i: int):
        pi, row = self.index[i]
        part = self.parts[pi]
        target = part.targets[row]
        return part.ds[row][part.column], target


class DocTargetDataset(_MixtureDataset):
    """Page images with their cached teacher targets.

    Tiling happens here rather than in the model so the expensive part runs in
    dataloader workers, and so a batch arrives as one padded tensor.
    """

    def __init__(
        self,
        sources: Sequence[SourceSpec],
        processor,
        embed_dim: int = 4096,
        tile_min_num: int = 1,
        tile_max_num: int = 6,
        tile_max_total: Optional[int] = None,
        tile_use_thumbnail: bool = True,
        image_size: int = 448,
        column: str = "image",
        verbose: bool = True,
    ):
        super().__init__(sources, embed_dim, column, verbose)
        self.processor = processor
        self.tile_min_num = tile_min_num
        self.tile_max_num = tile_max_num
        self.tile_use_thumbnail = tile_use_thumbnail
        self.image_size = image_size
        self.tile_max_total = tile_max_total or (tile_max_num + (1 if tile_use_thumbnail else 0))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        image, target = self._lookup(i)
        tiles = dynamic_tile(
            image.convert("RGB"),
            min_num=self.tile_min_num,
            max_num=self.tile_max_num,
            image_size=self.image_size,
            use_thumbnail=self.tile_use_thumbnail,
        )[: self.tile_max_total]
        px = self.processor(images=tiles, return_tensors="pt")["pixel_values"]

        padded = torch.zeros(self.tile_max_total, 3, self.image_size, self.image_size, dtype=px.dtype)
        padded[: px.size(0)] = px
        mask = torch.zeros(self.tile_max_total, dtype=torch.bool)
        mask[: px.size(0)] = True
        return {"pixel_values": padded, "tile_mask": mask, "target": target}


def doc_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "tile_mask": torch.stack([b["tile_mask"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


class QueryTargetDataset(_MixtureDataset):
    """Query strings with their cached teacher targets."""

    def __init__(
        self,
        sources: Sequence[SourceSpec],
        embed_dim: int = 4096,
        column: str = "query",
        verbose: bool = True,
    ):
        super().__init__(sources, embed_dim, column, verbose)

    def __getitem__(self, i: int) -> dict:
        text, target = self._lookup(i)
        return {"text": text, "target": target}


def make_query_collate(tower, max_length: int = 512):
    """Collate that tokenises through the tower, so the instruction prefix and
    truncation length are whatever the tower is configured with."""

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        enc = tower.tokenize([b["text"] for b in batch])
        enc["target"] = torch.stack([b["target"] for b in batch])
        return enc

    return collate
