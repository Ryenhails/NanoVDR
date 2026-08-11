"""Document tower: page image in, one vector out.

The model definition lives in :mod:`nanovdr.towers.modeling_doc`, which is
deliberately self-contained (it imports nothing from this package) because the
same file is shipped inside every Hub checkpoint and loaded there through
``trust_remote_code``. This module is the ergonomic wrapper around it: it
returns a :class:`~nanovdr.align.Repr` so the document tower composes with the
alignment registry and the scoring dispatch like any other tower.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from ..align import Repr
from .modeling_doc import NanoVDRDocConfig, NanoVDRDocModel, dynamic_tile

__all__ = ["DocTower", "NanoVDRDocConfig", "NanoVDRDocModel", "dynamic_tile"]


class DocTower(nn.Module):
    """Page encoder producing single vectors in a frozen teacher's space.

    The document side is single-vector in every release. A multi-vector
    document tower would put the index cost back, which is the thing these
    models exist to remove, so the head is not configurable here.

    Examples
    --------
    >>> tower = DocTower.from_pretrained("nanovdr/NanoVDR-D-HiRes")   # doctest: +SKIP
    >>> emb = tower.encode(pages)                                     # doctest: +SKIP
    """

    def __init__(self, model: NanoVDRDocModel, processor=None):
        super().__init__()
        self.model = model
        self.processor = processor

    # -- construction ------------------------------------------------------
    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> "DocTower":
        """Load a released checkpoint from the Hub or a local directory."""
        from transformers import AutoImageProcessor

        model = NanoVDRDocModel.from_pretrained(name_or_path, **kwargs)
        try:
            processor = AutoImageProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        except Exception:
            processor = AutoImageProcessor.from_pretrained(
                model.config.visual_encoder_name, trust_remote_code=True
            )
        return cls(model, processor)

    @classmethod
    def from_config(cls, config: NanoVDRDocConfig, processor=None) -> "DocTower":
        """Build an untrained tower, for training from the pretrained backbones."""
        return cls(NanoVDRDocModel(config), processor)

    # -- properties --------------------------------------------------------
    @property
    def config(self) -> NanoVDRDocConfig:
        return self.model.config

    @property
    def embed_dim(self) -> int:
        return self.model.cfg.embed_dim

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        tile_mask: Optional[torch.Tensor] = None,
    ) -> Repr:
        out = self.model(pixel_values=pixel_values, tile_mask=tile_mask)
        return Repr(vec=out.embedding)

    @torch.no_grad()
    def encode(
        self,
        images: Union["Image.Image", Sequence["Image.Image"]],  # noqa: F821
        batch_size: int = 8,
        device=None,
        processor=None,
    ) -> torch.Tensor:
        """Tile, preprocess and encode PIL pages. Returns (N, embed_dim) on CPU."""
        proc = processor or self.processor
        if proc is None:
            raise ValueError(
                "No image processor available. Pass processor=..., or construct "
                "the tower with DocTower.from_pretrained, which loads one."
            )
        return self.model.encode(images, proc, batch_size=batch_size, device=device)

    # -- persistence -------------------------------------------------------
    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path, safe_serialization=True)
        if self.processor is not None:
            self.processor.save_pretrained(path)

    def get_param_groups(self, base_lr: float, visual_lr_scale: float = 0.033):
        """Parameter groups with a reduced learning rate on the visual encoder.

        The visual encoder arrives already pretrained on document imagery and
        is the part most easily damaged, so it moves at a fraction of the rate
        used for the freshly initialised projection and the text backbone.
        """
        m = self.model
        return [
            {"params": m.visual_encoder.parameters(), "lr": base_lr * visual_lr_scale},
            {"params": m.connect.parameters(), "lr": base_lr},
            {"params": m.text_backbone.parameters(), "lr": base_lr},
            {"params": m.proj.parameters(), "lr": base_lr},
        ]
