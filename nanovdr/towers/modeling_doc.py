"""NanoVDR document tower — standalone modeling file for the Hub.

A page image is tiled, every tile is encoded by a vision transformer, the patch
tokens of all tiles are concatenated into one sequence, a bidirectional text
backbone re-encodes that sequence (no text tokens involved), and a mean pool
plus linear projection produces one L2-normalised vector in the teacher's
embedding space.

Retrieval is a dot product against a query vector from the matching NanoVDR
query tower, or against the teacher itself.

Tiling is *not* here. It lives in ``NanoVDRDocImageProcessor``, which ships
alongside this file, because it is preprocessing and because having one home for
it is what makes ``processor(images=pages)`` produce exactly the tensors this
model was trained on.

This file is self-contained: it is the only source of truth for the forward
pass and does not import from the training repository or from the processor.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

IMAGE_SIZE = 448


class NanoVDRDocConfig(PretrainedConfig):
    model_type = "nanovdr_doc"

    def __init__(
        self,
        visual_encoder_name: str = "OpenGVLab/InternViT-300M-448px-V2_5",
        text_backbone_name: str = "answerdotai/ModernBERT-base",
        embed_dim: int = 4096,
        pool_type: str = "mean",
        pixel_shuffle_r: int = 1,
        tile_min_num: int = 1,
        tile_max_num: int = 6,
        tile_max_total: int = 7,
        tile_use_thumbnail: bool = True,
        image_size: int = IMAGE_SIZE,
        use_flash_attn: bool = False,
        text_attn_implementation: str = "sdpa",
        **kwargs,
    ):
        self.visual_encoder_name = visual_encoder_name
        self.text_backbone_name = text_backbone_name
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        self.pixel_shuffle_r = pixel_shuffle_r
        self.tile_min_num = tile_min_num
        self.tile_max_num = tile_max_num
        self.tile_max_total = tile_max_total
        self.tile_use_thumbnail = tile_use_thumbnail
        self.image_size = image_size
        # InternViT ships a native flash-attention path and ModernBERT picks
        # flash-attention-2 when the package is importable. Both need Ampere or
        # newer, so the defaults here stay portable: plain attention for the
        # vision tower and sdpa for the text tower (sdpa dispatches to a fused
        # kernel on capable hardware anyway). Set use_flash_attn=True and
        # text_attn_implementation="flash_attention_2" on Ampere+ for speed.
        self.use_flash_attn = use_flash_attn
        self.text_attn_implementation = text_attn_implementation
        super().__init__(**kwargs)


@dataclass
class NanoVDRDocOutput(ModelOutput):
    embedding: Optional[torch.FloatTensor] = None


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class NanoVDRDocModel(PreTrainedModel):
    config_class = NanoVDRDocConfig
    base_model_prefix = "nanovdr_doc"
    _no_split_modules = ["InternVisionEncoderLayer", "ModernBertEncoderLayer"]

    def __init__(self, config: NanoVDRDocConfig):
        super().__init__(config)
        self.cfg = config

        vis_cfg = AutoConfig.from_pretrained(config.visual_encoder_name, trust_remote_code=True)
        if hasattr(vis_cfg, "use_flash_attn"):
            vis_cfg.use_flash_attn = config.use_flash_attn
        visual = AutoModel.from_config(vis_cfg, trust_remote_code=True)
        # Some releases wrap the tower; take the vision half when they do.
        self.visual_encoder = getattr(visual, "vision_model", visual)
        self.vis_dim = getattr(vis_cfg, "hidden_size", None) or vis_cfg.vision_config.hidden_size

        txt_cfg = AutoConfig.from_pretrained(config.text_backbone_name)
        if hasattr(txt_cfg, "reference_compile"):
            txt_cfg.reference_compile = False
        self.text_backbone = AutoModel.from_config(
            txt_cfg, attn_implementation=config.text_attn_implementation
        )
        text_dim = txt_cfg.hidden_size

        r = int(config.pixel_shuffle_r)
        self.connect = nn.Linear(self.vis_dim * r * r, text_dim)
        self.proj = nn.Linear(text_dim, config.embed_dim)
        if config.pool_type == "eos":
            self.eos_token = nn.Parameter(torch.randn(1, 1, text_dim) * 0.02)

        self.post_init()

    # -- internals ---------------------------------------------------------
    def _pixel_shuffle(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, D = patches.shape
        hw = int(round(math.sqrt(N)))
        r = int(self.cfg.pixel_shuffle_r)
        if hw * hw != N or hw % r != 0:
            return patches
        x = patches.reshape(B, hw, hw, D).reshape(B, hw // r, r, hw // r, r, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.reshape(B, (hw // r) ** 2, r * r * D)

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        tile_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """Encode pages into L2-normalised embeddings.

        Parameters
        ----------
        pixel_values : (B, T, 3, H, W) or (B, 3, H, W)
            Tiled pages, zero-padded to ``T`` tiles, or a single view per page.
        tile_mask : (B, T) bool
            True for real tiles. Required when ``pixel_values`` is 5-D.

        A 4-D input is accepted but is not how the released towers were
        trained. Downscaling a whole page to one 448px view destroys the small
        text that document retrieval turns on, and the resulting embeddings are
        substantially worse while looking perfectly well-formed, so this case
        warns rather than failing silently.
        """
        tiled = pixel_values.dim() == 5
        if tiled:
            B, T, C, H, W = pixel_values.shape
            pv = pixel_values.reshape(B * T, C, H, W)
            if tile_mask is None:
                raise ValueError("tile_mask is required for tiled (5-D) pixel_values.")
        else:
            B, T = pixel_values.size(0), 1
            pv = pixel_values
            warnings.warn(
                "NanoVDRDocModel received a single 448px view per page (4-D pixel_values) "
                "instead of tiles. Retrieval quality drops markedly. Use "
                "NanoVDRDocImageProcessor, which emits pixel_values and tile_mask together: "
                "processor = AutoImageProcessor.from_pretrained(<repo>, trust_remote_code=True)",
                UserWarning,
                stacklevel=2,
            )

        vis_dtype = next(self.visual_encoder.parameters()).dtype
        vis_out = self.visual_encoder(pixel_values=pv.to(vis_dtype))
        patches = vis_out.last_hidden_state if hasattr(vis_out, "last_hidden_state") else vis_out[0]
        patches = patches.float()

        if patches.dim() == 4:  # (B, C, H, W) spatial output
            patches = F.adaptive_avg_pool2d(patches, (16, 16)).flatten(2).transpose(1, 2)

        if int(self.cfg.pixel_shuffle_r) > 1:
            patches = self._pixel_shuffle(patches)

        n_per_tile, vis_dim = patches.size(1), patches.size(2)
        if tiled:
            patches = patches.reshape(B, T * n_per_tile, vis_dim)

        pseudo = self.connect(patches)
        max_pos = self.text_backbone.config.max_position_embeddings

        if tiled:
            attn = tile_mask.unsqueeze(-1).expand(B, T, n_per_tile).reshape(B, T * n_per_tile)
            attn = attn.to(pseudo.device).long()
        else:
            attn = torch.ones(B, pseudo.size(1), device=pseudo.device, dtype=torch.long)

        if self.cfg.pool_type == "eos":
            if pseudo.size(1) > max_pos - 1:
                pseudo, attn = pseudo[:, : max_pos - 1], attn[:, : max_pos - 1]
            eos = self.eos_token.expand(B, 1, -1).to(pseudo.dtype)
            pseudo = torch.cat([pseudo, eos], dim=1)
            attn = torch.cat([attn, torch.ones(B, 1, device=attn.device, dtype=torch.long)], dim=1)
        elif pseudo.size(1) > max_pos:
            pseudo, attn = pseudo[:, :max_pos], attn[:, :max_pos]

        text_out = self.text_backbone(inputs_embeds=pseudo, attention_mask=attn)
        if self.cfg.pool_type == "eos":
            pooled = text_out.last_hidden_state[:, -1, :]
        else:
            m = attn.unsqueeze(-1).float()
            pooled = (text_out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1)

        emb = F.normalize(self.proj(pooled), p=2, dim=-1)
        return NanoVDRDocOutput(embedding=emb) if return_dict else (emb,)

    # -- convenience -------------------------------------------------------
    @torch.no_grad()
    def encode(
        self,
        images: Union["Image.Image", Sequence["Image.Image"]],  # noqa: F821
        processor,
        batch_size: int = 8,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Batch, preprocess and encode PIL pages. Returns (N, embed_dim) on CPU.

        ``processor`` must be a ``NanoVDRDocImageProcessor``; a stock image
        processor emits one view per page and no tile mask, which this model
        would accept and quietly underperform on.
        """
        from PIL import Image  # local import; PIL is not a hard dependency of import

        if isinstance(images, Image.Image):
            images = [images]
        if not getattr(processor, "do_tile", False):
            raise ValueError(
                f"{type(processor).__name__} does not tile. The document tower needs "
                "NanoVDRDocImageProcessor, which ships with every NanoVDR document "
                "checkpoint: AutoImageProcessor.from_pretrained(<repo>, trust_remote_code=True)."
            )
        device = device or next(self.parameters()).device
        out = []
        self.eval()

        for start in range(0, len(images), batch_size):
            batch = processor(images=images[start : start + batch_size], return_tensors="pt")
            emb = self(**{k: v.to(device) for k, v in batch.items()}).embedding
            out.append(emb.float().cpu())
        return torch.cat(out, dim=0)


AutoConfig.register("nanovdr_doc", NanoVDRDocConfig)
AutoModel.register(NanoVDRDocConfig, NanoVDRDocModel)
