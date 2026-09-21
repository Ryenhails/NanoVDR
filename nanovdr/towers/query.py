"""Query tower: text in, one vector or a set of vectors out.

The query side is a plain text encoder plus an output head. Which head it
carries is the only thing that decides whether the tower is single-vector or
multi-vector, and therefore which alignment objective applies to it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..align import Repr
from ..heads import MultiVectorHead, SingleVectorHead

__all__ = ["QueryTower"]


class QueryTower(nn.Module):
    """Text encoder producing representations in a frozen teacher's space.

    Parameters
    ----------
    backbone : str
        Any HuggingFace text encoder, e.g. ``distilbert/distilbert-base-uncased``,
        ``bert-base-uncased``, ``answerdotai/ModernBERT-base``.
    embed_dim : int
        Output width; must equal the teacher's embedding dimension.
    geometry : {"single", "multi"}
        ``single`` mean-pools and projects to one vector. ``multi`` projects
        every contextual token, giving a set for late interaction.
    pool_type : {"mean", "eos"}
        Single-vector pooling. ``eos`` takes the last non-pad token, which
        matches a teacher that pools its last token. Mean pooling is the
        default because it measured better with these backbones.
    instruction : str, optional
        Prefix prepended to every query, so the student sees the same input the
        teacher was given when its targets were cached.
    learn_weights : bool
        Multi-vector only. Emit a per-token weight logit alongside each vector,
        consumed by the weighted-marginal variant of the transport objective.
        ``encode`` then folds the softmax weights into the returned vectors, so
        what comes out is what MaxSim should score.

    Notes
    -----
    A multi-vector tower drops ``[CLS]``, ``[SEP]`` and padding from its token
    set. Those positions carry no query content, and letting them into the
    measure gives transport somewhere cheap to put mass. Training and retrieval
    use the same mask, so the two cannot drift.
    """

    def __init__(
        self,
        backbone: str = "distilbert/distilbert-base-uncased",
        embed_dim: int = 4096,
        geometry: str = "single",
        pool_type: str = "mean",
        instruction: Optional[str] = None,
        learn_weights: bool = False,
        max_length: int = 512,
    ):
        super().__init__()
        if geometry not in ("single", "multi"):
            raise ValueError(f"geometry must be 'single' or 'multi', got {geometry!r}")
        if pool_type not in ("mean", "eos"):
            raise ValueError(f"pool_type must be 'mean' or 'eos', got {pool_type!r}")

        self.backbone = AutoModel.from_pretrained(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size

        self.geometry = geometry
        self.pool_type = pool_type
        self.instruction = instruction
        self.max_length = max_length
        self.head = (
            SingleVectorHead(hidden, embed_dim)
            if geometry == "single"
            else MultiVectorHead(hidden, embed_dim, learn_weights=learn_weights)
        )

    @property
    def embed_dim(self) -> int:
        return self.head.embed_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Repr:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if self.geometry == "multi":
            return self.head(hidden, self.scoring_mask(input_ids, attention_mask))

        if self.pool_type == "eos":
            idx = attention_mask.sum(dim=1).long() - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        else:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.head(pooled)

    def scoring_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Which positions are real query tokens: not padding, not a special token."""
        mask = attention_mask.bool()
        tok = self.tokenizer
        for sid in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
            if sid is not None:
                mask = mask & (input_ids != sid)
        return mask

    def tokenize(self, queries: Sequence[str], device=None) -> dict[str, torch.Tensor]:
        if self.instruction:
            queries = [f"{self.instruction}{q}" for q in queries]
        batch = self.tokenizer(
            list(queries),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    @torch.no_grad()
    def encode(
        self,
        queries: Union[str, Sequence[str]],
        batch_size: int = 64,
        device=None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Encode queries. Returns (N, D) for a single-vector tower, or a list
        of (K_i, D) token sets for a multi-vector one.

        On a weighted multi-vector tower the learned softmax weights are folded
        into the vectors here, so the per-query token norms sum to 1 and the
        output is scored by plain MaxSim. Do not re-normalise it."""
        if isinstance(queries, str):
            queries = [queries]
        device = device or next(self.parameters()).device
        self.eval()

        vecs, sets = [], []
        for start in range(0, len(queries), batch_size):
            batch = self.tokenize(queries[start : start + batch_size], device=device)
            rep = self(**batch)
            if rep.geometry == "single":
                vecs.append(rep.vec.float().cpu())
            else:
                toks, mask = rep.tokens.float().cpu(), rep.mask.cpu()
                logits = rep.weights.float().cpu() if rep.weights is not None else None
                for i in range(toks.size(0)):
                    tk = toks[i][mask[i]]
                    if logits is not None:
                        w = torch.softmax(logits[i][mask[i]], dim=0)
                        tk = tk * w.unsqueeze(-1)
                    sets.append(tk)
        return torch.cat(vecs, dim=0) if vecs else sets

    # -- persistence -------------------------------------------------------
    def save_pretrained(self, path: str) -> None:
        import json
        import os

        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.head.state_dict(), os.path.join(path, "head.pt"))
        with open(os.path.join(path, "nanovdr_query.json"), "w") as f:
            json.dump(
                {
                    "geometry": self.geometry,
                    "pool_type": self.pool_type,
                    "learn_weights": self.head.weight_head is not None
                    if self.geometry == "multi"
                    else False,
                    "embed_dim": self.embed_dim,
                    "instruction": self.instruction,
                    "max_length": self.max_length,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, path: str, **overrides) -> "QueryTower":
        import json
        import os

        with open(os.path.join(path, "nanovdr_query.json")) as f:
            cfg = json.load(f)
        cfg.update(overrides)
        tower = cls(backbone=path, **cfg)
        tower.head.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location="cpu"))
        return tower
