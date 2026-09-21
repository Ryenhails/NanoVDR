"""Output heads: the only place the two geometries differ.

A head takes the backbone's contextual states and produces a ``Repr`` in the
teacher's embedding space, L2-normalised. Swapping the head is what turns a
tower from single-vector into multi-vector; nothing else in the tower changes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .align import Repr

__all__ = ["SingleVectorHead", "MultiVectorHead"]


class SingleVectorHead(nn.Module):
    """Pooled state -> one L2-normalised vector.

    A plain linear map. An MLP was tried and did not help: against a strong
    teacher the extra capacity buys nothing, and against a weaker one it
    zero-pads the rear dimensions.
    """

    def __init__(self, hidden: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(hidden, embed_dim)

    def forward(self, pooled: torch.Tensor) -> Repr:
        return Repr(vec=F.normalize(self.proj(pooled), p=2, dim=-1))


class MultiVectorHead(nn.Module):
    """Per-token states -> a set of L2-normalised vectors.

    The projection is bias-free. A per-token bias survives L2 normalisation as
    a fixed direction every token is pulled towards, which is exactly the
    collapse balanced transport is there to prevent; the released ColNanoVDR
    towers are all trained and packaged without it.

    With ``learn_weights`` the head also emits one logit per token, taken from
    the pre-projection hidden state. Those become the student marginal in the
    weighted transport objective, letting a token carry more mass when the
    student has fewer atoms than the teacher. At retrieval time the tokens must
    then be scaled by the same softmax weights before MaxSim; ``QueryTower``
    does that for you.
    """

    def __init__(self, hidden: int, embed_dim: int, learn_weights: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(hidden, embed_dim, bias=False)
        self.weight_head = nn.Linear(hidden, 1) if learn_weights else None

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> Repr:
        tokens = F.normalize(self.proj(hidden), p=2, dim=-1)
        weights: Optional[torch.Tensor] = None
        if self.weight_head is not None:
            weights = self.weight_head(hidden).squeeze(-1)
        return Repr(tokens=tokens, mask=mask, weights=weights)
