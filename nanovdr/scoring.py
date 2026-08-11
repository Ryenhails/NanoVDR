"""Retrieval scoring, dispatched on representation geometry.

Callers should not have to know whether a checkpoint is single- or
multi-vector. ``score`` looks at the ``Repr`` and picks the right routine:
a dot product when both sides are single vectors, MaxSim late interaction when
the query side is a token set.
"""

from __future__ import annotations

import torch

from .align import Repr

__all__ = ["score", "dot_product", "maxsim"]


def dot_product(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """(Q, D) x (N, D) -> (Q, N). Both sides must be L2-normalised."""
    return query @ doc.T


def maxsim(
    query_tokens: torch.Tensor,
    doc: torch.Tensor,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Late interaction between query token sets and single document vectors.

    ``query_tokens`` is (Q, K, D), ``doc`` is (N, D). For every query token we
    take its best-matching document and sum, which is the ColBERT scoring rule
    specialised to a single-vector document side.
    """
    sim = torch.einsum("qkd,nd->qkn", query_tokens, doc)
    if query_mask is not None:
        sim = sim.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        denom = query_mask.sum(1, keepdim=True).clamp(min=1)
        return sim.sum(dim=1) / denom
    return sim.mean(dim=1)


def score(query: Repr, doc: Repr) -> torch.Tensor:
    """Score every query against every document, returning (Q, N)."""
    if doc.geometry != "single":
        raise ValueError(
            "The document side is single-vector in every NanoVDR release; "
            f"got a {doc.geometry}-vector document representation."
        )
    if query.geometry == "single":
        return dot_product(query.vec, doc.vec)
    return maxsim(query.tokens, doc.vec, query.mask)
