"""Retrieval scoring, dispatched on representation geometry.

Callers should not have to know whether a checkpoint is single- or
multi-vector. ``score`` looks at the ``Repr`` on each side and picks the right
routine: a dot product when both sides are single vectors, MaxSim late
interaction when the query side is a token set, and full late interaction when
the document side is a token set too.

Every MaxSim variant here averages over query tokens rather than summing. The
two differ by a per-query constant, so they rank a document list identically,
but averaging is what the released ColNanoVDR packages declare
(``similarity_fn_name="meanmaxsim"``) and keeping one convention everywhere
means a reproduction cannot silently pick the other.
"""

from __future__ import annotations

import torch

from .align import Repr

__all__ = ["score", "dot_product", "maxsim", "mean_maxsim"]


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


def mean_maxsim(
    query_tokens: torch.Tensor,
    doc_tokens: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    doc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Late interaction between query token sets and document token sets.

    ``query_tokens`` is (Q, Kq, D) and ``doc_tokens`` is (N, Kd, D). For every
    query token we take its best-matching document token, then average over the
    query tokens. This is the scoring rule the ColNanoVDR towers are evaluated
    and packaged with; the query side carries its learned weights folded in, so
    nothing is re-weighted here.
    """
    sim = torch.einsum("qkd,nmd->qnkm", query_tokens, doc_tokens)
    if doc_mask is not None:
        sim = sim.masked_fill(~doc_mask[None, :, None, :], torch.finfo(sim.dtype).min)
    best = sim.max(dim=-1).values                                  # (Q, N, Kq)
    if query_mask is None:
        return best.mean(dim=-1)
    qm = query_mask[:, None, :]
    return (best * qm).sum(dim=-1) / qm.sum(dim=-1).clamp(min=1)


def score(query: Repr, doc: Repr) -> torch.Tensor:
    """Score every query against every document, returning (Q, N)."""
    if query.geometry == "single":
        if doc.geometry != "single":
            raise ValueError(
                "A single-vector query cannot score a multi-vector document; "
                "the two sides must come from the same teacher's geometry."
            )
        return dot_product(query.vec, doc.vec)
    if doc.geometry == "single":
        return maxsim(query.tokens, doc.vec, query.mask)
    return mean_maxsim(query.tokens, doc.tokens, query.mask, doc.mask)
