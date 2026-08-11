"""Alignment-objective registry.

Every NanoVDR model is trained the same way: a student produces a
representation, a frozen teacher produces a target representation, and the
loss is a direct discrepancy between the two.  Nothing else enters the
objective -- no relevance labels, no negative sampling, no in-batch
contrastive term, no document representations at query-training time.  That
is what keeps the two towers independent: each one only ever sees its own
cached teacher targets, so they can be trained separately, in parallel, and
paired afterwards.

The only thing that varies across the family is the *geometry* of the
representation, and therefore which discrepancy applies:

    single vector   R = q in S^{d-1}              ->  cosine distance
    multi vector    R = {t_1..t_K} on S^{d-1}     ->  set discrepancy

Single-vector alignment is the one-atom special case of the multi-vector
objective, so both live in one registry rather than in two parallel loss
modules.

Adding an objective is one decorator; nothing else in the codebase changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Literal

import torch
import torch.nn as nn

Geometry = Literal["single", "multi"]

__all__ = [
    "Repr",
    "AlignLoss",
    "register_align_loss",
    "get_align_loss",
    "available_align_losses",
    "describe_align_loss",
    "build_align_loss",
]


# --------------------------------------------------------------------------
# Representation container
# --------------------------------------------------------------------------
@dataclass
class Repr:
    """A student or teacher representation, in either geometry.

    Exactly one of ``vec`` / ``tokens`` is set.  Vectors are expected to be
    L2-normalised by the tower before they reach a loss; losses do not
    normalise defensively, so a mis-normalised tower fails loudly in tests
    rather than silently degrading.

    Attributes
    ----------
    vec : (B, D) float tensor, or None
        Single-vector representation.
    tokens : (B, K, D) float tensor, or None
        Multi-vector representation.  ``K`` may differ between student and
        teacher; no token correspondence is assumed.
    mask : (B, K) bool tensor, or None
        Valid-token mask.  Required whenever ``tokens`` is set.
    weights : (B, K) float tensor, or None
        Optional per-token logits for the student measure.  Consumed only by
        objectives supporting a non-uniform marginal; ignored otherwise.
    """

    vec: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.vec is None) == (self.tokens is None):
            raise ValueError("Repr takes exactly one of `vec` or `tokens`.")
        if self.tokens is not None and self.mask is None:
            raise ValueError("Multi-vector Repr requires `mask`.")

    @property
    def geometry(self) -> Geometry:
        return "single" if self.vec is not None else "multi"

    @property
    def dim(self) -> int:
        return self.vec.shape[-1] if self.vec is not None else self.tokens.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.vec.shape[0] if self.vec is not None else self.tokens.shape[0]


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
class AlignLoss(nn.Module):
    """Base class for every alignment objective.

    Subclasses declare the geometry they operate on and implement ``compute``.
    The signature is deliberately narrow: a loss sees the student
    representation and the teacher target, and nothing else.

    ``forward`` returns a dict so training logs can report named components;
    the total is always under key ``"loss"``.
    """

    geometry: ClassVar[Geometry]

    def compute(self, student: Repr, teacher: Repr) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, student: Repr, teacher: Repr) -> dict[str, torch.Tensor]:
        if student.geometry != self.geometry:
            raise ValueError(
                f"{type(self).__name__} expects a {self.geometry}-vector student "
                f"representation, got {student.geometry}. Check the tower's head."
            )
        if teacher.geometry != self.geometry:
            raise ValueError(
                f"{type(self).__name__} expects a {self.geometry}-vector teacher "
                f"target, got {teacher.geometry}."
            )
        if student.dim != teacher.dim:
            raise ValueError(
                f"Student dim {student.dim} != teacher dim {teacher.dim}. The "
                f"output head must project to the teacher's embedding size."
            )
        out = self.compute(student, teacher)
        if "loss" not in out:
            raise RuntimeError(f"{type(self).__name__}.compute must return a 'loss' key.")
        return out


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Entry:
    cls: type[AlignLoss]
    geometry: Geometry
    summary: str


_REGISTRY: dict[str, _Entry] = {}


def register_align_loss(
    name: str,
    *,
    geometry: Geometry,
    summary: str = "",
) -> Callable[[type[AlignLoss]], type[AlignLoss]]:
    """Class decorator registering an objective under ``name``.

    ``name`` is what appears in a config file, so it is part of the public
    API: renaming one breaks saved configs.
    """

    def deco(cls: type[AlignLoss]) -> type[AlignLoss]:
        if name in _REGISTRY:
            raise ValueError(f"Alignment objective '{name}' is already registered.")
        cls.geometry = geometry
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        _REGISTRY[name] = _Entry(cls, geometry, summary or doc)
        return cls

    return deco


def get_align_loss(name: str, *, geometry: Geometry | None = None, **kwargs) -> AlignLoss:
    """Instantiate a registered objective.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"cosine"`` or ``"ot"``.
    geometry : {"single", "multi"}, optional
        If given, assert the objective matches the tower's output geometry.
        Trainers always pass this, so a doc-tower config asking for a
        multi-vector objective fails at construction rather than at the first
        backward pass.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown alignment objective '{name}'. Available: {sorted(_REGISTRY)}")
    entry = _REGISTRY[name]
    if geometry is not None and entry.geometry != geometry:
        raise ValueError(
            f"Objective '{name}' is {entry.geometry}-vector, but this tower emits "
            f"{geometry}-vector representations. Valid here: "
            f"{available_align_losses(geometry)}."
        )
    return entry.cls(**kwargs)


def available_align_losses(geometry: Geometry | None = None) -> list[str]:
    """List registry keys, optionally filtered by geometry."""
    return sorted(n for n, e in _REGISTRY.items() if geometry is None or e.geometry == geometry)


def describe_align_loss(name: str) -> str:
    e = _REGISTRY[name]
    return f"{name}  [{e.geometry}-vector]  {e.summary}"


def build_align_loss(cfg: dict) -> AlignLoss:
    """Build an objective from a config fragment::

        geometry: single
        align: cosine
        align_kwargs: {}

    ``geometry`` is written into the config by the tower, so the check in
    ``get_align_loss`` can fire.
    """
    return get_align_loss(
        cfg["align"],
        geometry=cfg["geometry"],
        **cfg.get("align_kwargs", {}),
    )
