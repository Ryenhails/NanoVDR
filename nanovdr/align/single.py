"""Single-vector alignment objectives.

One student vector, one teacher vector, both on the unit sphere. There is only
one objective here because there only needs to be one: pointwise cosine
alignment is what every released single-vector checkpoint is trained with, and
the ablations that added ranking terms on top of it are reported in the paper
as null results.
"""

from __future__ import annotations

import torch

from .registry import AlignLoss, Repr, register_align_loss

__all__ = ["CosineAlignLoss"]


@register_align_loss(
    "cosine",
    geometry="single",
    summary="1 - <student, teacher> between L2-normalised vectors",
)
class CosineAlignLoss(AlignLoss):
    """Pointwise cosine alignment.

    ``L = 1 - <f(x), T(x)>`` with both sides L2-normalised, so the loss lies in
    [0, 2] and equals 0 exactly when the student reproduces the teacher's
    direction. Nothing else enters: no negatives, no relevance labels, no
    in-batch interaction, which is what allows a batch to contain unrelated
    examples and the two towers to train apart.
    """

    def compute(self, student: Repr, teacher: Repr) -> dict[str, torch.Tensor]:
        cos = (student.vec * teacher.vec).sum(dim=-1)
        return {"loss": 1.0 - cos.mean(), "cosine": cos.mean().detach()}
