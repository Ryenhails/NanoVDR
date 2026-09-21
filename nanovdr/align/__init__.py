"""Alignment objectives: one registry, both geometries.

Importing this package registers every shipped objective, so
``get_align_loss("ot")`` works without importing the module that defines it.
"""

from .registry import (  # noqa: F401
    AlignLoss,
    Repr,
    available_align_losses,
    build_align_loss,
    describe_align_loss,
    get_align_loss,
    register_align_loss,
)
from . import single as _single  # noqa: F401,E402  (registers "cosine")
from . import multi as _multi    # noqa: F401,E402  (registers "ot")

__all__ = [
    "AlignLoss",
    "Repr",
    "available_align_losses",
    "build_align_loss",
    "describe_align_loss",
    "get_align_loss",
    "register_align_loss",
]
