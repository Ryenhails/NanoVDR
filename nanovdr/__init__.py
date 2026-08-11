"""NanoVDR: small retrievers for visual documents, trained by aligning
directly to a frozen vision-language teacher's embedding space.

Two towers, one method:

    QueryTower   text      -> single vector or token set
    DocTower     page image -> single vector

Both are trained with the same objective family, a direct discrepancy between
the student's representation and the teacher's cached target. The geometry of
that representation is the only thing that varies.

    >>> from nanovdr import DocTower, QueryTower, score
    >>> docs = DocTower.from_pretrained("nanovdr/NanoVDR-D-HiRes")     # doctest: +SKIP
    >>> d = docs.encode(pages)                                          # doctest: +SKIP
"""

from .align import (  # noqa: F401
    AlignLoss,
    Repr,
    available_align_losses,
    build_align_loss,
    describe_align_loss,
    get_align_loss,
    register_align_loss,
)
from .heads import MultiVectorHead, SingleVectorHead  # noqa: F401
from .scoring import dot_product, maxsim, score  # noqa: F401
from .tiling import dynamic_tile  # noqa: F401
from .towers import DocTower, QueryTower  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "DocTower",
    "QueryTower",
    "Repr",
    "AlignLoss",
    "get_align_loss",
    "available_align_losses",
    "describe_align_loss",
    "build_align_loss",
    "register_align_loss",
    "SingleVectorHead",
    "MultiVectorHead",
    "score",
    "dot_product",
    "maxsim",
    "dynamic_tile",
    "__version__",
]
