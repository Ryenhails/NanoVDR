"""Dynamic page tiling (InternVL-V2 partition rule).

Re-exported from the document tower's standalone modeling file so that the
Hub copy and the package share one implementation.
"""

from .towers.modeling_doc import IMAGE_SIZE, dynamic_tile  # noqa: F401

__all__ = ["dynamic_tile", "IMAGE_SIZE"]
