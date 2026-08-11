"""Dynamic page tiling (InternVL-V2 partition rule).

Re-exported from the document tower's standalone image processor, which is the
single home for tiling: the same file ships inside every Hub checkpoint, so the
partition rule cannot drift between training and release.
"""

from .towers.processing_doc import IMAGE_SIZE, dynamic_tile  # noqa: F401

__all__ = ["dynamic_tile", "IMAGE_SIZE"]
