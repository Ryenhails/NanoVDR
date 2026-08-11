"""The two towers, plus the document tower's image processor.

Registration with the ``Auto*`` classes happens here rather than inside the
standalone modeling and processing files, so those two stay importable on their
own and free of imports of each other. That is what lets both be copied
verbatim into a Hub checkpoint.
"""

from transformers import AutoImageProcessor

from .doc import DocTower
from .modeling_doc import NanoVDRDocConfig, NanoVDRDocModel
from .processing_doc import NanoVDRDocImageProcessor
from .query import QueryTower

AutoImageProcessor.register(NanoVDRDocConfig, slow_image_processor_class=NanoVDRDocImageProcessor)

__all__ = [
    "DocTower",
    "QueryTower",
    "NanoVDRDocConfig",
    "NanoVDRDocModel",
    "NanoVDRDocImageProcessor",
]
