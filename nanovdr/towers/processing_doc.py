"""NanoVDR document tower — standalone image processor for the Hub.

The document tower does not consume a page as one 448x448 view. It consumes a
variable number of aspect-ratio-matched crops plus a thumbnail, zero-padded to
a fixed tile budget and accompanied by a mask saying which tiles are real. That
is preprocessing, so it belongs here rather than in the modeling file or in a
training dataset, and it is the reason a stock image processor is not
interchangeable with this one: a stock processor emits a single view and the
model, given a single view, silently produces a materially worse embedding.

``NanoVDRDocImageProcessor`` therefore emits both of the model's inputs::

    pixel_values : (B, T, 3, H, W) float32   zero-padded in normalised space
    tile_mask    : (B, T) bool               True for real tiles

which is what makes ``processor(images=pages)`` -> ``model(**inputs)`` correct,
and what lets sentence-transformers drive the tower without a custom module.

This file is self-contained: it does not import from the rest of the package,
and it is the only source of truth for tiling.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Union

import numpy as np
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import (
    center_crop,
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
)

IMAGE_SIZE = 448

# InternViT-300M-448px-V2_5's statistics. Kept explicit rather than inherited so
# that a checkpoint's preprocessing is fully described by its own config.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

__all__ = ["NanoVDRDocImageProcessor", "dynamic_tile", "IMAGE_SIZE"]


# --------------------------------------------------------------------------
# Dynamic tiling (InternVL-V2 partition rule)
# --------------------------------------------------------------------------
def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best = ratio
    return best


def dynamic_tile(
    image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = IMAGE_SIZE,
    use_thumbnail: bool = True,
) -> list:
    """Split a page into aspect-ratio-matched ``image_size`` crops.

    The grid whose aspect ratio is closest to the page's is chosen, the page is
    resized to that grid's exact pixel extent and cut into tiles. The final
    element is a whole-page thumbnail when ``use_thumbnail`` is set and more
    than one crop was produced, which is what keeps global layout available
    after the page has been cut up. Returns a list of PIL images.
    """
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        },
        key=lambda x: x[0] * x[1],
    )
    cols, rows = _closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    resized = image.resize((image_size * cols, image_size * rows))
    tiles = []
    for i in range(cols * rows):
        c, r = i % cols, i // cols
        box = (c * image_size, r * image_size, (c + 1) * image_size, (r + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


# --------------------------------------------------------------------------
# Processor
# --------------------------------------------------------------------------
class NanoVDRDocImageProcessor(BaseImageProcessor):
    """Turn page images into the document tower's exact inputs.

    ``model_input_names`` advertises ``tile_mask`` alongside ``pixel_values``,
    so frameworks that forward a processor's outputs into a model - including
    sentence-transformers, which filters kwargs against the forward signature -
    carry the mask through without special-casing this model.

    Setting ``do_tile=False`` reproduces a plain single-view processor. It is
    available for ablations and for encoders that were trained without tiling;
    it is not the right setting for any released NanoVDR document tower.
    """

    model_input_names = ["pixel_values", "tile_mask"]

    def __init__(
        self,
        do_tile: bool = True,
        tile_min_num: int = 1,
        tile_max_num: int = 6,
        tile_max_total: Optional[int] = None,
        tile_use_thumbnail: bool = True,
        image_size: int = IMAGE_SIZE,
        do_resize: bool = True,
        size: Optional[dict] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Optional[dict] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, Sequence[float]]] = None,
        image_std: Optional[Union[float, Sequence[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_tile = do_tile
        self.tile_min_num = tile_min_num
        self.tile_max_num = tile_max_num
        self.tile_use_thumbnail = tile_use_thumbnail
        self.image_size = image_size
        # The budget the model pads to. Derived from the tile settings when not
        # given, because a mismatch here is a silent shape bug at training time.
        self.tile_max_total = int(
            tile_max_total if tile_max_total is not None
            else tile_max_num + (1 if tile_use_thumbnail else 0)
        )

        self.do_resize = do_resize
        self.size = get_size_dict(size if size is not None else {"shortest_edge": image_size},
                                  default_to_square=False)
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = get_size_dict(
            crop_size if crop_size is not None else {"height": image_size, "width": image_size},
            param_name="crop_size",
        )
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = list(image_mean) if image_mean is not None else list(IMAGE_MEAN)
        self.image_std = list(image_std) if image_std is not None else list(IMAGE_STD)
        self.do_convert_rgb = do_convert_rgb

    # -- construction ------------------------------------------------------
    @classmethod
    def from_image_processor(cls, processor, **tile_kwargs) -> "NanoVDRDocImageProcessor":
        """Adopt an existing processor's pixel statistics and add tiling.

        The document tower's pixel normalisation is its visual encoder's, so the
        statistics are taken from that encoder's own processor instead of being
        restated. Everything about tiling comes from ``tile_kwargs``.
        """
        src = processor.to_dict() if hasattr(processor, "to_dict") else dict(processor)
        pixel_keys = (
            "do_resize", "size", "resample", "do_center_crop", "crop_size",
            "do_rescale", "rescale_factor", "do_normalize", "image_mean",
            "image_std", "do_convert_rgb",
        )
        kept = {k: src[k] for k in pixel_keys if k in src and src[k] is not None}
        kept.update(tile_kwargs)
        return cls(**kept)

    # -- internals ---------------------------------------------------------
    def _to_pixel_array(
        self,
        image,
        input_data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """Apply the pixel pipeline to one already-cropped view -> (3, H, W)."""
        if self.do_convert_rgb:
            image = convert_to_rgb(image)
        image = to_numpy_array(image)
        fmt = input_data_format or infer_channel_dimension_format(image)

        if self.do_resize:
            out_size = get_resize_output_image_size(
                image,
                size=(self.size["shortest_edge"] if "shortest_edge" in self.size
                      else (self.size["height"], self.size["width"])),
                default_to_square="shortest_edge" not in self.size,
                input_data_format=fmt,
            )
            image = resize(image, size=out_size, resample=self.resample, input_data_format=fmt)
        if self.do_center_crop:
            image = center_crop(
                image,
                size=(self.crop_size["height"], self.crop_size["width"]),
                input_data_format=fmt,
            )
        if self.do_rescale:
            image = self.rescale(image, scale=self.rescale_factor, input_data_format=fmt)
        if self.do_normalize:
            image = self.normalize(
                image, mean=self.image_mean, std=self.image_std, input_data_format=fmt
            )
        return to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=fmt)

    def _tiles_for(self, image) -> List:
        return dynamic_tile(
            image.convert("RGB") if hasattr(image, "convert") else image,
            min_num=self.tile_min_num,
            max_num=self.tile_max_num,
            image_size=self.image_size,
            use_thumbnail=self.tile_use_thumbnail,
        )[: self.tile_max_total]

    # -- entry point -------------------------------------------------------
    def preprocess(
        self,
        images,
        return_tensors: Optional[str] = None,
        input_data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ) -> BatchFeature:
        """Tile and preprocess pages.

        Returns ``pixel_values`` of shape ``(B, T, 3, H, W)`` zero-padded to the
        tile budget and a ``(B, T)`` boolean ``tile_mask``, or ``(B, 3, H, W)``
        and no mask when ``do_tile`` is False. Padding is zeros *after*
        normalisation, matching what the model's masked pooling expects.
        """
        if kwargs:
            unknown = [k for k in kwargs if not hasattr(self, k)]
            if unknown:
                raise TypeError(f"unexpected preprocess argument(s): {unknown}")
            # Per-call overrides act on a copy, so a processor shared between
            # dataloader workers is never mutated mid-flight.
            clone = copy.copy(self)
            for k, v in kwargs.items():
                setattr(clone, k, v)
            return clone.preprocess(
                images, return_tensors=return_tensors, input_data_format=input_data_format
            )

        images = make_list_of_images(images)

        if not self.do_tile:
            px = np.stack([self._to_pixel_array(im, input_data_format) for im in images])
            return BatchFeature(data={"pixel_values": px}, tensor_type=return_tensors)

        T = self.tile_max_total
        pixel_values, tile_mask = [], []
        for image in images:
            views = [self._to_pixel_array(t, input_data_format) for t in self._tiles_for(image)]
            n = len(views)
            if n == 0:
                raise ValueError("tiling produced no views for an input image")
            stacked = np.stack(views)
            padded = np.zeros((T, *stacked.shape[1:]), dtype=stacked.dtype)
            padded[:n] = stacked
            mask = np.zeros(T, dtype=bool)
            mask[:n] = True
            pixel_values.append(padded)
            tile_mask.append(mask)

        return BatchFeature(
            data={
                "pixel_values": np.stack(pixel_values),
                "tile_mask": np.stack(tile_mask),
            },
            tensor_type=return_tensors,
        )
