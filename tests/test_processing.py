"""The document processor must reproduce the training preprocessing exactly.

The tower was trained on tensors produced by one specific sequence: tile the
page, run each tile through the visual encoder's own image processor, zero-pad
to the tile budget in normalised space. Moving that sequence into a processor is
only safe if the tensors are bit-identical, so that is what these tests assert
rather than checking shapes or approximate closeness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from nanovdr.towers.processing_doc import NanoVDRDocImageProcessor, dynamic_tile

TILE_KW = dict(tile_min_num=1, tile_max_num=6, tile_max_total=7, tile_use_thumbnail=True, image_size=448)

# Aspect ratios chosen to land on different grids: square, portrait A4, wide
# slide, extreme panorama (which overflows the budget and must be truncated),
# and a small image that produces a single tile with no thumbnail.
SHAPES = [(800, 800), (1240, 1754), (1920, 1080), (4000, 400), (300, 300)]


def _pages(seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        for (w, h) in SHAPES
    ]


@pytest.fixture(scope="module")
def reference_processor():
    """The visual encoder's own processor, i.e. what training actually used."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(
        "OpenGVLab/InternViT-300M-448px-V2_5", trust_remote_code=True
    )


def _legacy(image, processor, **kw):
    """The preprocessing as it was written, inline, before the processor existed."""
    tiles = dynamic_tile(
        image.convert("RGB"),
        min_num=kw["tile_min_num"],
        max_num=kw["tile_max_num"],
        image_size=kw["image_size"],
        use_thumbnail=kw["tile_use_thumbnail"],
    )[: kw["tile_max_total"]]
    px = processor(images=tiles, return_tensors="pt")["pixel_values"]
    padded = torch.zeros(kw["tile_max_total"], 3, kw["image_size"], kw["image_size"], dtype=px.dtype)
    padded[: px.size(0)] = px
    mask = torch.zeros(kw["tile_max_total"], dtype=torch.bool)
    mask[: px.size(0)] = True
    return padded, mask


def test_matches_legacy_preprocessing_bit_for_bit(reference_processor):
    proc = NanoVDRDocImageProcessor.from_image_processor(reference_processor, **TILE_KW)
    pages = _pages()
    out = proc(images=pages, return_tensors="pt")

    for i, page in enumerate(pages):
        want_px, want_mask = _legacy(page, reference_processor, **TILE_KW)
        assert torch.equal(out["tile_mask"][i], want_mask), f"mask differs on page {i}"
        assert torch.equal(out["pixel_values"][i], want_px), (
            f"pixel_values differ on page {i} "
            f"(max abs {(out['pixel_values'][i] - want_px).abs().max():.3e})"
        )


def test_adopts_the_encoders_pixel_statistics(reference_processor):
    proc = NanoVDRDocImageProcessor.from_image_processor(reference_processor, **TILE_KW)
    assert proc.image_mean == list(reference_processor.image_mean)
    assert proc.image_std == list(reference_processor.image_std)
    assert proc.rescale_factor == reference_processor.rescale_factor


def test_advertises_both_model_inputs():
    proc = NanoVDRDocImageProcessor(**TILE_KW)
    assert proc.model_input_names == ["pixel_values", "tile_mask"]
    out = proc(images=_pages()[:2], return_tensors="pt")
    assert set(out.keys()) == {"pixel_values", "tile_mask"}
    assert out["pixel_values"].shape == (2, 7, 3, 448, 448)
    assert out["tile_mask"].shape == (2, 7)
    assert out["tile_mask"].dtype == torch.bool


def test_padding_is_zero_and_masked_out():
    proc = NanoVDRDocImageProcessor(**TILE_KW)
    out = proc(images=[Image.new("RGB", (300, 300), "white")], return_tensors="pt")
    mask = out["tile_mask"][0]
    assert mask.sum() == 1, "a small square page is one tile with no thumbnail"
    assert torch.equal(out["pixel_values"][0][~mask], torch.zeros_like(out["pixel_values"][0][~mask]))


def test_tile_budget_truncates_rather_than_overflows():
    proc = NanoVDRDocImageProcessor(**{**TILE_KW, "tile_max_total": 3})
    out = proc(images=[Image.new("RGB", (4000, 400))], return_tensors="pt")
    assert out["pixel_values"].shape[1] == 3
    assert out["tile_mask"][0].all()


def test_single_image_and_list_agree():
    proc = NanoVDRDocImageProcessor(**TILE_KW)
    page = _pages()[1]
    a = proc(images=page, return_tensors="pt")["pixel_values"]
    b = proc(images=[page], return_tensors="pt")["pixel_values"]
    assert torch.equal(a, b)
    assert a.shape[0] == 1


def test_per_call_override_does_not_mutate_the_processor():
    proc = NanoVDRDocImageProcessor(**TILE_KW)
    page = _pages()[1]
    before = proc(images=page, return_tensors="pt")["pixel_values"]
    proc(images=page, return_tensors="pt", tile_max_total=2)
    after = proc(images=page, return_tensors="pt")["pixel_values"]
    assert torch.equal(before, after)
    assert proc.tile_max_total == 7


def test_untiled_mode_emits_a_single_view():
    proc = NanoVDRDocImageProcessor(**{**TILE_KW, "do_tile": False})
    out = proc(images=_pages()[:2], return_tensors="pt")
    assert set(out.keys()) == {"pixel_values"}
    assert out["pixel_values"].shape == (2, 3, 448, 448)


def test_training_dataset_yields_what_it_used_to(reference_processor):
    """The training path preprocesses through the processor now; same tensors."""
    from nanovdr.data import DocTargetDataset

    ds = DocTargetDataset.__new__(DocTargetDataset)      # skip the Arrow/HDF5 mixture
    ds.processor = NanoVDRDocImageProcessor.from_image_processor(reference_processor, **TILE_KW)
    page, target = _pages()[1], torch.zeros(4096)
    ds._lookup = lambda i: (page, target)

    item = ds[0]
    want_px, want_mask = _legacy(page, reference_processor, **TILE_KW)
    assert torch.equal(item["pixel_values"], want_px)
    assert torch.equal(item["tile_mask"], want_mask)
    assert item["pixel_values"].shape == (7, 3, 448, 448)


def test_the_doc_tower_upgrades_a_plain_processor(reference_processor):
    """Training configs and older checkpoints both hand over a single-view one."""
    from nanovdr.towers.doc import doc_processor_for
    from nanovdr.towers.modeling_doc import NanoVDRDocConfig

    cfg = NanoVDRDocConfig(tile_max_num=2, tile_max_total=3)
    assert not getattr(reference_processor, "do_tile", False)

    upgraded = doc_processor_for(cfg, reference_processor)
    assert upgraded.do_tile and upgraded.tile_max_total == 3
    assert upgraded.image_mean == list(reference_processor.image_mean)
    assert doc_processor_for(cfg, upgraded) is upgraded, "already-tiling processors pass through"


def test_roundtrips_through_save_and_load(tmp_path, reference_processor):
    proc = NanoVDRDocImageProcessor.from_image_processor(reference_processor, **TILE_KW)
    proc.save_pretrained(tmp_path)
    reloaded = NanoVDRDocImageProcessor.from_pretrained(tmp_path)

    page = _pages()[1]
    a = proc(images=page, return_tensors="pt")
    b = reloaded(images=page, return_tensors="pt")
    assert torch.equal(a["pixel_values"], b["pixel_values"])
    assert torch.equal(a["tile_mask"], b["tile_mask"])
    assert reloaded.tile_max_total == 7 and reloaded.do_tile is True
