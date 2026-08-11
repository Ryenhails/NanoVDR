"""Convert the trained doc-tower checkpoint into a Hub-ready package.

Reads the raw training checkpoint (a plain state dict), rebuilds the model from
the standalone modeling file, verifies every tensor lands where it should, and
writes config + safetensors + preprocessor into an upload directory.

Weights are stored in float32 so the package loads anywhere; cast to bfloat16
at load time on Ampere or newer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nanovdr" / "towers"))

from modeling_doc import NanoVDRDocConfig, NanoVDRDocModel  # noqa: E402
from processing_doc import NanoVDRDocImageProcessor  # noqa: E402


def write_sentence_transformers_files(out: Path) -> None:
    """Make the package loadable as a SentenceTransformer as well.

    The same directory then serves both APIs: ``AutoModel`` for the transformers
    idiom and ``SentenceTransformer`` for the one the query towers already use,
    which is the point - a dual-tower release where the two halves are called
    differently is a release with a seam in it.

    The tower is a single module: it already pools, projects and L2-normalises,
    so there is no Pooling/Dense/Normalize stack on top and the model's own
    output is the sentence embedding. Only three small files are needed, and
    none of them requires sentence-transformers at build time.
    """
    (out / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}
    ], indent=2) + "\n")

    # "image" is how sentence-transformers labels PIL inputs; "embedding" is the
    # field of NanoVDRDocOutput; "sentence_embedding" is where encode() looks.
    (out / "sentence_bert_config.json").write_text(json.dumps({
        "transformer_task": "feature-extraction",
        "modality_config": {"image": {"method": "forward", "method_output_name": "embedding"}},
        "module_output_name": "sentence_embedding",
    }, indent=2) + "\n")

    (out / "config_sentence_transformers.json").write_text(json.dumps({
        "model_type": "SentenceTransformer",
        # Embeddings are L2-normalised, so this agrees with cosine; dot is what
        # the index actually computes.
        "similarity_fn_name": "dot",
        # modality_config landed in 5.4; older versions load the model and then
        # route pages to the text branch.
        "__version__": {"sentence_transformers": "5.4.0"},
    }, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="directory with doc_model.pt + config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-max-num", type=int, default=6)
    ap.add_argument("--tile-max-total", type=int, default=7)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_cfg = json.loads((ckpt_dir / "config.json").read_text())
    print(f"training config: embed_dim={train_cfg['embed_dim']} "
          f"pool={train_cfg['pool_type']} tiles<={train_cfg['tile_max_num']} "
          f"total={train_cfg['tile_max_total']}")

    cfg = NanoVDRDocConfig(
        visual_encoder_name=train_cfg["doc_visual_encoder"],
        text_backbone_name=train_cfg["doc_text_backbone"],
        embed_dim=train_cfg["embed_dim"],
        pool_type=train_cfg["pool_type"],
        pixel_shuffle_r=train_cfg.get("pixel_shuffle_r", 1),
        tile_min_num=train_cfg.get("tile_min_num", 1),
        tile_max_num=train_cfg.get("tile_max_num", args.tile_max_num),
        tile_max_total=train_cfg.get("tile_max_total", args.tile_max_total),
        tile_use_thumbnail=train_cfg.get("tile_use_thumbnail", True),
    )
    cfg.auto_map = {
        "AutoConfig": "modeling_nanovdr_doc.NanoVDRDocConfig",
        "AutoModel": "modeling_nanovdr_doc.NanoVDRDocModel",
    }

    print("building skeleton from config (no pretrained sub-weights downloaded) ...")
    model = NanoVDRDocModel(cfg)

    sd = torch.load(ckpt_dir / "doc_model.pt", map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k: v.float() for k, v in sd.items()}          # bf16 visual -> fp32, lossless

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  MISSING ({len(missing)}):    {missing[:8]}")
        print(f"  UNEXPECTED ({len(unexpected)}): {unexpected[:8]}")
        raise SystemExit("state dict does not match the model definition — refusing to package.")
    n = sum(p.numel() for p in model.parameters())
    print(f"  loaded cleanly: {len(sd)} tensors, {n/1e6:.2f}M parameters")

    model = model.float().eval()
    model.save_pretrained(out, safe_serialization=True)
    towers = HERE.parent / "nanovdr" / "towers"
    shutil.copy(towers / "modeling_doc.py", out / "modeling_nanovdr_doc.py")
    shutil.copy(towers / "processing_doc.py", out / "processing_nanovdr_doc.py")

    # Image processor: the visual encoder's pixel statistics, plus this
    # checkpoint's tiling. Shipping the tiling settings inside the processor is
    # what makes processor(images=pages) -> model(**inputs) correct; a stock
    # single-view processor is silently wrong here, not loudly wrong.
    from transformers import AutoImageProcessor

    base = AutoImageProcessor.from_pretrained(cfg.visual_encoder_name, trust_remote_code=True)
    proc = NanoVDRDocImageProcessor.from_image_processor(
        base,
        tile_min_num=cfg.tile_min_num,
        tile_max_num=cfg.tile_max_num,
        tile_max_total=cfg.tile_max_total,
        tile_use_thumbnail=cfg.tile_use_thumbnail,
        image_size=cfg.image_size,
    )
    # Both keys: transformers resolves AutoImageProcessor, sentence-transformers
    # goes through AutoProcessor.
    proc.auto_map = {
        "AutoImageProcessor": "processing_nanovdr_doc.NanoVDRDocImageProcessor",
        "AutoProcessor": "processing_nanovdr_doc.NanoVDRDocImageProcessor",
    }
    proc.save_pretrained(out)
    write_sentence_transformers_files(out)

    print(f"\nwrote {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:34s} {f.stat().st_size/1e6:9.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
