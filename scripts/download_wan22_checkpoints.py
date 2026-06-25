"""Download Wan2.2 checkpoints from the Hugging Face Hub.

Pulls a Wan2.2 model repo into a local ``ckpts/<name>`` directory laid out the
way the training scripts expect (see ``scripts/train_wan22_i2v_metaworld.py``):

    ckpts/Wan2.2-TI2V-5B/
      Wan2.2_VAE.pth                                  # Wan2.2-VAE (48-ch, stride (4,16,16))
      diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors  # DiT shards
      diffusion_pytorch_model.safetensors.index.json
      models_t5_umt5-xxl-enc-bf16.pth                 # T5 text encoder
      google/umt5-xxl/...                             # T5 tokenizer
      config.json

The DiT ships **sharded**; the Wan wrapper's loader merges every ``*.safetensors``
in the directory (``models/base/wan.py:_load_state_dict``), so pointing the base
at this directory loads the real weights with no extra merge step.

Examples
--------
    # Default: the TI2V-5B world-model base used by the diffusion-forcing config.
    python scripts/download_wan22_checkpoints.py

    # Skip the docs assets/examples images (still gets VAE + DiT + T5).
    python scripts/download_wan22_checkpoints.py --essential-only

    # A different variant / output root.
    python scripts/download_wan22_checkpoints.py --model t2v-a14b --output-dir ckpts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Friendly aliases -> Hugging Face repo ids. TI2V-5B is the unified
# text+image-to-video model this repo trains against (provider ``wan2.2``).
MODELS: dict[str, str] = {
    "ti2v-5b": "Wan-AI/Wan2.2-TI2V-5B",
    "t2v-a14b": "Wan-AI/Wan2.2-T2V-A14B",
    "i2v-a14b": "Wan-AI/Wan2.2-I2V-A14B",
    "s2v-14b": "Wan-AI/Wan2.2-S2V-14B",
    "animate-14b": "Wan-AI/Wan2.2-Animate-14B",
}

# Heavy/irrelevant-for-training paths to drop with --essential-only.
_NONESSENTIAL_PATTERNS = ["assets/*", "examples/*", "*.png", "*.jpg", "*.JPG"]

# Files the diffusion-forcing training path actually needs present.
_REQUIRED = ("Wan2.2_VAE.pth",)
_REQUIRED_GLOBS = ("diffusion_pytorch_model*.safetensors",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="ti2v-5b",
        choices=sorted(MODELS),
        help="Wan2.2 variant to download (default: ti2v-5b).",
    )
    parser.add_argument(
        "--output-dir",
        default="ckpts",
        help="Root directory for checkpoints; the repo lands in <output-dir>/<repo-name> (default: ckpts).",
    )
    parser.add_argument(
        "--essential-only",
        action="store_true",
        help="Skip docs assets/examples images (still downloads VAE + DiT + T5).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision / branch / tag to pin (default: main).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is required. Install it with:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    repo_id = MODELS[args.model]
    local_dir = Path(args.output_dir) / repo_id.split("/", 1)[-1]
    local_dir.mkdir(parents=True, exist_ok=True)

    ignore_patterns = _NONESSENTIAL_PATTERNS if args.essential_only else None
    print(f"Downloading {repo_id} -> {local_dir}")
    if ignore_patterns:
        print(f"  (skipping: {', '.join(ignore_patterns)})")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        revision=args.revision,
        ignore_patterns=ignore_patterns,
        resume_download=True,
    )

    _verify(local_dir)
    print(f"\nDone. Point training at this directory, e.g.:\n    --ckpt-dir {local_dir}")
    return 0


def _verify(local_dir: Path) -> None:
    """Warn (don't fail) if the training-critical files are missing."""
    missing = [name for name in _REQUIRED if not (local_dir / name).exists()]
    for pattern in _REQUIRED_GLOBS:
        if not list(local_dir.glob(pattern)):
            missing.append(pattern)
    if missing:
        print(f"\nWARNING: expected file(s) not found in {local_dir}: {', '.join(missing)}", file=sys.stderr)
    else:
        shards = sorted(p.name for p in local_dir.glob("diffusion_pytorch_model*.safetensors"))
        print(f"\nVerified: Wan2.2_VAE.pth + DiT ({len(shards)} shard(s)) present.")


if __name__ == "__main__":
    raise SystemExit(main())
