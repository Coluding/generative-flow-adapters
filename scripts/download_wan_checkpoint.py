"""Prefetch the Wan2.1 DiT + VAE checkpoint on a node with internet.

Compute nodes on the cluster have no internet, so the training script cannot
download weights at run time. Run this once on the **login node** before
submitting the SLURM job:

    source .venv/bin/activate
    python scripts/download_wan_checkpoint.py --ckpt-dir ckpts/Wan2.1-T2V-1.3B

It fetches only the two files the trainer loads (the DiT safetensors and the
Wan-VAE), not the multi-GB T5 text encoder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_wan_shortcut_metaworld import WAN_REPO_ID, ensure_wan_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default="ckpts/Wan2.1-T2V-1.3B")
    parser.add_argument("--repo-id", default=WAN_REPO_ID)
    args = parser.parse_args()

    ensure_wan_checkpoint(Path(args.ckpt_dir), repo_id=args.repo_id)


if __name__ == "__main__":
    main()
