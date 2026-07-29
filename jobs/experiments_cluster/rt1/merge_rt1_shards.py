#!/usr/bin/env python
"""Merge the sharded RT-1 conversion (submit_convert_rt1_shards.sh) into ONE
dataset dir with a single metadata.pt, applying the per-dim action
std-normalization GLOBALLY.

Why this exists: convert_rt1_to_mp4meta.py normalizes over whatever slice it was
given. Running 18 shards with normalization ON would produce 18 different action
scales — a silent confound for the action test. So the shards run
``--no-normalize`` (raw deltas) and this script does the single global pass,
byte-for-byte the same arithmetic as the converter:

    mean = all_actions.mean(0);  std = all_actions.std(0).clamp_min(1e-6)
    a <- (a - mean) / std

Two things must be rewritten while concatenating, or the merged set is corrupt:

* ``video_path`` — per-shard it is ``episode_{i}.mp4`` relative to the SHARD dir.
  Rewritten to ``shard_{k}/episode_{i}.mp4``, relative to the merged root (the
  translator resolves ``data_dir / video_path``, so subdirs are fine).
* ``clip_id``  — per-shard it restarts at ``rt1_0``, so ids COLLIDE across
  shards. The caption table (precompute_clip_captions.py) is keyed by clip_id,
  so a collision would condition many clips on the wrong instruction.
  Reassigned to a global running index.

Usage:
    python jobs/experiments_cluster/rt1/merge_rt1_shards.py --root ~/scratch-shared/rt1/full
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

_SHARD_RE = re.compile(r"^shard_(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir holding shard_0/, shard_1/, ... ")
    ap.add_argument("--expect-shards", type=int, default=None,
                    help="fail unless exactly this many shard dirs are present (catches a dead array task).")
    ap.add_argument("--expect-episodes", type=int, default=None,
                    help="fail unless the merged set has exactly this many episodes (e.g. 87212).")
    ap.add_argument("--no-normalize", action="store_true", help="concatenate only, keep raw deltas.")
    ap.add_argument("--out", default=None, help="metadata path (default: <root>/metadata.pt)")
    args = ap.parse_args()

    root = Path(args.root)
    shards = sorted(
        (p for p in root.iterdir() if p.is_dir() and _SHARD_RE.match(p.name)),
        key=lambda p: int(_SHARD_RE.match(p.name).group(1)),
    )
    if not shards:
        raise SystemExit(f"no shard_* dirs under {root}")
    if args.expect_shards is not None and len(shards) != args.expect_shards:
        raise SystemExit(f"found {len(shards)} shards, expected {args.expect_shards} — a job-array task died?")

    merged: list[dict] = []
    for sd in shards:
        mp = sd / "metadata.pt"
        if not mp.exists():
            raise SystemExit(f"{mp} missing — shard {sd.name} did not finish; re-run that array index.")
        entries = torch.load(mp, weights_only=False)
        for e in entries:
            vp = str(e["video_path"])
            if not (sd / vp).exists():
                raise SystemExit(f"{sd / vp} referenced by {mp} but not on disk")
            if "action_mean" in e or "action_std" in e:
                raise SystemExit(
                    f"{mp} is ALREADY normalized (has action_mean/std). The shards must be converted with "
                    "--no-normalize so the stats can be computed globally here."
                )
            e = dict(e)
            e["video_path"] = f"{sd.name}/{vp}"
            e["clip_id"] = f"rt1_{len(merged)}"       # global, collision-free
            merged.append(e)
        print(f"  {sd.name}: {len(entries)} episodes")

    if args.expect_episodes is not None and len(merged) != args.expect_episodes:
        raise SystemExit(f"merged {len(merged)} episodes, expected {args.expect_episodes}")

    if not args.no_normalize and merged:
        alla = torch.cat([e["actions"] for e in merged], dim=0)     # [sum_T, 7]
        mean = alla.mean(dim=0)
        std = alla.std(dim=0).clamp_min(1e-6)
        for e in merged:
            e["actions"] = (e["actions"] - mean) / std
            e["action_mean"] = mean.clone()
            e["action_std"] = std.clone()
        print(f"normalized 7 dims over {alla.shape[0]} steps: mean={mean.tolist()} std={std.tolist()}")
    else:
        print("NO normalization applied.")

    out = Path(args.out) if args.out else root / "metadata.pt"
    torch.save(merged, out)
    print(f"wrote {len(merged)} episodes -> {out}")
    print(f"Next: RT1_OUT={root} sbatch jobs/experiments_cluster/rt1/submit_precompute_rt1_latents.sh")


if __name__ == "__main__":
    main()
