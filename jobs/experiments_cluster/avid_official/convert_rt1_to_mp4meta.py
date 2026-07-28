#!/usr/bin/env python
"""Convert RT-1 (fractal20220817_data, Open X-Embodiment) episodes to our
mp4 + metadata.pt schema, so the main framework's `rt1` translator can read
them exactly like ACWM data.

RUN WITH THE AVID POETRY ENV (has tfds + gcsfs, streams from GCS — no 111 GB
download needed for a small split):
  /home/lukas/.cache/pypoetry/virtualenvs/ldwma-MkWu4YH_-py3.10/bin/python \
    jobs/experiments_cluster/avid_official/convert_rt1_to_mp4meta.py \
    --split 'train[:8]' --out-dir ds/rt1/smoke

Output (matches ds/acwm-phys/.../<split>/):
  episode_{i}.mp4              RGB video (RT-1 native 256x320, uint8)
  metadata.pt                 list[dict] {video_path, actions[T,7] float32, length}

ACTION REPRESENTATION (RT-1 canonical 7-DoF end-effector delta):
  actions[:, 0:3] = observation-frame world_vector      (xyz translation delta)
  actions[:, 3:6] = rotation_delta                      (axis-angle rotation delta)
  actions[:, 6:7] = gripper_closedness_action           (gripper command)
  Per-dim STD-NORMALIZATION (2026-07-28): octo's own pipeline normalizes each of
  the 7 dims to ~unit std over the dataset before feeding the model, and a wrong
  action SCALE would confound the in-distribution action test. So we now compute
  per-dim mean/std over the converted split and normalize: a = (a - mean) / std.
  The stats are stored in metadata (`action_mean`/`action_std`) for traceability
  and to undo/redo. Pass --no-normalize to keep raw deltas (the old behaviour).

PER-CLIP TEXT (2026-07-28): each RT-1 episode ships a
``natural_language_instruction`` ("pick up the coke can") — the RIGHT caption.
We store it per episode as ``caption`` + a stable ``clip_id``; the RT1Translator
surfaces them so the per-clip text-context table conditions the base on this
episode's instruction (not a generic prompt).
"""
import argparse
import os

import imageio
import numpy as np
import tensorflow_datasets as tfds
import torch


def _instruction(step) -> str:
    """RT-1's per-episode language instruction (same across steps)."""
    obs = step.get("observation", {}) if hasattr(step, "get") else step["observation"]
    for key in ("natural_language_instruction",):
        if key in obs:
            v = obs[key].numpy()
            return v.decode("utf-8") if isinstance(v, bytes) else str(v)
    if "language_instruction" in step:
        v = step["language_instruction"].numpy()
        return v.decode("utf-8") if isinstance(v, bytes) else str(v)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="gs://gresearch/robotics",
                    help="TFDS data_dir: gs://gresearch/robotics (stream) or a local RT-1 copy.")
    ap.add_argument("--split", default="train[:8]", help="TFDS split slice, e.g. train[:8] or train.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=3, help="RT-1 is ~3 Hz; stored fps is cosmetic (decord reads frames).")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Keep raw per-step action deltas (skip per-dim std-normalization).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    builder = tfds.builder("fractal20220817_data", data_dir=args.data_dir)
    ds = builder.as_dataset(split=args.split)

    meta = []
    for i, ep in enumerate(ds):
        frames, acts = [], []
        instruction = ""
        for j, s in enumerate(ep["steps"]):
            frames.append(s["observation"]["image"].numpy())  # [256,320,3] uint8
            a = s["action"]
            wv = np.asarray(a["world_vector"]).reshape(-1)               # 3
            rd = np.asarray(a["rotation_delta"]).reshape(-1)             # 3
            gr = np.asarray(a["gripper_closedness_action"]).reshape(-1)  # 1
            acts.append(np.concatenate([wv, rd, gr]).astype(np.float32))  # 7
            if j == 0:
                instruction = _instruction(s)
        video = np.stack(frames)                                   # [T,256,320,3] uint8
        actions = torch.from_numpy(np.stack(acts).astype(np.float32))  # [T,7]
        vp = f"episode_{i}.mp4"
        imageio.mimwrite(os.path.join(args.out_dir, vp), list(video),
                         fps=args.fps, quality=8, macro_block_size=1)
        meta.append({"video_path": vp, "actions": actions, "length": int(len(frames)),
                     "caption": instruction, "clip_id": f"rt1_{i}"})
        print(f"ep {i}: T={len(frames)} video{tuple(video.shape)} act{tuple(actions.shape)} "
              f"instr={instruction!r}")

    # Per-dim std-normalization over ALL steps of ALL episodes (octo convention).
    if not args.no_normalize and meta:
        alla = torch.cat([e["actions"] for e in meta], dim=0)  # [sum_T, 7]
        mean = alla.mean(dim=0)
        std = alla.std(dim=0).clamp_min(1e-6)                  # avoid div-by-0 (gripper can be near-constant)
        for e in meta:
            e["actions"] = (e["actions"] - mean) / std
            e["action_mean"] = mean.clone()
            e["action_std"] = std.clone()
        print(f"normalized 7 dims: mean={mean.tolist()} std={std.tolist()}")
    else:
        print("NO normalization applied (--no-normalize or empty).")

    torch.save(meta, os.path.join(args.out_dir, "metadata.pt"))
    print(f"wrote {len(meta)} episodes + metadata.pt -> {args.out_dir}")


if __name__ == "__main__":
    main()
