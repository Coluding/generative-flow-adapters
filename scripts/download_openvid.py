#!/usr/bin/env python
"""Download an OpenVid-1M SUBSET (nkp37/OpenVid-1M) and convert it to our
mp4 + metadata.pt schema, so the `openvid` translator reads it like ACWM/RT-1.

OpenVid = real-world CAPTIONED web video — the D3 (few-step shortcut)
in-distribution test dataset (in-distribution for the Wan/SkyReels priors,
unlike synthetic ACWM). NO actions: shortcut is action-free.

Runs in the MAIN .venv (needs huggingface_hub + decord + imageio). RUN ON A
NODE WITH INTERNET. A few thousand clips is plenty — do NOT pull the full 1M.

Output (ds/openvid/<split>/, ACWM-compatible):
  episode_{i}.mp4        RGB, subsampled to --max-frames, resized to --height/--width
  metadata.pt            list[dict] {video_path, caption, clip_id,
                                     actions[T,1] (dummy zeros), length}

Then build the per-clip text-context table:
  python scripts/precompute_clip_captions.py --data-dir ds/openvid/<split> \
    --ckpt-dir ckpts/Wan2.2-TI2V-5B --out configs/prompts/openvid_<split>.contexts.pt
"""
import argparse
import csv
import os
import zipfile

import imageio
import numpy as np
import torch

REPO = "nkp37/OpenVid-1M"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ds/openvid/train")
    ap.add_argument("--num-clips", type=int, default=2000)
    ap.add_argument("--part", type=int, default=0, help="Which OpenVid_part<N>.zip to pull videos from.")
    ap.add_argument("--max-frames", type=int, default=32, help="Subsample each clip to at most this many frames.")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from decord import VideoReader  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    # 1) captions CSV (video filename -> caption)
    csv_path = hf_hub_download(REPO, "data/train/OpenVid-1M.csv", repo_type="dataset")
    captions: dict[str, str] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            # columns vary slightly across releases: try common names
            name = row.get("video") or row.get("videoid") or row.get("video_name")
            cap = row.get("caption") or row.get("text") or ""
            if name:
                captions[os.path.basename(str(name))] = str(cap)
    print(f"loaded {len(captions)} captions from {csv_path}")

    # 2) one video part zip (large) — extract clips lazily
    zip_path = hf_hub_download(REPO, f"OpenVid_part{args.part}.zip", repo_type="dataset")
    meta = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".mp4")]
        for i, n in enumerate(names[: args.num_clips]):
            base = os.path.basename(n)
            cap = captions.get(base, "")
            tmp = os.path.join(args.out_dir, "_tmp.mp4")
            with open(tmp, "wb") as out:
                out.write(zf.read(n))
            try:
                vr = VideoReader(tmp)
                T = len(vr)
                idx = np.linspace(0, T - 1, min(T, args.max_frames)).astype(int)
                frames = vr.get_batch(list(idx)).asnumpy()  # [t,H,W,3] uint8
            except Exception as e:  # noqa: BLE001
                print(f"  skip {base}: {e}"); continue
            frames = np.stack([cv2.resize(f, (args.width, args.height)) for f in frames]).astype(np.uint8)
            vp = f"episode_{len(meta)}.mp4"
            imageio.mimwrite(os.path.join(args.out_dir, vp), list(frames), fps=8, quality=8, macro_block_size=1)
            meta.append({
                "video_path": vp,
                "caption": cap,
                "clip_id": os.path.splitext(base)[0],
                "actions": torch.zeros(len(frames), 1, dtype=torch.float32),  # dummy: action-free
                "length": int(len(frames)),
            })
            if len(meta) % 100 == 0:
                print(f"  converted {len(meta)} clips")
    if os.path.exists(os.path.join(args.out_dir, "_tmp.mp4")):
        os.remove(os.path.join(args.out_dir, "_tmp.mp4"))

    torch.save(meta, os.path.join(args.out_dir, "metadata.pt"))
    n_capt = sum(1 for e in meta if e["caption"])
    print(f"wrote {len(meta)} clips ({n_capt} with captions) + metadata.pt -> {args.out_dir}")
    print("Next: scripts/precompute_clip_captions.py to build the per-clip text-context table.")


if __name__ == "__main__":
    main()
