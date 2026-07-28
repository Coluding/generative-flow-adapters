#!/usr/bin/env python
"""Per-clip caption → T5 text-context table, for datasets where EVERY clip has
its own caption (OpenVid, RT-1 instructions) — unlike the fixed prompt pool that
``precompute_prompt_contexts.py`` builds.

Reads a converted dataset's ``metadata.pt`` (list[dict] with ``clip_id`` +
``caption``), umT5-encodes each unique caption, and writes the SAME table shape
``PromptContextProvider`` already consumes in its **positive/task mode**:

    { "positive": {clip_id: Tensor[L,C], ..., "__default__": Tensor[L,C]},
      "negative": Tensor[L,C], "uncond": Tensor[L,C], "text_len": int }

At train time the translator emits ``task_name = clip_id`` (see
``data/translators/{openvid,rt1}.py``) and the provider looks up
``positive[clip_id]`` — so the frozen base is conditioned on THIS clip's caption
with NO T5 at train time. No preprocessor change was needed: this reuses the
existing positive-mode path (data/wan_batch_preprocessor.py:PromptContextProvider).

Run (main .venv, has the Wan T5; encodes on CPU — slow for many captions, run once):
    python scripts/precompute_clip_captions.py \
      --data-dir ds/openvid/train --ckpt-dir ckpts/Wan2.2-TI2V-5B \
      --out configs/prompts/openvid_train.contexts.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external_repos", "Wan2.2"))
from wan.configs import WAN_CONFIGS          # noqa: E402
from wan.modules.t5 import T5EncoderModel    # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Converted dataset split dir (holds metadata.pt).")
    p.add_argument("--ckpt-dir", required=True, help="Wan ckpt dir (holds the umT5 weights).")
    p.add_argument("--task", default="ti2v-5B", choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--out", default=None, help="Output .contexts.pt (default: <data-dir>/captions.contexts.pt).")
    p.add_argument("--default-caption", default="A high quality video.",
                   help="__default__ fallback for clips with an empty/missing caption.")
    args = p.parse_args()

    meta = torch.load(os.path.join(args.data_dir, "metadata.pt"), map_location="cpu", weights_only=False)
    # clip_id -> caption (dedup: two clips may share a caption, but keys are clip_ids)
    pairs = {}
    for e in meta:
        cid = str(e.get("clip_id"))
        cap = str(e.get("caption") or "").strip()
        if cid and cid != "None":
            pairs[cid] = cap or args.default_caption
    if not pairs:
        raise ValueError(f"{args.data_dir}/metadata.pt has no clip_id/caption entries.")
    print(f"{len(pairs)} clips to encode (unique captions: {len(set(pairs.values()))})")

    cfg = WAN_CONFIGS[args.task]
    te = T5EncoderModel(
        text_len=cfg.text_len, dtype=cfg.t5_dtype, device="cpu",
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
    )

    def encode(text: str):
        return te([text], torch.device("cpu"))[0]  # [L, C]

    positive = {}
    for i, (cid, cap) in enumerate(pairs.items()):
        positive[cid] = encode(cap)
        if i % 50 == 0:
            print(f"  encoded {i}/{len(pairs)}")
    positive["__default__"] = encode(args.default_caption)

    table = {
        "positive": positive,
        "negative": encode(cfg.sample_neg_prompt or ""),
        "uncond": encode(""),
        "text_len": cfg.text_len,
    }
    out = args.out or os.path.join(args.data_dir, "captions.contexts.pt")
    torch.save(table, out)
    print(f"wrote per-clip context table ({len(positive)} entries) -> {out}")


if __name__ == "__main__":
    main()
