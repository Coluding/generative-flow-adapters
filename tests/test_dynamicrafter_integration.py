"""End-to-end smoke test against the real DynamiCrafter checkpoint and OpenCLIP.

Heavy: loads a 9.8 GB checkpoint (CPU peak ~10 GB during ``torch.load``) and
~3.9 GB of OpenCLIP weights on first run. Auto-skips when either is missing.

Override the checkpoint path with ``DYNAMICRAFTER_CKPT=/path/to/file.ckpt``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from generative_flow_adapters.data import (
    SD_VAE_DDCONFIG,
    CachedNullCaptionEncoder,
    VideoAutoencoderKL,
    encode_with_openclip,
    precompute_null_text_embedding,
)


_CKPT_PATH = Path(os.environ.get("DYNAMICRAFTER_CKPT", "ckts/dynami512.ckpt"))
_HAS_OPEN_CLIP = importlib.util.find_spec("open_clip") is not None

pytestmark = [
    pytest.mark.skipif(
        not _CKPT_PATH.exists(),
        reason=f"DynamiCrafter checkpoint not found at {_CKPT_PATH}. "
        "Set DYNAMICRAFTER_CKPT to enable.",
    ),
    pytest.mark.skipif(
        not _HAS_OPEN_CLIP,
        reason="open_clip_torch not installed; install with `pip install open_clip_torch`.",
    ),
]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def loaded_vae() -> VideoAutoencoderKL:
    """Build the DynamiCrafter VAE from the checkpoint and freeze it."""
    device = _device()
    vae = VideoAutoencoderKL(ddconfig=dict(SD_VAE_DDCONFIG), embed_dim=4).to(device)
    keys = vae.load_dynamicrafter_checkpoint(str(_CKPT_PATH), strict=False)
    assert len(keys) > 100, f"only {len(keys)} VAE keys loaded — checkpoint filter likely missed the prefix."
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()
    return vae


def test_vae_encode_decode_round_trip(loaded_vae: VideoAutoencoderKL) -> None:
    device = _device()
    # 1 clip, 3-channel RGB, 4 frames, 256x256 -> 8x downsample -> 32x32 latent
    video = torch.randn(1, 3, 4, 256, 256, device=device)
    with torch.no_grad():
        latent = loaded_vae.encode_video(video)
    assert tuple(latent.shape) == (1, 4, 4, 32, 32), latent.shape
    assert torch.isfinite(latent).all(), "VAE produced non-finite latents."
    # Latent magnitudes should be in a reasonable range — SD VAE outputs are
    # roughly O(1) after the 0.18215 scale factor.
    assert latent.abs().max().item() < 50.0, f"latent abs-max={latent.abs().max().item()} looks blown up."

    with torch.no_grad():
        recon = loaded_vae.decode_video(latent)
    assert tuple(recon.shape) == video.shape
    assert torch.isfinite(recon).all()


def test_openclip_encodes_bunch_of_texts() -> None:
    device = _device()
    prompts = [
        "",
        "Robot arm performs the task: pick up the red block",
        "A teddy bear riding a skateboard in Times Square",
        "Close-up of a hand opening a jar",
        "An aerial view of a coastal city at sunset",
    ]
    embeddings = encode_with_openclip(prompts, device=device)
    assert tuple(embeddings.shape) == (len(prompts), 77, 1024), embeddings.shape
    assert torch.isfinite(embeddings).all(), "OpenCLIP produced non-finite text embeddings."

    # Every non-empty prompt should differ from the empty-string row beyond a
    # trivial tolerance — confirms the encoder is actually attending to text,
    # not returning a constant.
    null_row = embeddings[0:1]
    diffs = (embeddings[1:] - null_row).flatten(1).abs().max(dim=1).values
    assert (diffs > 1e-3).all(), f"non-null prompts collapsed to null embedding: diffs={diffs.tolist()}"

    # Distinct prompts should produce distinct embeddings.
    for i in range(1, len(prompts)):
        for j in range(i + 1, len(prompts)):
            row_diff = (embeddings[i] - embeddings[j]).abs().max().item()
            assert row_diff > 1e-3, f"prompts {i} and {j} produced near-identical embeddings."


def test_null_prompt_matches_explicit_empty_string_encode() -> None:
    """precompute_null_text_embedding must match encode_with_openclip(['''])."""
    device = _device()
    cached = precompute_null_text_embedding(device=device)
    explicit = encode_with_openclip([""], device=device)
    assert tuple(cached.shape) == (1, 77, 1024)
    assert torch.equal(cached, explicit), "cached null prompt diverged from a direct empty-string encode."


def test_end_to_end_with_cached_null_caption(loaded_vae: VideoAutoencoderKL) -> None:
    """Full pipeline: VAE encode + null-prompt cross-attention context."""
    from generative_flow_adapters.data import (
        BatchPreprocessConfig,
        DynamiCrafterBatchPreprocessor,
    )

    device = _device()
    null_embedding = precompute_null_text_embedding(device=device, dtype=next(loaded_vae.parameters()).dtype)
    assert tuple(null_embedding.shape) == (1, 77, 1024)

    pre = DynamiCrafterBatchPreprocessor(
        loaded_vae,
        BatchPreprocessConfig(uncond_prob=0.0, context_tokens=77, context_dim=1024),
        caption_encoder=CachedNullCaptionEncoder(null_embedding),
    )
    batch = {
        "video": torch.randint(0, 255, (1, 4, 256, 256, 3), dtype=torch.uint8),
        "act": torch.randn(1, 4, 4),
        "caption": [""],
        "frame_stride": 1,
    }
    out = pre(batch, train=False)
    assert tuple(out["target"].shape) == (1, 4, 4, 32, 32)
    assert tuple(out["cond"]["context"].shape) == (1, 77, 1024)
    assert tuple(out["cond"]["concat"].shape) == (1, 4, 4, 32, 32)
    assert torch.isfinite(out["target"]).all()
    assert torch.isfinite(out["cond"]["context"]).all()
    # Context should be the cached null prompt, not zeros.
    assert out["cond"]["context"].abs().max().item() > 0.0
