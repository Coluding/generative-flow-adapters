from __future__ import annotations

import pytest
import torch

from generative_flow_adapters.data import (
    BatchPreprocessConfig,
    CachedNullCaptionEncoder,
    DynamiCrafterBatchPreprocessor,
    VideoAutoencoderKL,
)


TINY_DDCONFIG = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 32,
    "ch_mult": [1, 2],
    "num_res_blocks": 1,
    "attn_resolutions": [],
    "dropout": 0.0,
}


def _make_vae() -> VideoAutoencoderKL:
    return VideoAutoencoderKL(ddconfig=dict(TINY_DDCONFIG), embed_dim=4, sample_posterior=False)


def _make_batch(batch_size: int, frames: int) -> dict:
    return {
        "video": torch.randint(0, 255, (batch_size, frames, 32, 32, 3), dtype=torch.uint8),
        "act": torch.randn(batch_size, frames, 4),
        "caption": [""] * batch_size,
        "frame_stride": 1,
    }


def test_cached_null_caption_encoder_broadcasts_to_batch() -> None:
    null = torch.randn(1, 77, 1024)
    encoder = CachedNullCaptionEncoder(null)

    out = encoder([""] * 5)
    assert tuple(out.shape) == (5, 77, 1024)
    for i in range(5):
        assert torch.equal(out[i], null[0])


def test_cached_null_caption_encoder_returns_same_tensor_regardless_of_text() -> None:
    null = torch.randn(1, 4, 8)
    encoder = CachedNullCaptionEncoder(null)

    a = encoder(["hello", "world"])
    b = encoder(["", ""])
    assert torch.equal(a, b)


def test_cached_null_caption_encoder_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        CachedNullCaptionEncoder(torch.randn(77, 1024))
    with pytest.raises(ValueError, match="shape"):
        CachedNullCaptionEncoder(torch.randn(2, 77, 1024))


def test_preprocessor_uses_cached_null_for_context() -> None:
    vae = _make_vae()
    null = torch.full((1, 4, 8), 3.5)
    encoder = CachedNullCaptionEncoder(null)
    pre = DynamiCrafterBatchPreprocessor(
        vae,
        BatchPreprocessConfig(uncond_prob=0.0, context_tokens=4, context_dim=8),
        caption_encoder=encoder,
    )
    out = pre(_make_batch(batch_size=3, frames=2), train=False)
    context = out["cond"]["context"]
    expected = null.expand(3, -1, -1).to(context.dtype)
    assert torch.equal(context, expected)


def test_preprocessor_cfg_swap_is_noop_with_cached_null() -> None:
    # With the cached null encoder, `context` and `null_context` inside the
    # preprocessor are identical, so even with high uncond_prob the CFG
    # `torch.where` swap leaves every sample as the same null embedding.
    torch.manual_seed(0)
    vae = _make_vae()
    null = torch.full((1, 4, 8), 1.25)
    encoder = CachedNullCaptionEncoder(null)
    pre = DynamiCrafterBatchPreprocessor(
        vae,
        BatchPreprocessConfig(uncond_prob=0.5, context_tokens=4, context_dim=8),
        caption_encoder=encoder,
    )
    out = pre(_make_batch(batch_size=8, frames=2), train=True)
    context = out["cond"]["context"]
    expected = null.expand(8, -1, -1).to(context.dtype)
    assert torch.equal(context, expected)
