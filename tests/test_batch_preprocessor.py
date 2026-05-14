from __future__ import annotations

import torch

from generative_flow_adapters.data import (
    BatchPreprocessConfig,
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


def _make_batch(batch_size: int, frames: int, h: int = 32, w: int = 32, action_dim: int = 4) -> dict:
    return {
        "video": torch.randint(0, 255, (batch_size, frames, h, w, 3), dtype=torch.uint8),
        "act": torch.randn(batch_size, frames, action_dim),
        "caption": ["hello"] * batch_size,
        "frame_stride": 1,
    }


def test_preprocessor_produces_expected_shapes() -> None:
    vae = _make_vae()
    pre = DynamiCrafterBatchPreprocessor(
        vae,
        BatchPreprocessConfig(uncond_prob=0.0, context_tokens=8, context_dim=16),
    )
    out = pre(_make_batch(batch_size=2, frames=4), train=False)
    # 32x32 input with ch_mult [1, 2] -> 2x spatial downsample -> 16x16 latent.
    assert tuple(out["target"].shape) == (2, 4, 4, 16, 16)
    assert tuple(out["cond"]["concat"].shape) == (2, 4, 4, 16, 16)
    assert tuple(out["cond"]["context"].shape) == (2, 8, 16)
    assert tuple(out["cond"]["act"].shape) == (2, 4, 4)
    assert out["cond"]["fs"].tolist() == [1, 1]


def test_concat_is_first_frame_replicated() -> None:
    vae = _make_vae()
    pre = DynamiCrafterBatchPreprocessor(
        vae,
        BatchPreprocessConfig(uncond_prob=0.0, cond_frame_index=0, context_tokens=4, context_dim=8),
    )
    out = pre(_make_batch(batch_size=1, frames=3), train=False)
    z = out["target"]
    concat = out["cond"]["concat"]
    # cond_frame_index=0 -> every temporal slice of concat == z[:, :, 0]
    expected = z[:, :, 0:1].expand_as(concat)
    assert torch.allclose(concat, expected)


def test_scale_factor_is_applied() -> None:
    vae = _make_vae()
    pre = DynamiCrafterBatchPreprocessor(
        vae,
        BatchPreprocessConfig(uncond_prob=0.0, context_tokens=4, context_dim=8),
    )
    batch = _make_batch(batch_size=1, frames=2)
    out = pre(batch, train=False)
    # Re-encode without the scale factor and verify the published latent
    # is exactly scale_factor times the raw posterior mode.
    pixels = batch["video"].to(dtype=torch.float32) / 127.5 - 1.0
    pixels = pixels.permute(0, 4, 1, 2, 3).contiguous().to(out["target"].dtype)
    raw = vae.encode_video(pixels) / vae.scale_factor
    assert torch.allclose(out["target"], vae.scale_factor * raw, atol=1e-5)


def test_cfg_dropout_zeros_concat_for_image_drop() -> None:
    vae = _make_vae()
    # uncond_prob=1.0 makes the dropout windows cover the full [0, 1] range:
    #   prompt drop:  random_num < 2*uncond  -> always
    #   image  drop:  uncond <= random_num < 3*uncond -> never (no values >=1)
    # So we need a different approach: set uncond high enough that the image
    # window is reachable. With uncond_prob=0.5: image-drop window is [0.5, 1.0).
    torch.manual_seed(0)
    pre = DynamiCrafterBatchPreprocessor(
        vae,
        BatchPreprocessConfig(uncond_prob=0.5, context_tokens=4, context_dim=8),
    )
    # With seed 0 and batch_size=8, at least one sample should land in the
    # image-drop window; that sample's concat tensor must be all zeros.
    out = pre(_make_batch(batch_size=8, frames=2), train=True)
    concat = out["cond"]["concat"]
    per_sample_max = concat.flatten(1).abs().amax(dim=1)
    assert (per_sample_max == 0).any(), "expected at least one sample to have image conditioning dropped"


def test_caption_encoder_hook_is_used() -> None:
    vae = _make_vae()
    sentinel = torch.full((3, 4, 8), 7.0)

    def fake_caption_encoder(captions: list[str]) -> torch.Tensor:
        assert len(captions) == 3
        return sentinel

    pre = DynamiCrafterBatchPreprocessor(
        vae,
        BatchPreprocessConfig(uncond_prob=0.0, context_tokens=4, context_dim=8),
        caption_encoder=fake_caption_encoder,
    )
    out = pre(_make_batch(batch_size=3, frames=2), train=False)
    assert torch.equal(out["cond"]["context"], sentinel.to(out["cond"]["context"].dtype))


def test_vae_round_trip_shapes() -> None:
    vae = _make_vae()
    video = torch.randn(2, 3, 4, 32, 32)
    z = vae.encode_video(video)
    assert tuple(z.shape) == (2, 4, 4, 16, 16)
    recon = vae.decode_video(z)
    assert tuple(recon.shape) == (2, 3, 4, 32, 32)
