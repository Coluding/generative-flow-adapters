"""Wan2.2 TI2V-5B (diffusion-forcing I2V) tests.

CPU smoke tests use the tiny Wan2.2 config + a fake VAE, so they run anywhere.
The GPU test is hardware/checkpoint-gated (real TI2V-5B + Wan2.2-VAE).

Note (zero-init head): a freshly-built Wan DiT zero-inits its head, so it emits
~zero velocity and is timestep-invariant at init. CPU tests here assert *wiring*
(masked loss fires, frozen base, the observation clamp holds) — not learned
behaviour — so they do not depend on the head being trained.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


class _FakeWan22VAE:
    """Stand-in Wan2.2-VAE: list-based encode at z_dim=48, stride (4,16,16)."""

    device = torch.device("cpu")

    def encode(self, videos):
        return [torch.randn(48, (v.shape[1] - 1) // 4 + 1, v.shape[2] // 16, v.shape[3] // 16) for v in videos]

    def decode(self, zs):
        # [48, T', h, w] -> [3, T, H, W] in [-1, 1]; T = (T'-1)*4 + 1, H = h*16.
        out = []
        for z in zs:
            _, tl, h, w = z.shape
            out.append(torch.tanh(torch.randn(3, (tl - 1) * 4 + 1, h * 16, w * 16)))
        return out


def _tiny_batch(preprocessor, batch_size=2, pixel_frames=17):
    raw = {
        "video": (torch.rand(batch_size, pixel_frames, 256, 256, 3) * 255).byte(),
        "act": torch.randn(batch_size, pixel_frames, 4),
    }
    return preprocessor(raw, train=True)


def _build_preprocessor(cond_frames=1):
    from generative_flow_adapters.data import (
        Wan22DiffusionForcingPreprocessor,
        WanBatchPreprocessConfig,
    )

    return Wan22DiffusionForcingPreprocessor(
        vae=_FakeWan22VAE(),
        config=WanBatchPreprocessConfig(target_height=256, target_width=256, timestep_scale=1000.0),
        condition_keys=("act",),
        cond_frames=cond_frames,
    )


def test_diffusion_forcing_preprocessor_builds_masked_batch():
    """Observation frame clamped clean (timestep 0); future frames noised."""
    pre = _build_preprocessor(cond_frames=1)
    batch = _tiny_batch(pre, batch_size=2, pixel_frames=17)  # 17 px -> 5 latent frames

    x_t, z0, fm, t = batch["x_t"], batch["x0"], batch["frame_mask"], batch["t"]
    assert x_t.shape == (2, 48, 5, 16, 16)
    assert fm.shape == (2, 5) and t.shape == (2, 5)
    # frame 0 = observation (mask 0), rest = predicted (mask 1).
    assert fm[0].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    # obs frame held clean; future frames noised away from z0.
    assert torch.allclose(x_t[:, :, 0], z0[:, :, 0])
    assert not torch.allclose(x_t[:, :, 1], z0[:, :, 1])
    # per-frame timestep: obs frame at 0, future frames at the same sigma*scale.
    assert torch.all(t[:, 0] == 0.0)
    assert torch.all(t[:, 1:] > 0.0)


def _tiny_experiment(disable_wandb=True):
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.training.builders import build_experiment

    cfg = load_config("configs/diffusion_wan22_i2v_metaworld.yaml")
    cfg.model.extra["wan_config_path"] = "configs/base/wan2.2_tiny.yaml"
    cfg.model.extra["dtype"] = "float32"
    cfg.model.extra["allow_missing_checkpoint"] = True
    cfg.model.pretrained_model_name_or_path = None
    cfg.training.extra["amp_dtype"] = "float32"
    if disable_wandb:
        cfg.training.extra["wandb"] = {"enable": False}
    return cfg, build_experiment(cfg)


def test_wan22_diffusion_forcing_training_step_fires():
    """End-to-end: tiny Wan2.2 base + adapter trains on a diffusion-forcing batch
    with a masked velocity loss; the frozen base receives no gradient."""
    from generative_flow_adapters.training.trainer import Trainer

    cfg, exp = _tiny_experiment()
    assert exp.model.base_model.model_type == "flow"
    trainer = Trainer(exp.model, exp.optimizer, exp.loss_fn, cfg.training)

    pre = _build_preprocessor(cond_frames=1)
    metrics = trainer.training_step(_tiny_batch(pre))

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert sum(p.numel() for p in exp.model.base_model.parameters() if p.requires_grad) == 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in exp.model.adapter.parameters())


def test_wan22_diffusion_forcing_sampler_clamps_observation():
    """The flow sampler holds the observation frame at the clean latent and
    generates the future frames; the no-mask path is unchanged."""
    from generative_flow_adapters.backbones.wan.modules.model2_2 import WanModel as Wan22Model
    from generative_flow_adapters.inference import FlowInferenceSampler
    from generative_flow_adapters.models.base.wan2_2 import Wan22DiTWrapper

    torch.manual_seed(0)
    module = Wan22Model(
        model_type="ti2v", patch_size=(1, 2, 2), text_len=512, in_dim=48, dim=64, ffn_dim=128,
        freq_dim=256, text_dim=4096, out_dim=48, num_heads=4, num_layers=2,
        window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6,
    )
    with torch.no_grad():  # un-zero the head so the model actually denoises the future
        module.head.head.weight.add_(torch.randn_like(module.head.head.weight) * 0.02)
    base = Wan22DiTWrapper(module=module, model_type="flow", prediction_type="velocity").eval()
    sampler = FlowInferenceSampler(model=base, num_train_timesteps=1000, shift=3.0, timestep_scale=1000.0)

    x0 = torch.randn(1, 48, 5, 16, 16)
    frame_mask = torch.ones(1, 5)
    frame_mask[:, 0] = 0.0
    out = sampler.sample_from_batch(
        {"target": torch.randn(1, 48, 5, 16, 16), "x0": x0, "frame_mask": frame_mask, "cond": {}},
        num_inference_steps=6,
    )
    assert out.shape == x0.shape and torch.isfinite(out).all()
    assert torch.allclose(out[:, :, 0], x0[:, :, 0], atol=1e-4)        # obs clamped
    assert not torch.allclose(out[:, :, 1], x0[:, :, 0], atol=1e-3)    # future generated

    # Wan2.1-style call (no frame_mask) still works unchanged.
    out2 = sampler.sample_from_batch({"target": torch.randn(1, 48, 5, 16, 16), "cond": {}}, num_inference_steps=4)
    assert out2.shape == x0.shape


# --------------------------------------------------------------------------- GPU

_CKPT_DIR = Path(os.environ.get("WAN22_CKPT_DIR", "ckpts/Wan2.2-TI2V-5B"))
_VAE_PTH = _CKPT_DIR / "Wan2.2_VAE.pth"
_HDF5 = Path("ds/metaworld_corner2.hdf5")


def _dit_complete(d: Path) -> bool:
    """True only when the full DiT is on disk: a single-file checkpoint, or every
    shard referenced by the safetensors index (the HF/ModelScope sharded form)."""
    if (d / "diffusion_pytorch_model.safetensors").exists():
        return True
    index = d / "diffusion_pytorch_model.safetensors.index.json"
    if not index.exists():
        return False
    import json

    shards = set(json.loads(index.read_text())["weight_map"].values())
    return bool(shards) and all((d / s).exists() for s in shards)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Wan2.2 I2V generation.")
def test_wan22_i2v_reconstructs_future_from_observation():
    """Frozen TI2V-5B, conditioned on a real first frame via diffusion forcing,
    predicts future frames that resemble ground truth far better than an
    unconditional rollout. Stores [GT | i2v | uncond] side by side."""
    import imageio.v3 as iio
    import numpy as np
    import torch.nn.functional as F

    from generative_flow_adapters.backbones.wan.modules.vae2_2 import Wan2_2_VAE
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.data import MetaWorldTranslator, TranslatedClipDataset
    from generative_flow_adapters.inference import FlowInferenceSampler
    from generative_flow_adapters.training.builders import build_experiment

    device = "cuda"
    os.makedirs("outputs", exist_ok=True)

    config = load_config("configs/diffusion_wan22_i2v_metaworld.yaml")
    config.model.pretrained_model_name_or_path = str(_CKPT_DIR)
    config.model.extra["allow_missing_checkpoint"] = False
    config.model.extra["dtype"] = "bf16"
    config.training.extra["wandb"] = {"enable": False}
    base = build_experiment(config).model.base_model.to(device).eval()

    # Pixel resolution = latent geometry × the Wan2.2-VAE spatial stride (16).
    # Driven by the config's latent_height/width so the test matches training;
    # override either with WAN22_PX=<latent_size> for a quick sweep (e.g.
    # WAN22_PX=32 -> 512 px). Wan2.2-TI2V is a 720p model, so 16 (=256 px) is far
    # below its training resolution — bump this to see clean frozen-base I2V.
    _VAE_STRIDE = 16
    lat_h = int(os.environ.get("WAN22_PX", config.model.extra.get("latent_height", 16)))
    lat_w = int(os.environ.get("WAN22_PX", config.model.extra.get("latent_width", 16)))
    px_h, px_w = lat_h * _VAE_STRIDE, lat_w * _VAE_STRIDE

    vae = Wan2_2_VAE(vae_pth=str(_VAE_PTH), device=device)
    translator = MetaWorldTranslator(str(_HDF5), caption_mode="empty")
    dataset = TranslatedClipDataset(translator, window_width=17, frame_stride=4, sampling="exhaustive")
    clip = torch.as_tensor(dataset[0]["video"]).permute(3, 0, 1, 2).float().div(127.5).sub(1.0)
    clip = F.interpolate(clip, size=(px_h, px_w), mode="bilinear", align_corners=False).to(device)
    z0 = vae.encode([clip])[0].unsqueeze(0).float()  # [1, 48, T', lat_h, lat_w]
    gt_px = vae.decode([z0[0]])[0]

    t_lat = z0.shape[2]
    frame_mask = torch.ones(1, t_lat, device=device)
    frame_mask[:, 0] = 0.0
    sampler = FlowInferenceSampler(model=base, num_train_timesteps=1000, shift=5.0, timestep_scale=1000.0, amp_dtype=torch.bfloat16)
    noise = torch.randn_like(z0)

    i2v = sampler.sample(shape=tuple(z0.shape), cond={}, device=z0.device, dtype=torch.bfloat16,
                         num_inference_steps=30, initial_sample=noise, frame_mask=frame_mask, obs_latent=z0)
    uncond = sampler.sample(shape=tuple(z0.shape), cond={}, device=z0.device, dtype=torch.bfloat16,
                            num_inference_steps=30, initial_sample=noise)
    i2v_px, uncond_px = vae.decode([i2v[0]])[0], vae.decode([uncond[0]])[0]

    # Compare the PREDICTED (future) frames only — the obs frame is given.
    fut = slice(1, None)
    def fut_px(latent_px):
        return latent_px[:, (1 * 4):]  # latent frame 1 -> pixel frame 4 onward
    mse_i2v = (fut_px(i2v_px).float() - fut_px(gt_px).float()).pow(2).mean().item()
    mse_unc = (fut_px(uncond_px).float() - fut_px(gt_px).float()).pow(2).mean().item()
    print(f"\n[wan22-i2v] MSE(i2v,gt)={mse_i2v:.4f}  MSE(uncond,gt)={mse_unc:.4f}  ratio={mse_i2v/max(mse_unc,1e-6):.2f}")

    def to_u8(v):
        return v.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 3, 0).cpu().numpy()
    panel = np.concatenate([to_u8(gt_px), to_u8(i2v_px), to_u8(uncond_px)], axis=2)
    iio.imwrite("outputs/test_wan22_i2v_vs_gt.mp4", panel, fps=8)
    assert os.path.exists("outputs/test_wan22_i2v_vs_gt.mp4")

    assert mse_i2v < 0.6 * mse_unc, (
        f"observation conditioning did not steer the future toward GT "
        f"(i2v={mse_i2v:.4f} vs uncond={mse_unc:.4f})."
    )
