"""GPU integration tests for Wan2.1 generation: real DiT load + Wan-VAE
frame-conditioning (SDEdit) encode/decode.

Heavy and hardware-gated: needs CUDA and the downloaded checkpoint. Auto-skips
otherwise. Override the checkpoint dir with ``WAN_CKPT_DIR=/path/to/ckpt``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

_CKPT_DIR = Path(os.environ.get("WAN_CKPT_DIR", "ckpts/Wan2.1-T2V-1.3B"))
_VAE_PTH = _CKPT_DIR / "Wan2.1_VAE.pth"
_DIT_SAFETENSORS = _CKPT_DIR / "diffusion_pytorch_model.safetensors"
_HDF5 = Path("ds/metaworld_corner2.hdf5")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Wan generation."),
    pytest.mark.skipif(not _VAE_PTH.exists(), reason=f"Wan-VAE not found at {_VAE_PTH}. Set WAN_CKPT_DIR."),
]


def _load_first_metaworld_frame():
    from generative_flow_adapters.data import MetaWorldTranslator, TranslatedClipDataset

    translator = MetaWorldTranslator(str(_HDF5), caption_mode="empty")
    dataset = TranslatedClipDataset(translator, window_width=8, frame_stride=4, sampling="exhaustive")
    return torch.as_tensor(dataset[0]["video"])[0]  # [H, W, 3] uint8


@pytest.mark.skipif(not _HDF5.exists(), reason=f"MetaWorld HDF5 not found at {_HDF5}.")
def test_wan_vae_frame_condition_roundtrip():
    """Encode a MetaWorld frame to a 16-ch Wan latent and decode it back.

    Validates the frame-conditioning encode path: latent geometry matches the
    (4,8,8) stride, and the decode reconstructs the frame's structure (low error
    vs. the input — far below the ~0.5 you'd get from an unrelated/noise frame).
    """
    import torch.nn.functional as F

    from generative_flow_adapters.backbones.wan.modules.vae import WanVAE

    device = "cuda"
    frames, height, width = 33, 256, 256
    frame = _load_first_metaworld_frame()
    img = frame.permute(2, 0, 1).float().div(127.5).sub(1.0)
    img = F.interpolate(img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
    video = img.unsqueeze(1).repeat(1, frames, 1, 1).to(device)  # [3, F, H, W] in [-1, 1]

    vae = WanVAE(vae_pth=str(_VAE_PTH), device=device)
    z0 = vae.encode([video])[0]
    assert z0.shape == (16, (frames - 1) // 4 + 1, height // 8, width // 8)

    recon = vae.decode([z0])[0]  # [3, F, H, W] in [-1, 1]
    assert recon.shape == video.shape
    err = (recon.float() - video.float()).abs().mean().item()
    assert err < 0.2, f"VAE roundtrip error too high ({err:.3f}); frame not reconstructed"


@pytest.mark.skipif(not _HDF5.exists(), reason=f"MetaWorld HDF5 not found at {_HDF5}.")
@pytest.mark.skipif(not _DIT_SAFETENSORS.exists(), reason="Wan DiT safetensors not found.")
def test_wan_frame_conditioned_base_vs_adapter():
    """Frame-conditioned (SDEdit) generation through base vs. AdaptedModel.

    Encodes a MetaWorld frame, denoises the schedule tail through (a) the frozen
    Wan base and (b) the full AdaptedModel (base + zero-init output adapter),
    stores both decoded videos + a comparison strip, and asserts the two latent
    trajectories match — the untrained adapter contributes no delta, so the
    frame-conditioned rollout is identical. Uses null text context (no T5) and a
    small/short setting to stay fast.
    """
    import numpy as np
    import imageio.v3 as iio
    import torch.nn.functional as F

    from generative_flow_adapters.backbones.wan.modules.vae import WanVAE
    from generative_flow_adapters.backbones.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from generative_flow_adapters.backbones.wan.utils.utils import cache_video
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.training.builders import build_experiment

    device, dtype = "cuda", torch.bfloat16
    frames, height, width, steps, strength = 17, 128, 128, 6, 0.6
    os.makedirs("outputs", exist_ok=True)

    # One DiT copy: the AdaptedModel holds the base; reuse base_model for path (a).
    config = load_config("configs/wan21/diffusion_wan_shortcut_metaworld.yaml")
    config.model.pretrained_model_name_or_path = str(_CKPT_DIR)
    config.model.extra["allow_missing_checkpoint"] = False
    config.model.extra["dtype"] = "bf16"
    model = build_experiment(config).model.to(device).eval()

    # Frame -> tiled video -> Wan-VAE latent (SDEdit anchor).
    vae = WanVAE(vae_pth=str(_VAE_PTH), device=device)
    frame = _load_first_metaworld_frame()
    img = frame.permute(2, 0, 1).float().div(127.5).sub(1.0)
    img = F.interpolate(img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
    video_in = img.unsqueeze(1).repeat(1, frames, 1, 1).to(device)
    z0 = vae.encode([video_in])[0].unsqueeze(0).float()

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)

    def sample(forward):
        sched = scheduler  # fresh schedule per run (stateful step_index)
        sched.set_timesteps(steps, device=device, shift=3.0)
        start_idx = min(int(round((1.0 - strength) * steps)), steps - 1)
        gen = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn(1, 16, *z0.shape[2:], generator=gen, device=device, dtype=torch.float32)
        sigma_start = sched.sigmas[start_idx].to(device)
        latent = ((1.0 - sigma_start) * z0 + sigma_start * noise).float()
        for t in sched.timesteps[start_idx:]:
            with torch.autocast("cuda", dtype=dtype):
                v = forward(latent, t.unsqueeze(0).to(device))
            latent = sched.step(v.float(), t, latent, return_dict=False, generator=gen)[0]
        return latent

    z = torch.zeros(1, 4, device=device)
    latent_base = sample(lambda x, t: model.base_model(x, t, {}))
    latent_adapter = sample(lambda x, t: model(x, t, {"action": z}))

    # Adapter is zero-init -> identical trajectory.
    max_diff = (latent_base - latent_adapter).abs().max().item()
    assert max_diff < 1e-3, f"base vs adapter latent diff too large: {max_diff}"

    # Store both videos + a comparison strip [cond | base@mid | adapter@mid].
    vid_base = vae.decode([latent_base[0]])[0]
    vid_adapter = vae.decode([latent_adapter[0]])[0]
    cache_video(vid_base[None], save_file="outputs/test_wan_cond_base.mp4", fps=8, value_range=(-1, 1))
    cache_video(vid_adapter[None], save_file="outputs/test_wan_cond_adapter.mp4", fps=8, value_range=(-1, 1))
    assert Path("outputs/test_wan_cond_base.mp4").exists()
    assert Path("outputs/test_wan_cond_adapter.mp4").exists()

    def mid(v):
        f = v[:, v.shape[1] // 2].add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        return f

    cond_vis = F.interpolate(img.unsqueeze(0), size=(height, width))[0].add(1).mul(127.5).byte().permute(1, 2, 0).numpy()
    strip = np.concatenate([cond_vis, mid(vid_base), mid(vid_adapter)], axis=1)
    iio.imwrite("outputs/test_wan_cond_base_vs_adapter_strip.png", strip)


@pytest.mark.skipif(not _HDF5.exists(), reason=f"MetaWorld HDF5 not found at {_HDF5}.")
@pytest.mark.skipif(not _DIT_SAFETENSORS.exists(), reason="Wan DiT safetensors not found.")
def test_wan_base_sdedit_reconstructs_metaworld_clip():
    """Is the WAN generation machinery actually correct? (WAN analogue of
    DynamiCrafter's ``test_base_model_image_to_video_rollout_is_semantically_
    consistent``.)

    Encodes a real MetaWorld clip to the ground-truth latent ``z0``, then rolls
    the frozen Wan base through two paths from the *same* noise: a frame-
    conditioned **SDEdit** rollout (anchored on ``z0``) and an **unconditional**
    pure-noise rollout. It asserts the SDEdit decode is close to GT in pixels
    **and clearly closer than the unconditional rollout** — the conditioning
    must demonstrably steer the result.

    This isolates the sampler / VAE / timestep convention from adapter training:
    if SDEdit reconstructs GT, the machinery is sound and washed-out eval-grid
    panels are an unconditional-null-text-base + undertrained-adapter artifact,
    not a pipeline bug. If even SDEdit fails, the generation path is broken.
    Stores ``[GT | sdedit | uncond]`` side by side for inspection. Thresholds
    are loose (null text, few steps); tune against the printed values.
    """
    import imageio.v3 as iio
    import numpy as np
    import torch.nn.functional as F

    from generative_flow_adapters.backbones.wan.modules.vae import WanVAE
    from generative_flow_adapters.backbones.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.training.builders import build_experiment

    device, dtype = "cuda", torch.bfloat16
    frames, height, width, steps = 17, 128, 128, 8
    os.makedirs("outputs", exist_ok=True)

    config = load_config("configs/wan21/diffusion_wan_shortcut_metaworld.yaml")
    config.model.pretrained_model_name_or_path = str(_CKPT_DIR)
    config.model.extra["allow_missing_checkpoint"] = False
    config.model.extra["dtype"] = "bf16"
    base = build_experiment(config).model.base_model.to(device).eval()

    # Tiled real frame -> Wan-VAE latent (same proven setup as the base-vs-adapter
    # test; frames=17 keeps the (4,8,8) temporal geometry clean). A static scene
    # still fully exercises the sampler / VAE / timestep convention.
    vae = WanVAE(vae_pth=str(_VAE_PTH), device=device)
    frame = _load_first_metaworld_frame()
    img = frame.permute(2, 0, 1).float().div(127.5).sub(1.0)
    img = F.interpolate(img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
    video_in = img.unsqueeze(1).repeat(1, frames, 1, 1).to(device)  # [3, F, H, W] in [-1, 1]
    z0 = vae.encode([video_in])[0].unsqueeze(0).float()            # ground-truth latent
    gt_px = vae.decode([z0[0]])[0]                                 # decode(z0): isolates sampler from VAE error

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    gen = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(1, 16, *z0.shape[2:], generator=gen, device=device, dtype=torch.float32)

    def rollout(strength: float):
        sched = scheduler
        sched.set_timesteps(steps, device=device, shift=3.0)
        start_idx = min(int(round((1.0 - strength) * steps)), steps - 1)
        sigma_start = sched.sigmas[start_idx].to(device)
        latent = ((1.0 - sigma_start) * z0 + sigma_start * noise).float()
        for t in sched.timesteps[start_idx:]:
            with torch.autocast("cuda", dtype=dtype):
                v = base(latent, t.unsqueeze(0).to(device), {})
            latent = sched.step(v.float(), t, latent, return_dict=False, generator=gen)[0]
        return latent

    sdedit_px = vae.decode([rollout(strength=0.5)[0]])[0]   # anchored on z0
    uncond_px = vae.decode([rollout(strength=1.0)[0]])[0]   # pure noise

    mse_sdedit = (sdedit_px.float() - gt_px.float()).pow(2).mean().item()
    mse_uncond = (uncond_px.float() - gt_px.float()).pow(2).mean().item()
    print(f"\n[wan-sdedit] MSE(sdedit, gt)={mse_sdedit:.4f}  MSE(uncond, gt)={mse_uncond:.4f}  ratio={mse_sdedit / max(mse_uncond, 1e-6):.2f}")

    # Store [GT | sdedit | uncond] side by side (per-frame, concatenated on width).
    def to_u8(v):
        return v.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 3, 0).cpu().numpy()  # [F, H, W, 3]

    panel = np.concatenate([to_u8(gt_px), to_u8(sdedit_px), to_u8(uncond_px)], axis=2)
    iio.imwrite("outputs/test_wan_sdedit_vs_gt.mp4", panel, fps=8)
    iio.imwrite("outputs/test_wan_sdedit_vs_gt_frame0.png", panel[0])
    assert os.path.exists("outputs/test_wan_sdedit_vs_gt.mp4")

    # The machinery works iff frame-conditioning demonstrably beats pure noise
    # at recovering GT, and the conditioned reconstruction is reasonably close.
    assert mse_sdedit < 0.6 * mse_uncond, (
        f"SDEdit conditioning did not steer toward GT (sdedit={mse_sdedit:.4f} "
        f"vs uncond={mse_uncond:.4f}); generation/convention likely broken."
    )
    assert mse_sdedit < 0.12, f"SDEdit reconstruction too far from GT: {mse_sdedit:.4f}"


@pytest.mark.skipif(not _HDF5.exists(), reason=f"MetaWorld HDF5 not found at {_HDF5}.")
@pytest.mark.skipif(not _DIT_SAFETENSORS.exists(), reason="Wan DiT safetensors not found.")
def test_wan_metaworld_training_step():
    """One real flow-matching training step: MetaWorld pixels -> Wan-VAE 16-ch
    latents -> (x_t,t,target) triple -> AdaptedModel -> finite loss, and only the
    adapter receives gradients (the base stays frozen)."""
    from torch.utils.data import DataLoader

    from generative_flow_adapters.backbones.wan.modules.vae import WanVAE
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.data import (
        WanBatchPreprocessConfig,
        WanBatchPreprocessor,
        build_metaworld_clip_dataset,
    )
    from generative_flow_adapters.training import build_experiment
    from generative_flow_adapters.training.trainer import Trainer

    device = "cuda"
    config = load_config("configs/wan21/diffusion_wan_shortcut_metaworld.yaml")
    config.model.pretrained_model_name_or_path = str(_CKPT_DIR)
    config.model.extra["allow_missing_checkpoint"] = False
    experiment = build_experiment(config)
    experiment.model.to(device)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)

    vae = WanVAE(vae_pth=str(_VAE_PTH), device=device)
    preprocessor = WanBatchPreprocessor(
        vae=vae,
        config=WanBatchPreprocessConfig(target_height=128, target_width=128),
        condition_keys=("act",),
    )
    _, dataset = build_metaworld_clip_dataset(
        config.data, default_window_width=8, hdf5=str(_HDF5), frame_stride=4, sampling="exhaustive"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    raw = next(iter(loader))
    batch = preprocessor(raw, train=True)

    assert batch["x_t"].shape[1] == 16  # Wan latent width
    assert batch["x_t"].shape == batch["target"].shape
    assert "action" in batch["cond"]

    metrics = trainer.training_step(batch)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert sum(p.numel() for p in experiment.model.base_model.parameters() if p.requires_grad) == 0
    assert any(p.grad is not None for p in experiment.model.adapter.parameters())


@pytest.mark.skipif(not _DIT_SAFETENSORS.exists(), reason="Wan DiT safetensors not found.")
def test_flow_inference_sampler_eval_rollout():
    """The trainer's FlowInferenceSampler runs a real eval rollout (bf16 +
    autocast) on the adapted Wan model and decodes to a coherent video."""
    import imageio.v3 as iio
    import numpy as np

    from generative_flow_adapters.backbones.wan.modules.vae import WanVAE
    from generative_flow_adapters.config import load_config
    from generative_flow_adapters.inference import FlowInferenceSampler
    from generative_flow_adapters.training.builders import build_experiment
    from generative_flow_adapters.training.trainer import Trainer

    device = "cuda"
    config = load_config("configs/wan21/diffusion_wan_shortcut_metaworld.yaml")
    config.model.pretrained_model_name_or_path = str(_CKPT_DIR)
    config.model.extra["allow_missing_checkpoint"] = False
    experiment = build_experiment(config)
    experiment.model.to(device)
    trainer = Trainer(experiment.model, experiment.optimizer, experiment.loss_fn, config.training)
    assert isinstance(trainer.inference_sampler, FlowInferenceSampler)

    # Small latent for speed; rollout from the eval sampler.
    batch = {"target": torch.zeros(1, 16, 3, 16, 16, device=device), "cond": {"action": torch.zeros(1, 4, device=device)}}
    latent = trainer.inference_sampler.sample_from_batch(batch, num_inference_steps=8)
    assert latent.shape == batch["target"].shape
    assert torch.isfinite(latent).all()

    # Decode + store for visual inspection.
    os.makedirs("outputs", exist_ok=True)
    vae = WanVAE(vae_pth=str(_VAE_PTH), device=device)
    video = vae.decode([latent[0].float()])[0]  # [3, F, H, W]
    frames = video.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 3, 0).cpu().numpy()
    iio.imwrite("outputs/test_wan_eval_rollout.mp4", frames, fps=8)
    assert os.path.exists("outputs/test_wan_eval_rollout.mp4")


@pytest.mark.skipif(not _DIT_SAFETENSORS.exists(), reason="Wan DiT safetensors not found.")
def test_wan_dit_loads_strict_into_wrapper():
    """The real 1.3B DiT weights load into our wrapper with an exact key match."""
    from generative_flow_adapters.models.base.wan import Wan21DiTWrapper

    base = Wan21DiTWrapper.from_config(
        wan_config_path="configs/base/wan2.1_t2v_1_3B.yaml",
        model_type="flow",
        prediction_type="velocity",
        checkpoint_path=str(_CKPT_DIR),
        strict_checkpoint=True,  # exact key match or raise
        dtype=torch.bfloat16,
    )
    n = sum(p.numel() for p in base.module.parameters())
    assert 1.3e9 < n < 1.5e9
    assert base.model_type == "flow" and base.prediction_type == "velocity"
