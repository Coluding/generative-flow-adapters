"""Native DynamiCrafter generation behind the :class:`BaseVideoModel` interface.

Generation-only wrapper (training still uses ``DynamicCrafterUNetWrapper``).

The lesson re-learned in :mod:`...inference.diffusion`: reimplementing
DynamiCrafter's DDIM (dynamic SNR rescale + first-frame anchoring + hybrid
concat/CLIP conditioning + fps embedding) is where correctness dies. So this
class does the opposite — it instantiates the **real vendored lvdm model**
(:class:`external_deps.lvdm.models.ddpm3d.LatentVisualDiffusion`) and delegates
generation to its own :class:`~external_deps.lvdm.models.samplers.ddim.DDIMSampler`,
exactly as the upstream ``evaluate_and_log`` does. The adapter is injected the
same way :class:`WanTI2V` wraps its DiT: we temporarily wrap the model's
``apply_model`` so every denoiser prediction passes through ``compose_fn``.

The vendored tree is importable only as ``external_deps.lvdm.*``, but the
upstream config (:file:`configs/base/dynamicrafter512.yaml`) carries bare
``lvdm.*`` ``target:`` strings, so we rewrite the prefix before instantiating.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import Tensor

from generative_flow_adapters.backbones.dynamicrafter.basics import instantiate_from_config
from generative_flow_adapters.models.base.video_model import BaseVideoModel, ComposeFn

_DEFAULT_MODEL_CONFIG = "configs/base/dynamicrafter512.yaml"


def _rewrite_lvdm_targets(node: object) -> None:
    """Recursively rewrite ``target: lvdm.*`` -> ``external_deps.lvdm.*`` in place.

    The vendored modules import each other via ``external_deps.lvdm`` but the
    upstream YAML targets the package as bare ``lvdm``; ``instantiate_from_config``
    resolves the string with ``importlib.import_module``, so the prefix must match
    what is importable here."""
    if OmegaConf.is_dict(node):
        for key, value in node.items():
            if key == "target" and isinstance(value, str) and value.startswith("lvdm."):
                node[key] = "external_deps." + value
            else:
                _rewrite_lvdm_targets(value)
    elif OmegaConf.is_list(node):
        for item in node:
            _rewrite_lvdm_targets(item)


class DynamiCrafterVideoModel(BaseVideoModel):
    """Frozen DynamiCrafter backbone that delegates generation to lvdm's own loop."""

    def __init__(self, lvdm_model: torch.nn.Module) -> None:
        super().__init__(model_type="diffusion", prediction_type="velocity")
        self.lvdm = lvdm_model

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(
        cls,
        checkpoint_path: str,
        model_config_path: str = _DEFAULT_MODEL_CONFIG,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> "DynamiCrafterVideoModel":
        cfg = OmegaConf.load(model_config_path)
        model_cfg = cfg.model
        _rewrite_lvdm_targets(model_cfg)
        # Training-only keys the LatentVisualDiffusion ctor does not accept.
        params = model_cfg.get("params")
        for stray in ("linear_warmup_steps",):
            if params is not None and stray in params:
                del params[stray]

        lvdm_model = instantiate_from_config(model_cfg)

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        missing, unexpected = lvdm_model.load_state_dict(state_dict, strict=False)
        print(f"[DynamiCrafterVideoModel] loaded {len(state_dict)} tensors from {checkpoint_path}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  first missing: {list(missing)[:6]}")
        if unexpected:
            print(f"  first unexpected: {list(unexpected)[:6]}")

        lvdm_model = lvdm_model.to(device)
        if dtype is not None:
            lvdm_model = lvdm_model.to(dtype)
        lvdm_model.eval()
        for p in lvdm_model.parameters():
            p.requires_grad_(False)
        return cls(lvdm_model)

    # -- VAE seam ---------------------------------------------------------
    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        return self.lvdm.encode_first_stage(pixels)

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        return self.lvdm.decode_first_stage(latent)

    # -- denoiser seam ----------------------------------------------------
    def denoise(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        x_recon, _ = self.lvdm.apply_model(x_t, t, cond)
        return x_recon

    # -- native generation ------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        conditioning: object,
        *,
        compose_fn: ComposeFn | None = None,
        ddim_steps: int = 50,
        guidance_scale: float = 1.0,
        guidance_rescale: float = 0.7,
        fs: int | None = None,
        ddim_eta: float = 1.0,
        timestep_spacing: str = "uniform_trailing",
        use_ema: bool = True,
        **kwargs: object,
    ) -> Tensor:
        """Run lvdm's own DDIM rollout on a raw ``batch`` (``conditioning``).

        ``batch`` must carry the keys ``prepare_batch_for_inference`` reads:
        ``video`` ``[b, c, t, h, w]`` in [-1, 1], ``caption`` (list[str]),
        ``act`` ``[b, t, a]``, and ``fps`` ``[b]``. With ``compose_fn`` set, every
        ``apply_model`` prediction is replaced by ``compose_fn(x_t, t, base_pred)``
        (the adapter seam). Returns decoded pixels ``[b, 3, t, H, W]`` in [-1, 1].
        """
        from external_deps.lvdm.models.samplers.ddim import DDIMSampler

        batch = conditioning
        model = self.lvdm

        z, c, uc, cond_mask, _log, sample_kwargs = model.prepare_batch_for_inference(batch)
        if fs is not None:
            sample_kwargs["fs"] = torch.full_like(sample_kwargs["fs"], int(fs))

        _, channels, temporal, height, width = z.shape
        shape = (channels, temporal, height, width)

        orig_apply_model = model.apply_model
        if compose_fn is not None:
            def _composed_apply_model(x_noisy, t, cond, **akw):
                base_recon, info = orig_apply_model(x_noisy, t, cond, **akw)
                return compose_fn(x_noisy, t, base_recon), info
            model.apply_model = _composed_apply_model

        try:
            ema = model.ema_scope("native-generate") if (use_ema and model.use_ema) else _nullctx()
            with ema:
                sampler = DDIMSampler(model)
                samples, _ = sampler.sample(
                    ddim_steps,
                    batch_size=z.shape[0],
                    shape=shape,
                    conditioning=c,
                    unconditional_conditioning=uc,
                    unconditional_guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    mask=cond_mask,
                    x0=z,
                    eta=ddim_eta,
                    timestep_spacing=timestep_spacing,
                    verbose=False,
                    **sample_kwargs,
                )
                pixels = model.decode_first_stage(samples)
        finally:
            model.apply_model = orig_apply_model
        return pixels


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
