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

import yaml
import torch
from torch import Tensor

from generative_flow_adapters.backbones.dynamicrafter.basics import instantiate_from_config
from generative_flow_adapters.models.base.video_model import BaseVideoModel, ComposeFn

_DEFAULT_MODEL_CONFIG = "configs/base/dynamicrafter512.yaml"


class _AttrDict(dict):
    """Dict that also allows attribute access. The vendored lvdm ctors reach
    config values both ways — ``config["target"]`` / ``config.get("params")``
    (in ``instantiate_from_config``) and ``unet_config.params.temporal_length``
    (attribute access in ``DDPM.__init__``) — so we need both. Used instead of
    OmegaConf, which is not installed in this env (which is why the rest of the
    repo hand-builds the UNet rather than instantiating the full model)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(node: object) -> object:
    """Recursively wrap dicts as :class:`_AttrDict` and rewrite ``target:``
    strings from the vendored-only ``external_deps.lvdm.*`` import path (the
    upstream YAML targets bare ``lvdm.*``, which is not importable here)."""
    if isinstance(node, dict):
        out = _AttrDict()
        for key, value in node.items():
            if key == "target" and isinstance(value, str) and value.startswith("lvdm."):
                out[key] = "external_deps." + value
            else:
                out[key] = _wrap(value)
        return out
    if isinstance(node, list):
        return [_wrap(item) for item in node]
    return node


class DynamiCrafterVideoModel(BaseVideoModel):
    """Frozen DynamiCrafter backbone that delegates generation to lvdm's own loop."""

    # ``generate`` takes a whole batch (video/caption/act/fps) + DDIM kwargs, not
    # a single frame + Wan kwargs, so the Trainer routes its periodic eval through
    # the batch-mode branch (``_native_dc_eval_grid``).
    native_eval_batch_mode = True

    def __init__(
        self,
        lvdm_model: torch.nn.Module,
        autocast_dtype: torch.dtype | None = None,
        schedule_params: dict | None = None,
    ) -> None:
        super().__init__(model_type="diffusion", prediction_type="velocity")
        self.lvdm = lvdm_model
        self._schedule_params = schedule_params or {}
        # When the weights are fp16, run the ops under autocast: several vendored
        # paths mint fp32 tensors (e.g. DiagonalGaussianDistribution.sample()'s
        # randn), so a plain fp16 forward hits "input float, weight half". The
        # fp32 schedule buffers are elementwise-only and untouched by autocast.
        self._autocast_dtype = autocast_dtype

    def _autocast(self):
        if self._autocast_dtype is None:
            return _nullctx()
        return torch.autocast("cuda", dtype=self._autocast_dtype)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(
        cls,
        checkpoint_path: str,
        model_config_path: str = _DEFAULT_MODEL_CONFIG,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> "DynamiCrafterVideoModel":
        with open(model_config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        model_cfg = _wrap(raw["model"])
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

        # Cast on CPU BEFORE moving to GPU so a fp32 copy never lands on-device
        # (the full model in fp32 OOMs a 24 GB card next to other jobs).
        if dtype is not None:
            lvdm_model = lvdm_model.to(dtype)
        lvdm_model = lvdm_model.to(device)
        if dtype is not None:
            # Keep the DDPM schedule buffers (betas / alphas_cumprod / scale_arr /
            # ... — the top-level, non-submodule buffers) in fp32: with
            # rescale_betas_zero_snr the terminal alphas underflow in fp16 and the
            # DDIM step math derived from them goes wrong.
            for name, buf in lvdm_model.named_buffers():
                if "." not in name and buf.is_floating_point():
                    lvdm_model._buffers[name] = buf.float()

        # Zero-SNR terminal fix: rescale_betas_zero_snr makes
        # alphas_cumprod[-1] == 0 exactly, so sqrt_recip{,m1}_alphas_cumprod[-1]
        # == inf and DDIM's predict_start_from_noise at t=T computes inf*x-inf*e
        # == NaN. The two coefficients are asymptotically equal as alpha->0 (so
        # pred_x0 -> 0 there), so replacing the inf terminal with the largest
        # finite value keeps the terminal step well-defined without perturbing
        # any other timestep. (Native lvdm hits this because uniform_trailing
        # starts the rollout exactly at t=T.)
        for name in ("sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod"):
            buf = getattr(lvdm_model, name, None)
            if isinstance(buf, torch.Tensor):
                inf_mask = torch.isinf(buf)
                if bool(inf_mask.any()):
                    buf[inf_mask] = buf[~inf_mask].max()

        lvdm_model.eval()
        for p in lvdm_model.parameters():
            p.requires_grad_(False)
        autocast_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else None
        # DynamiCrafter schedule params, so the Trainer builds the SAME diffusion
        # objective it noises training batches with (zero-SNR + dynamic rescale).
        mp = raw.get("model", {}).get("params", {}) if isinstance(raw, dict) else {}
        schedule_params = {
            "timesteps": int(mp.get("timesteps", 1000)),
            "beta_schedule": str(mp.get("beta_schedule", "linear")),
            "linear_start": float(mp.get("linear_start", 8.5e-4)),
            "linear_end": float(mp.get("linear_end", 1.2e-2)),
            "rescale_betas_zero_snr": bool(mp.get("rescale_betas_zero_snr", False)),
            "use_dynamic_rescale": bool(mp.get("use_dynamic_rescale", False)),
            "base_scale": float(mp.get("base_scale", 0.7)),
            "turning_step": int(mp.get("turning_step", 400)),
        }
        return cls(lvdm_model, autocast_dtype=autocast_dtype, schedule_params=schedule_params)

    @property
    def diffusion_schedule_config(self):
        """DynamiCrafter DDPM schedule (zero-SNR, dynamic rescale, v-param) so the
        Trainer's diffusion objective matches the base the loss is computed against.
        Returns a ``DiffusionScheduleConfig`` (dict subclass) or ``None``."""
        if not self._schedule_params:
            return None
        from generative_flow_adapters.losses.diffusion import DiffusionScheduleConfig

        return DiffusionScheduleConfig(**self._schedule_params)

    # -- VAE seam ---------------------------------------------------------
    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        with self._autocast():
            return self.lvdm.encode_first_stage(pixels)

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        with self._autocast():
            return self.lvdm.decode_first_stage(latent)

    @torch.no_grad()
    def decode_first_stage(self, latent: Tensor) -> Tensor:
        """Alias of :meth:`decode` under the name ``build_experiment`` /
        ``WandbLogger`` probe for (``getattr(base, "decode_first_stage")``)."""
        return self.decode(latent)

    @property
    def first_stage_model(self):
        """Exposed so ``build_experiment`` detects a usable VAE decoder
        (``has_vae = decode_first_stage is not None and first_stage_model is not None``)."""
        return self.lvdm.first_stage_model

    # -- denoiser seam ----------------------------------------------------
    def denoise(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        """One frozen-base denoiser step. Accepts EITHER our
        ``DynamiCrafterBatchPreprocessor`` cond (``{context, concat, act, fs,
        dropout_actions}``) — the training / adapter-composition seam — OR the
        native lvdm hybrid cond (``{c_concat, c_crossattn}``). Both route into
        ``apply_model`` -> ``DiffusionWrapper`` -> the same UNet, so the output
        matches ``DynamicCrafterUNetWrapper.forward`` for the same inputs."""
        lvdm_cond, akw = _to_lvdm_cond(cond)
        # Run under the model's own autocast so the frozen (possibly bf16) base is
        # dtype-safe even when a caller invokes it OUTSIDE the Trainer's autocast
        # (e.g. the shortcut-target prep in Trainer._maybe_prepare_shortcut). A
        # nested autocast is idempotent, so this is safe under the training loop too.
        with self._autocast():
            x_recon, _ = self.lvdm.apply_model(x_t, t, lvdm_cond, **akw)
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
        ddim_eta: float = 0.0,
        timestep_spacing: str = "uniform_trailing",
        use_ema: bool = False,
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

        orig_apply_model = model.apply_model
        if compose_fn is not None:
            def _composed_apply_model(x_noisy, t, cond, **akw):
                base_recon, info = orig_apply_model(x_noisy, t, cond, **akw)
                return compose_fn(x_noisy, t, base_recon), info
            model.apply_model = _composed_apply_model

        try:
            ema = model.ema_scope("native-generate") if (use_ema and model.use_ema) else _nullctx()
            with ema, self._autocast():
                # Conditioning build (VAE encode + CLIP) must be inside autocast:
                # get_input force-casts video to fp32, which would clash with
                # fp16 VAE weights otherwise.
                z, c, uc, cond_mask, _log, sample_kwargs = model.prepare_batch_for_inference(batch)
                if fs is not None:
                    sample_kwargs["fs"] = torch.full_like(sample_kwargs["fs"], int(fs))
                _, channels, temporal, height, width = z.shape
                sampler = DDIMSampler(model)
                samples, _ = sampler.sample(
                    ddim_steps,
                    batch_size=z.shape[0],
                    shape=(channels, temporal, height, width),
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


def _to_lvdm_cond(cond: object) -> tuple[object, dict]:
    """Map a conditioning dict to ``apply_model``'s ``(cond, **kwargs)`` form.

    Two accepted input shapes:
      * lvdm hybrid (already ``{c_concat, c_crossattn}``) — pass the hybrid keys
        through, lift ``fs``/``act``/``dropout_actions`` into kwargs;
      * our ``DynamiCrafterBatchPreprocessor`` cond (``{context, concat, act, fs,
        dropout_actions}``) — ``concat`` -> ``c_concat=[concat]``, ``context`` ->
        ``c_crossattn=[context]``, and ``fs``/``act``/``dropout_actions`` -> kwargs.
    ``DiffusionWrapper.forward`` (hybrid) reads exactly these.
    """
    if not isinstance(cond, dict):
        return cond, {}
    akw: dict = {}
    for key in ("fs", "act", "dropout_actions"):
        if cond.get(key) is not None:
            akw[key] = cond[key]
    if "c_concat" in cond or "c_crossattn" in cond:
        lvdm_cond = {k: cond[k] for k in ("c_concat", "c_crossattn") if k in cond}
        return lvdm_cond, akw
    lvdm_cond = {}
    if cond.get("concat") is not None:
        lvdm_cond["c_concat"] = [cond["concat"]]
    if cond.get("context") is not None:
        lvdm_cond["c_crossattn"] = [cond["context"]]
    return lvdm_cond, akw


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
