"""SkyReels-V2-I2V-1.3B :class:`BaseVideoModel`, backed by the vendored
``external_repos/SkyReels-V2`` package.

SkyReels-V2 is a Wan(2.1)-lineage flow-matching DiT (velocity) with the 16-channel
Wan2.1 VAE (``vae_stride=(4,8,8)``). It is the WEAK flow base for the
base-strength axis (vs Wan2.2-5B strong, DynamiCrafter diffusion) — see
thesis-vault ``30_Knowledge/writing/ablation-axes.md`` (Axis 5).

Mirrors :class:`~...models.base.wan_ti2v.WanTI2VVideoModel`: instead of
reimplementing SkyReels' sampling loop, this wraps SkyReels' own
``Image2VideoPipeline`` and calls **its** rollout for :meth:`generate`, so
base-only output matches upstream by construction.

**Conditioning difference from Wan TI2V.** SkyReels-V2-I2V is a *classic* i2v
model, NOT diffusion-forcing: every DiT call needs ``clip_fea`` (CLIP features of
the conditioning frame) and ``y`` (the VAE-encoded conditioning frame + a
temporal mask, concatenated on the channel axis — the DiT does
``x = cat([x, y], dim=1)`` internally). :meth:`generate` builds those via the
native pipeline. :meth:`denoise` (the training/composition seam) therefore
expects them in ``cond`` — a SkyReels-specific batch preprocessor must supply
``cond['context']``, ``cond['clip_fea']`` and ``cond['y']`` (see the TODO in
``from_config`` and the config headers). That preprocessor is the one remaining
piece before a real SkyReels training run.

Adapter injection is the same single seam as Wan: the pipeline calls
``self.transformer(x, t=..., context=..., clip_fea=..., y=...)`` each step; with a
``compose_fn`` we swap ``pipeline.transformer`` for :class:`_ComposedSkyReelsDiT`,
which returns ``compose_fn(x, t, base_pred)``. Because the additive residual is
applied identically to the positive and negative CFG branches (same ``x``,``t``),
it composes cleanly under CFG (``uncond+δ + g·((cond+δ)-(uncond+δ)) == base+δ``).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from generative_flow_adapters.models.base.video_model import BaseVideoModel, ComposeFn

# Repo root -> external_repos/SkyReels-V2 (the vendored SkyReels package).
_SKYREELS_REPO = Path(__file__).resolve().parents[4] / "external_repos" / "SkyReels-V2"


def _ensure_skyreels_importable() -> None:
    """Put ``external_repos/SkyReels-V2`` on ``sys.path`` so ``import
    skyreels_v2_infer`` resolves — the deliberate no-pip-install pattern used for
    the Wan repo (its pins would churn the working venv)."""
    path = str(_SKYREELS_REPO)
    if path not in sys.path:
        if not _SKYREELS_REPO.exists():
            raise FileNotFoundError(
                f"Vendored SkyReels-V2 not found at {_SKYREELS_REPO}. "
                "Run jobs/experiments/setup_skyreels.sh to clone it."
            )
        sys.path.insert(0, path)


def _patch_flash_attention_fallback() -> None:
    """Route SkyReels' attention through SDPA when flash-attn is absent.

    ``skyreels_v2_infer.modules.attention`` already ships the fallback: its
    ``attention()`` dispatches to ``torch.nn.functional.scaled_dot_product_attention``
    when neither flash-attn 2 nor 3 imports. But nothing calls it — ``clip.py``
    (2 sites) and ``transformer.py`` (4 sites) call ``flash_attention()``
    directly, which opens with a bare ``assert FLASH_ATTN_2_AVAILABLE``. On a
    venv without flash-attn that is an ``AssertionError`` the moment CLIP or the
    DiT runs a forward pass.

    Equivalent here, not merely a degradation: every call site uses
    ``window_size=(-1, -1)`` (flash-attn's "no windowing") and none passes
    ``q_lens``/``k_lens``, so the two paths compute the same full, unmasked
    attention. On H100 + bf16 torch's SDPA picks its own flash kernel, so the
    throughput cost is small.

    Patching the module globals is enough: Python resolves ``flash_attention``
    at call time, and both modules bound it with ``from .attention import ...``,
    so the name must be replaced in each importing module, not just the source.

    No-op when flash-attn IS installed — the real kernels are preferred.
    """
    from skyreels_v2_infer.modules import attention as attn_mod  # noqa: PLC0415

    if attn_mod.FLASH_ATTN_2_AVAILABLE or attn_mod.FLASH_ATTN_3_AVAILABLE:
        return

    from skyreels_v2_infer.modules import clip as clip_mod  # noqa: PLC0415
    from skyreels_v2_infer.modules import transformer as transformer_mod  # noqa: PLC0415

    def _sdpa_flash_attention(q, k, v, version=None, fa_version=None, **kwargs):
        # `version`/`fa_version` select a flash-attn build; meaningless here.
        return attn_mod.attention(q=q, k=k, v=v, **kwargs)

    for module in (clip_mod, transformer_mod):
        module.flash_attention = _sdpa_flash_attention

    warnings.warn(
        "flash-attn not installed — SkyReels attention is running on torch SDPA. "
        "Numerically equivalent (window_size=(-1,-1), no padding masks at any "
        "call site); install flash-attn 2 to use the native kernels.",
        RuntimeWarning,
        stacklevel=2,
    )


class _ComposedSkyReelsDiT:
    """Drop-in stand-in for ``Image2VideoPipeline.transformer`` that routes each
    prediction through ``compose_fn`` (the adapter residual).

    Matches the SkyReels DiT call convention
    ``(x, t, context, clip_fea=None, y=None, fps=None)`` where ``x`` is a BATCHED
    tensor ``[B,C,F,h,w]`` and the return is a **list** of ``[C_out,F,h,w]`` (the
    pipeline indexes ``[0]``). ``compose_fn`` receives the batched ``x`` (the 16-ch
    latent, BEFORE the internal ``y`` concat) and the batched base prediction.

    Unlike the Wan TI2V stand-in this does NOT memoize: SkyReels' CFG loop calls
    the DiT with a different ``context`` for the positive vs negative branch, so
    both are genuine forwards.
    """

    def __init__(self, dit, compose_fn: ComposeFn | None = None) -> None:
        self._dit = dit
        self._compose_fn = compose_fn

    def __call__(self, x, t, context, clip_fea=None, y=None, fps=None):
        base = self._dit(x, t=t, context=context, clip_fea=clip_fea, y=y, fps=fps)  # list[[C,F,h,w]]
        if self._compose_fn is None:
            return base
        base_b = torch.stack(list(base), dim=0)  # [B, C, F, h, w]
        final = self._compose_fn(x, t, base_b)
        return [final[i] for i in range(final.shape[0])]

    def __getattr__(self, name):  # delegate .to / .cpu / .parameters / .dtype / ...
        return getattr(self._dit, name)


class SkyReelsVideoModel(BaseVideoModel):
    """Frozen SkyReels-V2-I2V-1.3B backbone (flow / velocity) via the vendored repo."""

    def __init__(
        self,
        model_path: str,
        *,
        dit_path: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        offload: bool = True,
        default_prompt: str = "",
        default_negative: str = "",
    ) -> None:
        super().__init__(model_type="flow", prediction_type="velocity")
        _ensure_skyreels_importable()
        _patch_flash_attention_fallback()
        from skyreels_v2_infer.pipelines import Image2VideoPipeline  # noqa: PLC0415

        self.device = device
        self.dtype = dtype
        self.offload = offload
        self.default_prompt = default_prompt
        self.default_negative = default_negative

        # The pipeline builds transformer (WanModel, in_dim=36 for i2v), the 16-ch
        # Wan2.1 VAE, T5 text encoder and CLIP image encoder — same components the
        # native rollout uses, so base-only generate() matches upstream.
        self._pipeline = Image2VideoPipeline(
            model_path=model_path,
            dit_path=dit_path or model_path,
            device=device,
            weight_dtype=dtype,
            offload=offload,
        )
        # Register the DiT so .parameters()/.freeze()/.to() see the real weights.
        self.dit = self._pipeline.transformer
        self.freeze()

    # -- VAE seam ---------------------------------------------------------
    # SkyReels' WanVAE.encode/decode take a BATCHED tensor [B,3,T,H,W] / [B,C,T',h,w]
    # (matching Image2VideoPipeline's own usage) and return the same, already
    # scaled + clamped to [-1,1] on decode.
    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        vae = self._pipeline.vae
        return vae.encode(pixels.to(self.device).float()).float()

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        vae = self._pipeline.vae
        return vae.decode(latent.to(self.device).float())

    # -- denoiser seam (training + adapter composition) -------------------
    @torch.no_grad()
    def denoise(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        """Single frozen SkyReels DiT step (velocity) — the training/composition
        seam. ``cond`` MUST carry the i2v conditioning the DiT requires:

        - ``cond['context']`` : text embeddings ``[B, L, C]`` (or list of ``[L,C]``)
        - ``cond['clip_fea']`` : CLIP features of the conditioning frame
        - ``cond['y']`` : encoded conditioning frame + mask, ``[B, 20, T', h, w]``

        These are produced by the (still-to-be-written) SkyReels batch
        preprocessor — the native pipeline builds them internally in
        :meth:`generate`, but training needs them precomputed in the batch.
        """
        if not isinstance(cond, dict):
            raise ValueError(
                "SkyReelsVideoModel.denoise requires cond with 'context'/'clip_fea'/'y' "
                "(i2v conditioning). A SkyReels batch preprocessor must supply them."
            )
        device = self.device
        context = cond.get("context")
        clip_fea = cond.get("clip_fea")
        y = cond.get("y")
        if context is None or clip_fea is None or y is None:
            raise ValueError(
                "SkyReels denoise needs cond['context'], cond['clip_fea'] and cond['y']. "
                f"Got keys: {sorted(cond)}."
            )
        x_b = x_t.to(device)
        t_model = torch.as_tensor(t, device=device).float()
        if t_model.dim() == 0:
            t_model = t_model.reshape(1)
        out = self.dit(
            x_b, t=t_model, context=context,
            clip_fea=clip_fea.to(device) if isinstance(clip_fea, Tensor) else clip_fea,
            y=y.to(device) if isinstance(y, Tensor) else y,
        )
        return torch.stack(list(out), dim=0)

    # -- native generation ------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        conditioning: object,
        *,
        compose_fn: ComposeFn | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 544,
        width: int = 960,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 5.0,
        seed: int = 0,
        # --- Wan-dialect aliases, named so they are NOT swallowed by **kwargs ---
        sampling_steps: int | None = None,
        guide_scale: float | None = None,
        frame_num: int | None = None,
        max_area: int | None = None,
        context: Tensor | None = None,
        context_null: Tensor | None = None,
        offload_model: bool | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Run SkyReels' native i2v rollout conditioned on the frame
        ``conditioning`` (PIL / HWC uint8 array / ``[3,H,W]`` or ``[H,W,3]``
        tensor). ``compose_fn is None`` -> byte-for-byte upstream; otherwise every
        DiT prediction is replaced by ``compose_fn(x, t, base_pred)``.

        Returns ``[3, N, H, W]`` float in ``[-1, 1]`` — the
        :class:`BaseVideoModel` contract that ``Trainer._native_clip_rollout``
        consumes. The vendored pipeline instead returns a *list* of uint8 numpy
        ``[T, H, W, 3]`` arrays in 0-255, so we convert. (Round-tripping through
        uint8 is lossless for the caller: the grid is re-quantised to uint8 by
        ``Trainer._pixels_to_uint8`` anyway.)

        WAN-DIALECT ALIASES: ``_native_clip_rollout`` is shared across backbones
        and speaks Wan's kwarg names. They used to land in ``**kwargs`` and be
        dropped in silence, which is worse than an error — every eval row would
        have rendered at the *defaults* (50 steps, 544x960) no matter what the
        schedule and config asked for, so the 8-vs-50-step grid would have been
        two identical rows at the wrong resolution. Declared explicitly so they
        bind and get translated:

            sampling_steps -> num_inference_steps
            guide_scale    -> guidance_scale
            frame_num      -> num_frames
            max_area       -> (width, height) via best_output_size, aspect
                              preserved, aligned to 16 (VAE stride 8 x patch 2)

        ``context``/``context_null`` are accepted and DELIBERATELY ignored: they
        are Wan umT5 embeddings, and SkyReels encodes text with its own T5 from
        ``prompt`` (the config header says the umT5 tables are not reusable).
        ``offload_model`` is likewise ignored — the pipeline drives offload from
        its own flag.
        """
        # Local import: keeps models/ -> data/ off the module import graph.
        from generative_flow_adapters.data.wan_batch_preprocessor import (  # noqa: PLC0415
            best_output_size,
        )

        if sampling_steps is not None:
            num_inference_steps = int(sampling_steps)
        if guide_scale is not None:
            guidance_scale = float(guide_scale)
        if frame_num is not None:
            num_frames = int(frame_num)

        img = self._to_pil(conditioning)
        if max_area is not None:
            src_w, src_h = img.size
            width, height = best_output_size(src_w, src_h, 16, 16, int(max_area))
        pipe = self._pipeline
        original = pipe.transformer
        pipe.transformer = _ComposedSkyReelsDiT(original, compose_fn)
        try:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            video = pipe(
                image=img,
                prompt=prompt if prompt is not None else self.default_prompt,
                negative_prompt=negative_prompt if negative_prompt is not None else self.default_negative,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=gen,
            )
        finally:
            pipe.transformer = original
        return self._pipeline_video_to_tensor(video)

    @staticmethod
    def _pipeline_video_to_tensor(video: object) -> Tensor:
        """Pipeline output -> ``[3, N, H, W]`` float in ``[-1, 1]``.

        ``Image2VideoPipeline.__call__`` ends with

            videos = [v.permute(1, 2, 3, 0) * 255 for v in videos]
            videos = [v.cpu().numpy().astype(np.uint8) for v in videos]

        i.e. a LIST of uint8 numpy ``[T, H, W, 3]`` in 0-255 — one entry per
        batch item, and we always roll out a single clip.
        """
        if isinstance(video, (list, tuple)):
            if not video:
                raise ValueError("SkyReels pipeline returned no video.")
            video = video[0]
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(np.ascontiguousarray(video))
        if not isinstance(video, Tensor):
            raise TypeError(f"Unexpected SkyReels pipeline output type: {type(video)!r}")
        if video.dtype == torch.uint8:                  # [T,H,W,3] 0..255
            video = video.float().div_(127.5).sub_(1.0)
            return video.permute(3, 0, 1, 2).contiguous()
        # Already float: accept [3,T,H,W] as-is, else assume [T,H,W,3] in [0,1].
        if video.shape[0] == 3:
            return video.float()
        return video.float().mul(2.0).sub(1.0).permute(3, 0, 1, 2).contiguous()

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(
        cls,
        *,
        model_id: str = "Skywork/SkyReels-V2-I2V-1.3B-540P",
        model_path: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        offload: bool = True,
        default_prompt: str = "",
        default_negative: str = "",
    ) -> "SkyReelsVideoModel":
        """Resolve the checkpoint (local ``model_path`` or HF ``model_id``, which
        ``download_model`` returns from the local cache if already fetched — the
        earlier probe downloaded ``Skywork/SkyReels-V2-I2V-1.3B-540P``) and build
        the model."""
        _ensure_skyreels_importable()
        if model_path is None:
            from skyreels_v2_infer.modules import download_model  # noqa: PLC0415
            model_path = download_model(model_id)
        return cls(
            model_path=model_path,
            device=device,
            dtype=dtype or torch.bfloat16,
            offload=offload,
            default_prompt=default_prompt,
            default_negative=default_negative,
        )

    @staticmethod
    def _to_pil(frame: object) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, Tensor):
            arr = frame.detach().cpu()
            if arr.dim() == 3 and arr.shape[0] == 3:  # [3,H,W]
                arr = arr.permute(1, 2, 0)
            frame = arr.numpy()
        if isinstance(frame, np.ndarray):
            arr = frame
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8) * 255.0).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported conditioning frame type: {type(frame)!r}")
