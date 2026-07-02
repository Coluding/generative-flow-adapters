"""Wan2.2 TI2V-5B :class:`BaseVideoModel`, backed by the *external* Wan repo.

Instead of vendoring Wan's model + sampling code (and reimplementing the
denoising loop — which is what produced washed-out garbage), this wraps the
upstream ``wan.WanTI2V`` from ``external_repos/wan22`` and calls **its own**
``generate`` for sampling. Base-only output therefore matches the upstream repo
by construction.

Conditioning is frame-only: ``WanTI2V(skip_text_encoder=True)`` loads a cached
unconditional text embedding (``uncond_context.pt``) and drives generation
purely from the observation frame via Wan's native diffusion-forcing mask — no
prompt, no CLIP, no concat channel.

Adapter injection happens at a single seam: Wan's denoising loop calls
``self.model(latent_list, t=..., context=..., seq_len=...)`` every step. When a
``compose_fn`` is given we temporarily swap ``wan.model`` for a wrapper that
returns ``compose_fn(x, t, base_pred)`` — i.e. the caller
(:class:`AdaptedModel`) decides how the adapter residual is composed. Under
Wan's CFG an additive residual composes cleanly
(``uncond+δ + g·((cond+δ)-(uncond+δ)) == base + δ``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from generative_flow_adapters.models.base.video_model import BaseVideoModel, ComposeFn

# Repo root -> external_repos/wan22 (the upstream Wan2.2 package lives here).
_WAN22_REPO = Path(__file__).resolve().parents[4] / "external_repos" / "wan22"


def _ensure_wan_importable() -> None:
    """Put ``external_repos/wan22`` on ``sys.path`` so ``import wan`` resolves.

    A deliberate alternative to ``pip install -e external_repos/wan22``: the repo
    pins ``flash_attn`` / ``numpy<2`` etc., and installing it would churn the
    working venv even though its runtime deps are already satisfied here."""
    path = str(_WAN22_REPO)
    if path not in sys.path:
        if not _WAN22_REPO.exists():
            raise FileNotFoundError(
                f"External Wan2.2 repo not found at {_WAN22_REPO}. Clone it into external_repos/wan22."
            )
        sys.path.insert(0, path)


class _ComposedDiT:
    """Drop-in stand-in for ``wan.WanTI2V.model`` that routes each prediction
    through a ``compose_fn``.

    Mirrors the DiT's call convention — ``(x, t, context, seq_len, y=None)`` with
    ``x`` a list of ``[C, F, h, w]`` latents and the return a list of the same —
    so Wan's denoising loop is unchanged. Attribute access (``.to``, ``.cpu``,
    ``.parameters`` ...) passes through to the real DiT."""

    def __init__(self, dit, compose_fn: ComposeFn = lambda x,y,z : x) -> None:
        self._dit = dit
        self._compose_fn = compose_fn

    def __call__(self, x, t, context, seq_len, y=None):
        base = self._dit(x, t=t, context=context, seq_len=seq_len, y=y)  # list[[C,F,h,w]]
        x_b = torch.stack(list(x), dim=0)          # [B, C, F, h, w]
        base_b = torch.stack(list(base), dim=0)    # [B, C, F, h, w]
        final = self._compose_fn(x_b, t, base_b)   # [B, C, F, h, w]
        return [final[i] for i in range(final.shape[0])]

    def __getattr__(self, name):  # delegate .to / .cpu / .parameters / ...
        return getattr(self._dit, name)


class WanTI2VVideoModel(BaseVideoModel):
    """Frozen Wan2.2 TI2V-5B backbone (flow / velocity) via the upstream repo."""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        device_id: int = 0,
        convert_model_dtype: bool = True,
        offload_model: bool = True,
    ) -> None:
        super().__init__(model_type="flow", prediction_type="velocity")
        _ensure_wan_importable()
        import wan  # noqa: PLC0415  (path set above)
        from wan.configs import WAN_CONFIGS  # noqa: PLC0415

        self._config = WAN_CONFIGS["ti2v-5B"]
        self.offload_model = offload_model
        # Frame-only conditioning: skip T5, use the cached uncond embedding.
        self.wan = wan.WanTI2V(
            config=self._config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=convert_model_dtype,
            skip_text_encoder=True,
        )
        # Register the DiT so .parameters()/.freeze()/.to() see the real weights.
        self.dit = self.wan.model
        self.freeze()

    # -- VAE seam ---------------------------------------------------------
    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        clips = [pixels[i].to(self.wan.device).float() for i in range(pixels.shape[0])]
        return torch.stack(self.wan.vae.encode(clips), dim=0).float()

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        clips = self.wan.vae.decode([latent[i].to(self.wan.device).float() for i in range(latent.shape[0])])
        return torch.stack(clips, dim=0)

    # -- denoiser seam ----------------------------------------------------
    @torch.no_grad()
    def denoise(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        """Single frozen Wan DiT step (velocity), the training/composition seam.

        ``t`` may be a scalar / ``[B]`` (uniform), ``[B, T']`` (the
        diffusion-forcing per-latent-frame form the preprocessor builds), or
        already ``[B, seq_len]`` per token — all are normalised to what the Wan
        DiT expects. ``cond`` may carry a ``context``; absent, the cached
        unconditional context is used (frame-only regime)."""
        batch, _, f, h, w = x_t.shape
        tokens_per_frame = (h // self._config.patch_size[1]) * (w // self._config.patch_size[2])
        seq_len = f * tokens_per_frame
        device = self.wan.device

        context = cond.get("context") if isinstance(cond, dict) else None
        if context is None:
            context = [self.wan._uncond_ctx.to(device)] * batch

        x_list = [x_t[i].to(device) for i in range(batch)]
        t_model = self._to_dit_timesteps(t, batch, f, seq_len, tokens_per_frame, device)
        out = self.wan.model(x_list, t=t_model, context=context, seq_len=seq_len)
        return torch.stack(list(out), dim=0)

    @staticmethod
    def _to_dit_timesteps(
        t: object, batch: int, f: int, seq_len: int, tokens_per_frame: int, device: torch.device
    ) -> Tensor:
        """Normalise ``t`` to the ``[B]`` or per-token ``[B, seq_len]`` form the
        Wan DiT accepts. Per-latent-frame ``[B, T']`` is expanded frame-major
        (each frame -> ``tokens_per_frame`` tokens), matching the DiT's
        ``flatten`` over the patch grid."""
        t = torch.as_tensor(t, device=device).float()
        if t.dim() == 0:
            t = t.reshape(1)
        # Already per token: [B, seq_len] (or shared [1, seq_len]).
        if t.dim() == 2 and t.shape[1] == seq_len:
            return t.expand(batch, -1) if t.shape[0] == 1 else t
        # Per latent frame: [B, T'] / [1, T'] -> per token.
        if t.dim() == 2 and t.shape[1] == f:
            per_token = t.repeat_interleave(tokens_per_frame, dim=1)
            return per_token.expand(batch, -1) if per_token.shape[0] == 1 else per_token
        if t.dim() == 1 and t.shape[0] == f and f != batch:
            per_token = t.repeat_interleave(tokens_per_frame).unsqueeze(0)
            return per_token.expand(batch, -1)
        # Uniform per sample [B] (or scalar) -> let the DiT broadcast over tokens.
        t = t.reshape(-1)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch)
        return t

    # -- native generation ------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        conditioning: object,
        *,
        compose_fn: ComposeFn | None = None,
        max_area: int = 704 * 1280,
        frame_num: int = 121,
        sampling_steps: int = 50,
        shift: float = 5.0,
        guide_scale: float = 5.0,
        seed: int = 0,
        **kwargs: object,
    ) -> Tensor:
        """Run Wan's native i2v rollout conditioned on the observation frame
        ``conditioning`` (a PIL image, HWC uint8 array, or ``[H,W,3]``/``[3,H,W]``
        tensor). Returns pixels ``[3, N, H, W]`` in ``[-1, 1]``.

        ``compose_fn is None`` -> byte-for-byte upstream output. Otherwise every
        DiT prediction is replaced by ``compose_fn(x, t, base_pred)`` (see
        :class:`~...models.base.video_model.BaseVideoModel.generate`)."""
        img = self._to_pil(conditioning)

        original = self.wan.model
        if compose_fn is not None:
            self.wan.model = _ComposedDiT(original, compose_fn)
        try:
            video = self.wan.generate(
                None,  # input_prompt: unused in frame-only mode
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver="unipc",
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed,
                offload_model=self.offload_model,
            )
        finally:
            self.wan.model = original
        return video

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
            if arr.dtype != np.uint8:  # assume [-1,1] or [0,1] float
                arr = ((arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8) * 255.0)
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported conditioning frame type: {type(frame)!r}")
