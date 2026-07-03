"""Wan2.2 TI2V-5B :class:`BaseVideoModel`, backed by the *external* Wan repo.

Instead of vendoring Wan's model + sampling code (and reimplementing the
denoising loop — which is what produced washed-out garbage), this wraps the
upstream ``wan.WanTI2V`` from ``external_repos/Wan2.2`` and calls **its own**
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
_WAN22_REPO = Path(__file__).resolve().parents[4] / "external_repos" / "Wan2.2"


def _ensure_wan_importable() -> None:
    """Put ``external_repos/Wan2.2`` on ``sys.path`` so ``import wan`` resolves.

    A deliberate alternative to ``pip install -e external_repos/Wan2.2``: the repo
    pins ``flash_attn`` / ``numpy<2`` etc., and installing it would churn the
    working venv even though its runtime deps are already satisfied here."""
    path = str(_WAN22_REPO)
    if path not in sys.path:
        if not _WAN22_REPO.exists():
            raise FileNotFoundError(
                f"External Wan2.2 repo not found at {_WAN22_REPO}. Clone it into external_repos/Wan2.2."
            )
        sys.path.insert(0, path)


class _ComposedDiT:
    """Drop-in stand-in for ``wan.WanTI2V.model`` that (a) optionally routes each
    prediction through a ``compose_fn`` (the adapter residual) and (b) **memoizes**
    so a redundant, identical model call inside a denoising step runs only once.

    Wan's loop calls the DiT twice per step — positive context then negative —
    and combines them as ``uncond + g·(cond - uncond)``. In frame-only /
    prompt-free generation both branches use the *same* cached unconditional
    embedding, so the two calls are identical and the CFG term is zero. The
    1-entry cache turns the second call into a hit, so the frozen DiT **and** the
    adapter run once, not twice. With a genuine positive≠negative prompt the two
    calls differ (different ``context`` tensor) → cache miss → both run, so real
    CFG is untouched.

    The key is the per-step latent + timestep + context *identity*: a new step
    brings fresh latent/timestep tensors, so there are no stale cross-step hits.
    Correctness relies on generation being deterministic (``eval``/``no_grad``),
    which the frozen base and the adapter both are here.

    Mirrors the DiT call convention ``(x, t, context, seq_len, y=None)`` (``x`` a
    list of ``[C,F,h,w]`` latents, returning a list of the same), so Wan's loop
    is unchanged. Attribute access passes through to the real DiT.
    """

    def __init__(self, dit, compose_fn: ComposeFn | None = None) -> None:
        self._dit = dit
        self._compose_fn = compose_fn
        self._cache_key: object | None = None
        self._cache_out: list | None = None

    def __call__(self, x, t, context, seq_len, y=None):
        # Identity of this (step, CFG-branch): the timestep value pins the step
        # (monotonic schedule), the context tensor id distinguishes cond/uncond.
        key = (float(t.reshape(-1).max()), id(x[0]), id(context[0]))
        if key == self._cache_key:
            return self._cache_out  # identical call this step -> skip the forward
        base = self._dit(x, t=t, context=context, seq_len=seq_len, y=y)  # list[[C,F,h,w]]
        if self._compose_fn is None:
            out = base
        else:
            x_b = torch.stack(list(x), dim=0)        # [B, C, F, h, w]
            base_b = torch.stack(list(base), dim=0)  # [B, C, F, h, w]
            final = self._compose_fn(x_b, t, base_b)
            out = [final[i] for i in range(final.shape[0])]
        self._cache_key, self._cache_out = key, out
        return out

    def __getattr__(self, name):  # delegate .to / .cpu / .parameters / ...
        return getattr(self._dit, name)


class _CachedT5:
    """Stand-in for ``wan.WanTI2V.text_encoder`` that returns **precomputed** T5
    embeddings by string, so upstream's own positive/negative CFG path runs at
    inference without ever loading the 11 GB T5.

    Upstream (in ``t5_cpu`` mode) calls ``text_encoder([prompt], device)`` and
    expects a list of ``[L, C]`` tensors, then moves them to the DiT device — so
    this only needs to look each prompt up in a small ``{string: embedding}`` map.
    """

    def __init__(self, table: dict) -> None:
        self._table = table

    def __call__(self, prompts, device=None):  # noqa: ARG002 (device applied upstream)
        return [self._table[p] for p in prompts]


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
            # Frame-only regime: cached unconditional (empty-prompt) embedding.
            context = [self.wan._uncond_ctx.to(device)] * batch
        else:
            # Text-conditioned: a list of per-sample [Lᵢ, C] embeddings (or a
            # [B, L, C] tensor). Move to the DiT device; the DiT pads to text_len.
            if isinstance(context, Tensor):
                context = [context[i] for i in range(context.shape[0])]
            context = [c.to(device) for c in context]
            if len(context) != batch:
                raise ValueError(f"cond['context'] has {len(context)} entries but batch is {batch}.")

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
        context: Tensor | None = None,
        context_null: Tensor | None = None,
        max_area: int = 704 * 1280,
        frame_num: int = 121,
        sampling_steps: int = 50,
        shift: float = 5.0,
        guide_scale: float = 5.0,
        seed: int = 0,
        offload_model: bool | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Run Wan's native i2v rollout conditioned on the observation frame
        ``conditioning`` (a PIL image, HWC uint8 array, or ``[H,W,3]``/``[3,H,W]``
        tensor). Returns pixels ``[3, N, H, W]`` in ``[-1, 1]``.

        ``compose_fn is None`` -> byte-for-byte upstream output. Otherwise every
        DiT prediction is replaced by ``compose_fn(x, t, base_pred)`` (see
        :class:`~...models.base.video_model.BaseVideoModel.generate`).

        **Text conditioning.** ``context is None`` -> frame-only / prompt-free
        (both CFG branches use the cached unconditional embedding, so ``guide_scale``
        is inert and the redundant forward is memoized away). Pass a precomputed
        positive embedding ``context`` (``[L, C]``, from ``precompute_prompt_contexts``)
        to run **real CFG** at ``guide_scale``: the positive prompt vs ``context_null``
        (the negative embedding; defaults to the cached unconditional). No T5 is
        loaded — a :class:`_CachedT5` stub feeds the embeddings into upstream's own
        positive/negative loop.

        The DiT is always wrapped in :class:`_ComposedDiT`, which memoizes the
        redundant CFG forward when the two branches are identical (frame-only), so
        the base (and, when adapted, the adapter) run once per step there; real CFG
        (positive != negative) still runs both branches."""
        img = self._to_pil(conditioning)
        prompted = context is not None

        original = self.wan.model
        self.wan.model = _ComposedDiT(original, compose_fn)  # compose_fn None -> pass-through + memoize
        saved_te, saved_t5_cpu = self.wan.text_encoder, self.wan.t5_cpu
        input_prompt, n_prompt = None, ""
        if prompted:
            # Real CFG from cached embeddings: install a stub text encoder so
            # upstream's own positive/negative path runs without T5.
            neg = context_null if context_null is not None else self.wan._uncond_ctx
            self.wan.text_encoder = _CachedT5({"__pos__": context, "__neg__": neg})
            self.wan.t5_cpu = True  # take the branch that just looks up + moves to device
            input_prompt, n_prompt = "__pos__", "__neg__"
        try:
            video = self.wan.generate(
                input_prompt,  # None -> frame-only; "__pos__" -> stub lookup
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver="unipc",
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=self.offload_model if offload_model is None else offload_model,
            )
        finally:
            self.wan.model = original
            self.wan.text_encoder, self.wan.t5_cpu = saved_te, saved_t5_cpu
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
