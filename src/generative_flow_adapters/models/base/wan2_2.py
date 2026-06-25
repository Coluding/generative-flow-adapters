from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from generative_flow_adapters.models.base.wan import (
    Wan21DiTWrapper,
    _load_wan_params,
)


class Wan22DiTWrapper(Wan21DiTWrapper):
    """Frozen base wrapper for the **Wan2.2 TI2V** DiT (``modules.model2_2``).

    Wan2.2 TI2V-5B is the unified text+image-to-video model. It differs from the
    Wan2.1 wrapper in two ways that matter here:

    - The vendored model lives in ``modules.model2_2`` (``model_type='ti2v'``,
      ``in_dim = z_dim = 48``). Image conditioning uses **neither CLIP nor a
      concat channel** — the observation frame is conditioned purely by clamping
      its clean latent into the sequence and giving those tokens timestep 0
      ("diffusion forcing"). So the wrapper stays a pure ``(x_t, t) -> velocity``
      map; the clamping + per-token timestep are built by the preprocessor /
      sampler and threaded through ``t``.

    - Wan2.2's forward consumes a **per-token** timestep ``[B, seq_len]`` (it
      flattens it into a per-token time embedding). This wrapper accepts the
      ergonomic **per-latent-frame** form ``[B, T']`` (e.g. ``frame 0 = 0`` for
      the observation, the rest ``= sigma*scale``) and expands it across each
      frame's patch tokens, as well as the scalar/``[B]`` uniform form and the
      raw per-token ``[B, seq_len]`` form.
    """

    @classmethod
    def from_config(
        cls,
        wan_config_path: str,
        model_type: str = "flow",
        checkpoint_path: str | None = None,
        prediction_type: str | None = None,
        strict_checkpoint: bool = False,
        allow_missing_checkpoint: bool = False,
        dtype: torch.dtype | None = None,
    ) -> "Wan22DiTWrapper":
        # Import the Wan2.2 DiT (per-token timestep, ti2v) rather than the 2.1 one.
        from generative_flow_adapters.backbones.wan.modules.model2_2 import WanModel

        params = _load_wan_params(wan_config_path)
        module = WanModel(**params)
        wrapper = cls(module=module, model_type=model_type, prediction_type=prediction_type)

        if checkpoint_path:
            if not (allow_missing_checkpoint and not Path(checkpoint_path).exists()):
                wrapper.load_checkpoint(checkpoint_path=checkpoint_path, strict=strict_checkpoint)

        if dtype is not None:
            wrapper.module.to(dtype)
            wrapper._param_dtype = dtype
        return wrapper

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        """``x_t``: ``[B, 48, T, H, W]`` -> velocity ``[B, 48, T, H, W]``.

        ``t`` may be a scalar / ``[B]`` (uniform timestep), ``[B, T']`` (one
        timestep per latent frame — the diffusion-forcing form), or already
        ``[B, seq_len]`` (per patch token). All are normalised to the per-token
        ``[B, seq_len]`` shape Wan2.2 expects.
        """
        if x_t.dim() != 5:
            raise ValueError(f"Wan22DiTWrapper expects x_t of shape [B,C,T,H,W], got {tuple(x_t.shape)}")

        target_dtype = self.dtype
        if x_t.is_floating_point() and x_t.dtype != target_dtype:
            x_t = x_t.to(target_dtype)

        batch = x_t.shape[0]
        device = x_t.device
        x_list = [x_t[i] for i in range(batch)]

        seq_len = self._seq_len(x_t.shape[2:])
        t_tokens = self._prepare_token_timesteps(t, x_t.shape[2:], batch, device)
        context = self._prepare_context(cond, batch, device, target_dtype)

        out_list = self.module(x_list, t=t_tokens, context=context, seq_len=seq_len)
        out = torch.stack(out_list, dim=0)
        if out.dtype != target_dtype:
            out = out.to(target_dtype)
        return out

    def _prepare_token_timesteps(self, t: Tensor, spatial_shape, batch: int, device: torch.device) -> Tensor:
        """Normalise ``t`` to a per-token timestep ``[B, seq_len]``.

        Token order is frame-major (each latent frame contributes
        ``H'*W'`` patch tokens, ``H'=H//ph``), matching the model's
        ``flatten(2)`` over the patch grid — so a per-frame timestep expands by
        ``repeat_interleave`` over ``tokens_per_frame``.
        """
        pt, ph, pw = self.module.patch_size
        f, h, w = (int(s) for s in spatial_shape)
        t_frames = f // pt
        tokens_per_frame = (h // ph) * (w // pw)
        seq_len = t_frames * tokens_per_frame

        t = torch.as_tensor(t, device=device).float()
        if t.dim() == 0:
            t = t.reshape(1)

        # Per-token already: [B, seq_len] (or shared [1, seq_len]).
        if t.dim() == 2 and t.shape[1] == seq_len:
            return t.expand(batch, -1) if t.shape[0] == 1 else t

        # Per-latent-frame: [B, T'] (or shared [1, T'] / bare [T']) -> per token.
        if t.dim() == 2 and t.shape[1] == t_frames:
            per_token = t.repeat_interleave(tokens_per_frame, dim=1)
            return per_token.expand(batch, -1) if per_token.shape[0] == 1 else per_token
        if t.dim() == 1 and t.shape[0] == t_frames and t_frames != batch:
            per_token = t.repeat_interleave(tokens_per_frame).unsqueeze(0)
            return per_token.expand(batch, -1)

        # Uniform per-batch [B] (or scalar) -> same timestep on every token.
        t = t.reshape(-1)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch)
        return t.unsqueeze(1).expand(batch, seq_len)
