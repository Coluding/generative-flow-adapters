from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from generative_flow_adapters.models.base.interfaces import BaseGenerativeModel, infer_prediction_type


class Wan21DiTWrapper(BaseGenerativeModel):
    """Thin wrapper around the vendored Wan2.1 ``WanModel`` DiT.

    Wan2.1 is a single-stage rectified-flow (flow-matching) video DiT that
    predicts velocity. The backbone is exposed through the common
    ``forward(x_t, t, cond) -> velocity`` contract so it composes additively
    with adapters exactly like the DynamiCrafter U-Net.

    The base stays a pure ``(x_t, t)`` map: text context is optional (a null
    zero-context is synthesised when absent), and **all action conditioning is
    expected to live in the adapter**, not here. That keeps the frozen base
    identical to the pretrained checkpoint.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        model_type: str = "flow",
        prediction_type: str | None = None,
    ) -> None:
        super().__init__(model_type=model_type, prediction_type=infer_prediction_type(model_type, prediction_type))
        self.module = module

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
    ) -> "Wan21DiTWrapper":
        from generative_flow_adapters.backbones.wan.modules.model import WanModel

        params = _load_wan_params(wan_config_path)
        module = WanModel(**params)
        wrapper = cls(module=module, model_type=model_type, prediction_type=prediction_type)

        if checkpoint_path:
            if not (allow_missing_checkpoint and not Path(checkpoint_path).exists()):
                wrapper.load_checkpoint(checkpoint_path=checkpoint_path, strict=strict_checkpoint)

        if dtype is not None:
            # Cast after weight load so checkpoint tensors downcast in one shot.
            wrapper.module.to(dtype)
            wrapper._param_dtype = dtype
        return wrapper

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load Wan DiT weights.

        Accepts either a single ``.pth``/``.safetensors`` file or a HF-style
        directory of sharded ``*.safetensors``. Keys map directly onto
        ``WanModel`` (``patch_embedding.*``, ``blocks.N.*``, ``head.*``); a
        couple of candidate prefixes are tried for robustness.
        """
        state_dict = _load_state_dict(checkpoint_path)
        candidate_prefixes = ("", "model.", "model.diffusion_model.")
        selected = state_dict
        for prefix in candidate_prefixes:
            if not prefix:
                if any(k.startswith("patch_embedding") or k.startswith("blocks.") for k in state_dict):
                    selected = state_dict
                    break
                continue
            filtered = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if filtered:
                selected = filtered
                break
        self.module.load_state_dict(selected, strict=strict)

    @property
    def dtype(self) -> torch.dtype:
        explicit = getattr(self, "_param_dtype", None)
        if explicit is not None:
            return explicit
        return next(self.module.parameters()).dtype

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        """``x_t``: ``[B, C_in, T, H, W]`` -> velocity ``[B, C_out, T, H, W]``.

        ``cond`` may carry a precomputed text ``context`` (a ``[B, L, text_dim]``
        tensor or a list of ``[L, text_dim]``); otherwise a null zero-context is
        used (the action-conditioned world-model case).
        """
        if x_t.dim() != 5:
            raise ValueError(f"Wan21DiTWrapper expects x_t of shape [B,C,T,H,W], got {tuple(x_t.shape)}")

        target_dtype = self.dtype
        if x_t.is_floating_point() and x_t.dtype != target_dtype:
            x_t = x_t.to(target_dtype)

        batch = x_t.shape[0]
        device = x_t.device
        x_list = [x_t[i] for i in range(batch)]

        seq_len = self._seq_len(x_t.shape[2:])
        t = self._prepare_timesteps(t, batch, device)
        context = self._prepare_context(cond, batch, device, target_dtype)

        out_list = self.module(x_list, t=t, context=context, seq_len=seq_len)
        out = torch.stack(out_list, dim=0)
        if out.dtype != target_dtype:
            out = out.to(target_dtype)
        return out

    def _seq_len(self, spatial_shape) -> int:
        pt, ph, pw = self.module.patch_size
        f, h, w = (int(s) for s in spatial_shape)
        if f % pt or h % ph or w % pw:
            raise ValueError(
                f"latent grid {(f, h, w)} not divisible by Wan patch_size {(pt, ph, pw)}"
            )
        return (f // pt) * (h // ph) * (w // pw)

    @staticmethod
    def _prepare_timesteps(t: Tensor, batch: int, device: torch.device) -> Tensor:
        t = torch.as_tensor(t, device=device)
        if t.dim() == 0:
            t = t.expand(batch)
        t = t.reshape(-1)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch)
        # Wan's sinusoidal time embedding runs in fp32 internally.
        return t.float()

    def _prepare_context(self, cond, batch, device, dtype) -> list[Tensor]:
        text_dim = self.module.text_dim
        context = cond.get("context") if isinstance(cond, dict) else None

        if context is None:
            # Null context: a single zero token per sample, padded to text_len
            # by the backbone. Keeps the frozen base a pure (x_t, t) map.
            return [torch.zeros(1, text_dim, device=device, dtype=dtype) for _ in range(batch)]

        if isinstance(context, Tensor):
            if context.dim() == 2:  # [L, C] shared across batch
                context = context.unsqueeze(0).expand(batch, -1, -1)
            context = context.to(device=device, dtype=dtype)
            return [context[i] for i in range(batch)]

        if isinstance(context, (list, tuple)):
            return [c.to(device=device, dtype=dtype) for c in context]

        raise TypeError(f"Unsupported context type for Wan backbone: {type(context)!r}")


def _load_wan_params(config_path: str) -> dict[str, Any]:
    """Read ``WanModel`` kwargs from ``model.params.dit_config.params`` (falling
    back to ``model.params``) of a YAML config, mirroring the DynamiCrafter
    ``_load_unet_params`` pattern."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    model_params = raw.get("model", {}).get("params", {})
    dit_cfg = model_params.get("dit_config", {})
    params = dit_cfg.get("params")
    if not isinstance(params, dict):
        # Fall back to a flat `model.params` of WanModel kwargs.
        params = {k: v for k, v in model_params.items() if k != "dit_config"}
    if not params:
        raise ValueError(f"Could not find Wan DiT params in {path}")

    params = dict(params)
    if "patch_size" in params and isinstance(params["patch_size"], list):
        params["patch_size"] = tuple(params["patch_size"])
    if "window_size" in params and isinstance(params["window_size"], list):
        params["window_size"] = tuple(params["window_size"])
    return params


def _load_state_dict(checkpoint_path: str) -> dict[str, Tensor]:
    path = Path(checkpoint_path)
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No *.safetensors shards found in {path}")
        from safetensors.torch import load_file

        merged: dict[str, Tensor] = {}
        for shard in shards:
            merged.update(load_file(str(shard)))
        return merged
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    state = torch.load(str(path), map_location="cpu")
    return state.get("state_dict", state)
