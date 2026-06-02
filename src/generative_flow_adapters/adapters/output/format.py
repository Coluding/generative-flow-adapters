"""Shared output-format machinery for output adapters.

The point of this module is to make the *output parameterisation* an axis
that is orthogonal to the adapter backbone, so we can ask a clean research
question: holding the backbone fixed, does emitting an affine ``(scale,
shift)`` modulation of the base prediction beat emitting the residual delta
directly?

Two formats:

- ``direct``: the backbone emits ``C`` channels which are the delta added to
  the base output (``prediction = base + delta``).
- ``affine``: the backbone emits ``2C`` channels split into ``(scale,
  shift)``; the returned delta is ``base * scale + shift``. Because every
  output adapter returns ``output_kind="delta"`` and ``AdaptedModel`` adds it
  back, the realised transform is ``base * (1 + scale) + shift`` — a residual
  FiLM, identical in spirit to ``AffineOutputAdapter`` but driven by an
  arbitrary backbone.

Two granularities (affine only):

- ``dense``: per-element scale/shift maps (the full ``2C`` spatial/temporal
  tensor). Capacity-matched to a dense delta, so it is the fair "format only"
  comparison.
- ``channel``: one scale/shift per channel, pooled over space/time and
  broadcast back. Matches the lightweight FiLM of the original affine head.

Both formats are returned as ``output_kind="delta"`` so they compose as
residuals on the frozen base — that is what makes the comparison apples to
apples.
"""

from __future__ import annotations

from torch import Tensor

from generative_flow_adapters.adapters.output.interface import OutputAdapterResult

_OUTPUT_FORMAT_ALIASES = {
    "direct": "direct",
    "delta": "direct",
    "residual": "direct",
    "affine": "affine",
    "scale_shift": "affine",
    "film": "affine",
}

_AFFINE_GRANULARITY_ALIASES = {
    "dense": "dense",
    "spatial": "dense",
    "per_element": "dense",
    "channel": "channel",
    "per_channel": "channel",
    "global": "channel",
}


def normalize_output_format(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _OUTPUT_FORMAT_ALIASES:
        raise ValueError(
            f"Unsupported output_format: {value!r} (expected one of "
            f"{sorted(set(_OUTPUT_FORMAT_ALIASES.values()))})."
        )
    return _OUTPUT_FORMAT_ALIASES[key]


def normalize_affine_granularity(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _AFFINE_GRANULARITY_ALIASES:
        raise ValueError(
            f"Unsupported affine_granularity: {value!r} (expected one of "
            f"{sorted(set(_AFFINE_GRANULARITY_ALIASES.values()))})."
        )
    return _AFFINE_GRANULARITY_ALIASES[key]


def output_channel_multiplier(output_format: str) -> int:
    """How many channels the backbone must emit per base channel.

    ``direct`` → 1 (the delta). ``affine`` → 2 (scale and shift).
    """
    return 2 if normalize_output_format(output_format) == "affine" else 1


def build_output_result(
    raw: Tensor,
    reference: Tensor | None,
    *,
    output_format: str,
    affine_granularity: str = "dense",
    channel_dim: int = 1,
    gate: Tensor | None = None,
) -> OutputAdapterResult:
    """Turn a backbone's raw output tensor into a residual ``OutputAdapterResult``.

    ``raw`` has ``C`` channels for ``direct`` and ``2C`` for ``affine`` along
    ``channel_dim``. ``reference`` is the (detached) base output the affine
    transform modulates; it is ignored for ``direct``. Both formats return
    ``output_kind="delta"``. ``gate`` (if given) is attached unchanged for the
    mixing layer to consume under ``gated_residual``/``mask_mix`` composition;
    the format helper never applies it — gating is the composition's job.
    """
    fmt = normalize_output_format(output_format)
    if fmt == "direct":
        # Spatial backbones emit a full-rank delta (broadcast is a no-op); the
        # mlp backbone emits a [B, C] per-channel delta that must broadcast over
        # the reference's spatial/temporal dims.
        delta = raw if reference is None else _broadcast_to_rank(raw, reference.dim())
        return OutputAdapterResult(adapter_output=delta, output_kind="delta", gate=gate)

    if reference is None:
        raise ValueError("affine output_format requires a reference tensor (base_output or x_t).")

    granularity = normalize_affine_granularity(affine_granularity)
    channels = raw.shape[channel_dim]
    if channels % 2 != 0:
        raise ValueError(
            f"affine output_format expects an even channel count to split into "
            f"(scale, shift); got {channels} channels on dim {channel_dim}."
        )
    scale, shift = raw.chunk(2, dim=channel_dim)

    if granularity == "channel":
        scale = _pool_spatial(scale, channel_dim)
        shift = _pool_spatial(shift, channel_dim)

    scale = _broadcast_to_rank(scale, reference.dim())
    shift = _broadcast_to_rank(shift, reference.dim())
    delta = reference * scale + shift
    return OutputAdapterResult(adapter_output=delta, output_kind="delta", gate=gate)


def prepare_gate(
    gate_raw: Tensor,
    reference: Tensor,
    *,
    granularity: str = "dense",
    channel_dim: int = 1,
) -> Tensor:
    """Shape a backbone's raw gate channels to broadcast against the delta/base.

    ``channel`` pools the gate to one value per channel (then broadcasts over
    space/time); ``dense`` keeps it per-element. The raw (pre-sigmoid) gate is
    returned — the mixing layer applies ``sigmoid(gate + gate_bias)``.
    """
    gate = gate_raw
    if normalize_affine_granularity(granularity) == "channel":
        gate = _pool_spatial(gate, channel_dim)
    return _broadcast_to_rank(gate, reference.dim())


def _pool_spatial(tensor: Tensor, channel_dim: int) -> Tensor:
    """Average over every dim except batch (0) and channel, keeping rank."""
    reduce_dims = [d for d in range(tensor.dim()) if d not in (0, channel_dim)]
    if not reduce_dims:
        return tensor
    return tensor.mean(dim=reduce_dims, keepdim=True)


def _broadcast_to_rank(tensor: Tensor, rank: int) -> Tensor:
    """Append trailing singleton dims so ``tensor`` broadcasts against a
    higher-rank reference (e.g. a ``[B, C]`` vector against ``[B, C, T, H, W]``)."""
    while tensor.dim() < rank:
        tensor = tensor.unsqueeze(-1)
    return tensor
