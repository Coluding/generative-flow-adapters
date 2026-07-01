"""Output-side modality specification.

The output-side dual of :class:`generative_flow_adapters.config.ConditionSpec`.
A :class:`OutputModalitySpec` declares one *output stream* of the multimodal
world model — the thing the model is trained to predict the future of. The
video stream is the only one carried by a frozen prior; every other modality
(proprio, depth, tactile, ...) is predicted from scratch by a trainable
modality adapter.

See thesis-vault ``50_Decisions/open/multimodal-adapter-broadening.md`` — this
is item 4 of the shared substrate ("output modality spec").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Recognised stream kinds. ``video`` routes to the existing output-adapter
# factory (DynamiCrafter UNet / OutputHead); ``vector``/``map`` route to the
# lightweight per-modality heads in ``multimodal.modality_adapter``.
VIDEO_KIND = "video"
VECTOR_KIND = "vector"
MAP_KIND = "map"
KNOWN_KINDS = frozenset({VIDEO_KIND, VECTOR_KIND, MAP_KIND})


@dataclass(slots=True)
class OutputModalitySpec:
    """One predicted output stream.

    Attributes:
        name: batch key the raw modality arrives under (e.g. ``"proprio"``)
            and the key its target/prediction is stored under downstream.
        kind: ``video`` | ``vector`` | ``map`` — selects the prediction head.
        has_frozen_prior: ``True`` only for the video stream, which the frozen
            base already denoises. Modality streams are ``False`` (no prior;
            the adapter predicts the whole target — composition is adapter-only).
        loss_weight: ``w_m`` — fixed scalar weight on this stream's denoising
            loss in the summed objective (UWM-style; sub-decision 1).
        codec: ``identity`` (raw, optionally normalised) | ``vae`` (frozen
            video VAE latent). Selects how raw clip data becomes a target.
        feature_shape: per-frame target shape *after* the codec, excluding the
            leading ``(B, T)`` axes. Required for ``vector``/``map`` heads
            (e.g. ``(7,)`` for proprio, ``(2, 64, 64)`` for tactile). For
            ``video`` it is inferred by the output adapter and may be empty.
        hidden_dim: width of this modality's prediction head.
        visualize: when ``True``, the eval loop rolls out this modality (full
            reverse diffusion) and logs predicted-vs-ground-truth at the eval
            cadence. Best for low-dim streams (e.g. ``proprio``); video is not
            supported here (use the video eval grid instead).
        adapter: extra knobs for the ``video`` head (forwarded into the output
            adapter factory's ``AdapterConfig.extra``); unused for other kinds.
        codec_kwargs: extra knobs for the codec (e.g. ``mean``/``std`` for
            ``identity``, ``resize`` for maps).
        extra: catch-all for forward-compatible knobs.
    """

    name: str
    kind: str
    has_frozen_prior: bool = False
    loss_weight: float = 1.0
    codec: str = "identity"
    feature_shape: tuple[int, ...] = ()
    hidden_dim: int = 128
    visualize: bool = False
    adapter: dict[str, Any] = field(default_factory=dict)
    codec_kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KNOWN_KINDS:
            raise ValueError(
                f"OutputModalitySpec.kind={self.kind!r} is not one of {sorted(KNOWN_KINDS)}."
            )
        # Normalise YAML lists to tuples so the shape is hashable / comparable.
        if not isinstance(self.feature_shape, tuple):
            self.feature_shape = tuple(int(d) for d in self.feature_shape)
        if self.kind in {VECTOR_KIND, MAP_KIND} and not self.feature_shape:
            raise ValueError(
                f"OutputModalitySpec({self.name!r}) of kind={self.kind!r} requires a "
                "non-empty feature_shape (per-frame target shape, e.g. (7,) or (2, 64, 64))."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputModalitySpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown OutputModalitySpec fields: {sorted(unknown)}")
        payload = dict(data)
        if "feature_shape" in payload and payload["feature_shape"] is not None:
            payload["feature_shape"] = tuple(int(d) for d in payload["feature_shape"])
        return cls(**payload)
