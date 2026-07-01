"""Contract tests for the real-backbone **compositional** adapter (the contribution).

These exercise the compositional wiring added for "wire the multimodal adapter to
a real backbone" — one AVID output adapter *per modality* emitting a per-modality
video adjustment Δ_m, blended with the base + action adapter by the learned mask
m ∈ ℝ^{n+2} (``LearnedMaskFusion``), plus the bidirectional video↔modality
coupling (``ModalityEncoder`` / ``VideoReadout``). They run *without* the 4.4GB
DynamiCrafter checkpoint or a GPU: lightweight fake adapters stand in for the 11M
AVID adapters, reproducing the contract the real ones honour (consume
``cond['context']``, honour the fixed text/image token split, return an
``OutputAdapterResult`` prediction + gate).

What is asserted (docs/composite (2).png):
- each modality adapter sees ONLY its own tokens appended to context (one-adapter-
  per-modality, no modality↔modality coupling); the action adapter sees none;
- the text boundary (first 77 tokens) is never shifted;
- the learned mask has n+2 slots, is a normalised softmax, and receives gradient;
- both coupling directions get gradient (encoders via context, readout via heads).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from generative_flow_adapters.adapters.output.interface import OutputAdapterResult
from generative_flow_adapters.multimodal.fusion import LearnedMaskFusion
from generative_flow_adapters.multimodal.modality_adapter import ModalityPredictionHead
from generative_flow_adapters.multimodal.modality_encoder import ModalityEncoder, VideoReadout
from generative_flow_adapters.multimodal.model import MultiModalAdaptedModel
from generative_flow_adapters.multimodal.spec import OutputModalitySpec

TEXT_CONTEXT_LEN = 77
IMAGE_TOKENS = 16
CONTEXT_DIM = 1024
LATENT_C, T, H, W = 4, 4, 8, 8
COND_DIM = 32


class _DummyVideoBase(nn.Module):
    """Frozen video prior stub: returns a (B, C, T, H, W) prediction."""

    model_type = "diffusion"
    prediction_type = "velocity"
    diffusion_schedule_config = None

    def forward(self, x_t: Tensor, t: Tensor, cond: object | None = None) -> Tensor:
        return torch.zeros_like(x_t)


class _FakeAVIDAdapter(nn.Module):
    """Stands in for one DynamiCrafter 11M output adapter.

    Honours the real contract: splits ``context`` at ``text_context_len`` into a
    text and an image stream (any appended modality tokens ride the image stream),
    makes its prediction depend on the appended tokens so gradients flow back, and
    returns a full prediction (+ gate) — the compositional fusion uses the global
    mask, not this gate.
    """

    def __init__(self) -> None:
        super().__init__()
        self.text_context_len = TEXT_CONTEXT_LEN
        self.seen_context_tokens: int | None = None
        self.seen_text: Tensor | None = None
        self.image_proj = nn.Linear(CONTEXT_DIM, LATENT_C)
        self.gate_param = nn.Parameter(torch.zeros(1))

    def forward(self, x_t, t, cond, base_output=None):
        context = cond["context"]
        self.seen_context_tokens = int(context.shape[1])
        text, image = context[:, : self.text_context_len, :], context[:, self.text_context_len :, :]
        self.seen_text = text
        summary = self.image_proj(image.mean(dim=1))  # (B, LATENT_C) — depends on appended tokens
        pred = x_t + summary[:, :, None, None, None]
        gate = self.gate_param.expand(x_t.shape[0], 1, 1, 1, 1).expand_as(x_t)
        return OutputAdapterResult(adapter_output=pred, output_kind="prediction", gate=gate)


class _PerFrameEncoder(nn.Module):
    """Stand-in for a per-frame action encoder: returns (B, T, cond_dim)."""

    def forward(self, cond) -> Tensor:
        act = cond["act"]  # (B, A)
        return act.new_zeros(act.shape[0], T, COND_DIM)


def _build_model(condition_encoder=None):
    specs = [
        OutputModalitySpec(name="video", kind="video", has_frozen_prior=True, loss_weight=1.0),
        OutputModalitySpec(name="proprio", kind="vector", feature_shape=[7], loss_weight=0.5),
        OutputModalitySpec(name="tactile", kind="map", feature_shape=[2, 8, 8], loss_weight=0.2),
    ]
    adapter_specs = [s for s in specs if not s.has_frozen_prior]
    return MultiModalAdaptedModel(
        base_model=_DummyVideoBase(),
        video_adapter=_FakeAVIDAdapter(),  # ε_adj — the action adapter
        modality_heads={
            s.name: ModalityPredictionHead(s.feature_shape, COND_DIM, hidden_dim=32)
            for s in adapter_specs
        },
        modality_specs=specs,
        condition_encoder=condition_encoder,
        fusion=LearnedMaskFusion(1 + len(adapter_specs)),  # m ∈ ℝ^{n+2}
        modality_video_adapters={s.name: _FakeAVIDAdapter() for s in adapter_specs},  # Δ_m
        modality_encoders={
            s.name: ModalityEncoder(s.feature_shape, context_dim=CONTEXT_DIM, hidden_dim=32)
            for s in adapter_specs
        },
        video_readout=VideoReadout(LATENT_C, COND_DIM),
    )


def _inputs(batch=2):
    x_t = {
        "video": torch.randn(batch, LATENT_C, T, H, W),
        "proprio": torch.randn(batch, T, 7),
        "tactile": torch.randn(batch, T, 2, 8, 8),
    }
    t = {k: torch.randint(0, 1000, (batch,)) for k in x_t}
    context = torch.randn(batch, TEXT_CONTEXT_LEN + IMAGE_TOKENS, CONTEXT_DIM)
    return x_t, t, {"context": context, "act": torch.randn(batch, 4)}


def test_compositional_per_modality_token_routing():
    """Each modality adapter sees ONLY its own tokens; the action adapter none."""
    model = _build_model()
    x_t, t, cond = _inputs()
    original_text = cond["context"][:, :TEXT_CONTEXT_LEN, :].clone()

    preds = model(x_t, t, cond)

    assert preds["video"].shape == x_t["video"].shape
    assert preds["proprio"].shape == x_t["proprio"].shape
    assert preds["tactile"].shape == x_t["tactile"].shape

    base_tokens = TEXT_CONTEXT_LEN + IMAGE_TOKENS
    # Action adapter: context untouched (no modality tokens).
    assert model.video_adapter.seen_context_tokens == base_tokens
    # Each modality adapter: base context + ONLY its own T tokens.
    assert model.modality_video_adapters["proprio"].seen_context_tokens == base_tokens + T
    assert model.modality_video_adapters["tactile"].seen_context_tokens == base_tokens + T
    # Text boundary preserved everywhere.
    assert torch.allclose(model.modality_video_adapters["proprio"].seen_text, original_text)


def test_learned_mask_blends_n_plus_2_streams():
    model = _build_model()
    x_t, t, cond = _inputs()
    fusion = model.fusion
    assert isinstance(fusion, LearnedMaskFusion)
    # n+2 slots: base + action + (proprio, tactile) video adjustments.
    assert fusion.logits.numel() == 4
    preds = model(x_t, t, cond)
    weights = fusion.mask_weights()
    assert abs(float(weights.sum()) - 1.0) < 1e-5
    assert torch.isfinite(preds["video"]).all()


def test_per_frame_conditioning_broadcasts_video_readout():
    """m←video must broadcast the per-sample video feature across a per-frame
    (B, T, cond_dim) conditioning embedding (regression: dim-1 broadcast clash)."""
    model = _build_model(condition_encoder=_PerFrameEncoder())
    x_t, t, cond = _inputs()
    preds = model(x_t, t, cond)  # would raise RuntimeError before the broadcast fix
    assert preds["proprio"].shape == x_t["proprio"].shape
    assert preds["tactile"].shape == x_t["tactile"].shape


def test_evaluator_rollout_shapes_and_npz(tmp_path):
    """Eval rollout samples the modality trajectory and dumps pred/gt npz."""
    import numpy as np

    from generative_flow_adapters.losses.diffusion import DiffusionTrainingObjective
    from generative_flow_adapters.multimodal.eval import MultiModalEvaluator

    torch.manual_seed(0)
    specs = [
        OutputModalitySpec(name="video", kind="video", has_frozen_prior=True),
        OutputModalitySpec(name="proprio", kind="vector", feature_shape=[7], visualize=True),
    ]
    adapter_specs = [s for s in specs if not s.has_frozen_prior]
    model = MultiModalAdaptedModel(
        base_model=_DummyVideoBase(),
        video_adapter=_FakeAVIDAdapter(),
        modality_heads={
            s.name: ModalityPredictionHead(s.feature_shape, COND_DIM, hidden_dim=16) for s in adapter_specs
        },
        modality_specs=specs,
        condition_encoder=None,
        fusion=LearnedMaskFusion(1 + len(adapter_specs)),
        modality_video_adapters={s.name: _FakeAVIDAdapter() for s in adapter_specs},
        modality_encoders={
            s.name: ModalityEncoder(s.feature_shape, context_dim=CONTEXT_DIM, hidden_dim=16) for s in adapter_specs
        },
        video_readout=VideoReadout(LATENT_C, COND_DIM),
    )
    objective = DiffusionTrainingObjective(timesteps=20, beta_schedule="linear", linear_start=1e-4, linear_end=2e-2)
    batch = {
        "targets": {"video": torch.randn(2, LATENT_C, T, H, W), "proprio": torch.randn(2, T, 7)},
        "cond": {"act": torch.randn(2, 4)},
    }
    evaluator = MultiModalEvaluator(
        model=model, objective=objective, video_objective=objective, prediction_type="velocity",
        specs=specs, codecs=None, eval_batch=batch, every_n_steps=5, num_inference_steps=4,
        video_cond_t=5, num_samples=2, wandb_logger=None, out_dir=str(tmp_path),
    )
    evaluator.maybe_eval(step=3)  # not a multiple of 5 -> no-op
    assert not (tmp_path / "eval").exists()
    evaluator.maybe_eval(step=5)  # fires
    saved = tmp_path / "eval" / "step5_proprio.npz"
    assert saved.exists()
    data = np.load(saved)
    assert data["pred"].shape == (2, T, 7)
    assert data["gt"].shape == (2, T, 7)


def test_bidirectional_and_mask_gradients_flow():
    model = _build_model()
    x_t, t, cond = _inputs()
    preds = model(x_t, t, cond)
    loss = sum(p.pow(2).mean() for p in preds.values())
    loss.backward()

    # video←m: per-modality encoders received gradient (through their adapter's context).
    for name, enc in model.modality_encoders.items():
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert grads, f"ModalityEncoder[{name}] received no gradient"
    # m←video: the video readout received gradient (through the modality heads).
    assert any(p.grad is not None for p in model.video_readout.parameters()), "VideoReadout got no grad"
    # the learned mask received gradient.
    assert model.fusion.logits.grad is not None, "LearnedMaskFusion mask got no grad"
