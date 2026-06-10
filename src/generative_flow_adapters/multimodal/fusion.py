"""Video-stream fusion strategies.

The fusion module owns how the video prediction is assembled from the frozen
base output ``ε_pre`` and the trainable contributions (the action adapter's
video adjustment ``ε_adj`` and, in the compositional variant, each modality's
video adjustment ``ε_adj,i``). Modality *predictions* are not fused here — they
go straight to their own per-stream losses.

- ``TrivialFusion`` — `ε_video = ε_pre + Σ contributions`. The Phase-2 substrate
  bring-up wiring (and a degenerate baseline). Identity at init since every
  contribution head is zero-initialised.
- ``LearnedMaskFusion`` — `ε_video = Σ m_k ⊙ stack_k` with a learned vector mask
  `m ∈ ℝ^{n+2}` (softmax over {base, action, modality₁…ₙ}); the compositional
  adapter from ``docs/composite (2).png`` (Phase 3).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class TrivialFusion(nn.Module):
    def forward(self, base_output: Tensor, contributions: Sequence[Tensor]) -> Tensor:
        out = base_output
        for contribution in contributions:
            out = out + contribution
        return out


class LearnedMaskFusion(nn.Module):
    """Softmax-weighted blend over {base, action, modality₁…ₙ} video streams.

    ``num_streams`` counts the trainable contributions (action + one per
    modality that emits a video adjustment); the base adds one more slot, giving
    the ``m ∈ ℝ^{n+2}`` mask of the decision note. The mask is a free parameter
    (one scalar weight per stream, shared across batch/space) initialised so the
    base dominates and contributions start ~off, then learned. A per-element
    mask is a later refinement.
    """

    def __init__(self, num_contributions: int, base_bias: float = 4.0) -> None:
        super().__init__()
        # Logits over [base, contribution_0, ..., contribution_{n-1}].
        logits = torch.zeros(num_contributions + 1)
        logits[0] = base_bias  # start with the frozen base dominating the blend
        self.logits = nn.Parameter(logits)

    def forward(self, base_output: Tensor, contributions: Sequence[Tensor]) -> Tensor:
        if len(contributions) + 1 != self.logits.numel():
            raise ValueError(
                f"LearnedMaskFusion expected {self.logits.numel() - 1} contributions, "
                f"got {len(contributions)}."
            )
        weights = torch.softmax(self.logits, dim=0)
        out = weights[0] * base_output
        for i, contribution in enumerate(contributions):
            out = out + weights[i + 1] * contribution
        return out

    @torch.no_grad()
    def mask_weights(self) -> Tensor:
        """Current softmax mask ``m`` — for logging / "modalities used" checks."""
        return torch.softmax(self.logits, dim=0).detach().cpu()
