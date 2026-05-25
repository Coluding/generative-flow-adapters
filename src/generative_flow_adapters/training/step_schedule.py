"""Configurable shortcut step-size schedule.

Shortcut models condition on a *step size* — how far along the trajectory a
single model call should jump. This module turns a config block into a
schedule of such step sizes, canonicalised to a **normalised fraction of the
trajectory** ``s ∈ (0, 1]`` (paper convention: ``s = 1`` is one-step
generation, ``s = 1/128`` the finest step). Keeping one canonical scale lets
the same schedule drive training (sample an ``s`` per batch) and the eval grid
(enumerate the levels, with sampling-step count ``N = round(1/s)``), and lets
configs be written in whichever units are natural — normalised ``(0,1]`` or
raw timesteps ``[1, T]``.

Config block (``training.extra.shortcut_step_schedule``)::

    shortcut_step_schedule:
      units: normalized      # "normalized" (0,1]  |  "timesteps" [1, T]
      mode: log2             # "log2" | "explicit" | "uniform"
      min: 1/128             # log2 / uniform bounds, in `units` (fractions ok)
      max: 1
      base: 2                # log2 only
      # values: [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]   # explicit only
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import torch


def _to_float(value: object) -> float:
    """Parse a config scalar to ``float``, accepting fraction strings.

    YAML parses ``1/128`` as the string ``"1/128"`` (not a number), so the
    schedule can be written with readable fractions — ``min: 1/128`` instead
    of ``0.0078125`` — and explicit ``values: [1/128, 1/64, …, 1]``. Plain
    ints/floats and decimal/scientific strings still work.
    """
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            return float(text)  # e.g. "1e-2"; let a bad value raise here
    return float(value)  # type: ignore[arg-type]


@dataclass(slots=True, frozen=True)
class ShortcutStepSchedule:
    """A schedule of normalised step sizes ``s ∈ (0, 1]``.

    ``levels`` is the sorted (ascending) discrete support for ``log2`` /
    ``explicit`` modes, or ``None`` for the continuous ``uniform`` mode.
    ``low``/``high`` are the normalised bounds. ``timesteps`` is the base
    process length ``T`` used to convert ``s`` to integer timestep jumps.
    """

    mode: str
    low: float
    high: float
    timesteps: int
    levels: tuple[float, ...] | None

    # ------------------------------------------------------------------ parsing

    @classmethod
    def from_config(cls, raw: Mapping[str, object], *, timesteps: int) -> "ShortcutStepSchedule":
        if not isinstance(raw, Mapping):
            raise TypeError(f"shortcut_step_schedule must be a mapping; got {type(raw).__name__}.")
        t = int(timesteps)
        if t <= 0:
            raise ValueError(f"timesteps must be positive; got {t}.")
        units = str(raw.get("units", "normalized")).lower()
        if units not in {"normalized", "normalised", "timesteps"}:
            raise ValueError(f"shortcut_step_schedule.units must be 'normalized' or 'timesteps'; got {units!r}.")
        mode = str(raw.get("mode", "log2")).lower()

        def to_norm(value: object) -> float:
            v = _to_float(value)
            if units == "timesteps":
                v = v / t
            # Clamp into (0, 1]: a timestep value of T maps to 1.0 (one-step);
            # guard against 0 / negatives and floating drift above 1.
            return min(max(v, 1.0 / t), 1.0)

        if mode == "explicit":
            values = raw.get("values")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) == 0:
                raise ValueError("shortcut_step_schedule mode='explicit' requires a non-empty 'values' list.")
            levels = tuple(sorted({to_norm(v) for v in values}))
            return cls(mode="explicit", low=levels[0], high=levels[-1], timesteps=t, levels=levels)

        default_min = 1.0 if units == "timesteps" else 1.0 / 128
        default_max = float(t) if units == "timesteps" else 1.0
        low = to_norm(raw.get("min", default_min))
        high = to_norm(raw.get("max", default_max))
        if low > high:
            low, high = high, low

        if mode == "uniform":
            return cls(mode="uniform", low=low, high=high, timesteps=t, levels=None)

        if mode == "log2":
            base = _to_float(raw.get("base", 2))
            if base <= 1.0:
                raise ValueError(f"shortcut_step_schedule.base must be > 1; got {base}.")
            levels_list: list[float] = []
            v = low
            # Geometric ladder low·base^k up to high (inclusive within tolerance).
            while v <= high * (1.0 + 1e-9):
                levels_list.append(min(v, 1.0))
                v *= base
            if not levels_list or levels_list[-1] < high * (1.0 - 1e-9):
                levels_list.append(high)
            levels = tuple(sorted(set(levels_list)))
            return cls(mode="log2", low=levels[0], high=levels[-1], timesteps=t, levels=levels)

        raise ValueError(f"shortcut_step_schedule.mode must be 'log2', 'explicit', or 'uniform'; got {mode!r}.")

    # ------------------------------------------------------------------ queries

    def smallest(self) -> float:
        """Finest step size — used to ground the model with the standard
        (flow/diffusion) loss, mirroring the paper's empirical-velocity target
        at the smallest ``d``."""
        return self.low

    def discrete_levels(self) -> tuple[float, ...] | None:
        """The discrete normalised levels, or ``None`` for continuous modes."""
        return self.levels

    def to_timestep_jump(self, step_size: float) -> int:
        """Convert a normalised step size to an integer timestep jump
        ``round(s · T)``, clamped to at least 1 (a DDIM micro-step cannot be
        zero-length)."""
        return max(1, int(round(float(step_size) * self.timesteps)))

    # ------------------------------------------------------------------ sampling

    def sample(self, *, exclude_smallest: bool = False, generator: torch.Generator | None = None) -> float:
        """Draw one normalised step size for the whole batch (matching the
        paper's per-step batch convention).

        ``exclude_smallest`` drops the finest discrete level so the sampled
        ``s`` always has a representable half ``s/2`` for self-consistency
        chaining (the finest level is handled by the grounding/anchor path).
        Ignored for ``uniform`` (continuous half is always valid).
        """
        if self.mode == "uniform" or self.levels is None:
            u = torch.rand((), generator=generator).item()
            return self.low + u * (self.high - self.low)
        levels = self.levels
        if exclude_smallest and len(levels) > 1:
            levels = levels[1:]
        idx = int(torch.randint(0, len(levels), (1,), generator=generator).item())
        return levels[idx]
