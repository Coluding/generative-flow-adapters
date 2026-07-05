from __future__ import annotations

from bisect import bisect_left

import torch
from torch.utils.data import Dataset

from generative_flow_adapters.data.translators.base import EpisodeRef, Translator

SAMPLING_RANDOM = "random"
SAMPLING_EXHAUSTIVE = "exhaustive"


class TranslatedClipDataset(Dataset):
    """A torch ``Dataset`` that turns a :class:`Translator` into clip samples.

    Two sampling modes (mirrors AVID):

    - ``random`` (default): one example per episode; each ``__getitem__`` picks
      a random start within the episode's valid range. Matches the
      ``latent_diffusion`` / DynamiCrafter pathway in AVID.
    - ``exhaustive``: every valid (episode, start) pair is one example;
      ``__len__`` = sum of valid starts. Matches AVID's ``pixel_diffusion``
      (Procgen) pathway and is fully deterministic.

    Random-start sampling uses ``torch.randint`` so DataLoader's
    per-worker RNG seeding via ``worker_init_fn`` / ``torch.manual_seed``
    propagates naturally.
    """

    def __init__(
        self,
        translator: Translator,
        window_width: int,
        frame_stride: int = 1,
        sampling: str = SAMPLING_RANDOM,
        num_windows: int | None = None,
    ) -> None:
        if window_width <= 0:
            raise ValueError(f"window_width must be positive, got {window_width}")
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {frame_stride}")
        if sampling not in {SAMPLING_RANDOM, SAMPLING_EXHAUSTIVE}:
            raise ValueError(
                f"sampling must be {SAMPLING_RANDOM!r} or {SAMPLING_EXHAUSTIVE!r}, "
                f"got {sampling!r}"
            )
        if num_windows is not None and num_windows <= 0:
            raise ValueError(f"num_windows must be positive or None, got {num_windows}")

        self.translator = translator
        self.window_width = window_width
        self.frame_stride = frame_stride
        self.sampling = sampling
        # In ``random`` mode, restrict each episode to a fixed pool of ``num_windows``
        # deterministic evenly-spaced starts (instead of a fresh random start every
        # access). This makes the sampled windows a finite, stable set — a prerequisite
        # for latent caching, which keys on ``start_idx``. ``None`` = unbounded random
        # (any start; latent cache cannot hit). Ignored in exhaustive mode.
        self.num_windows = num_windows

        # Episodes must hold the action-aggregation window, not just the pixel
        # span: actions are summed over `frame_stride` raw steps per kept frame,
        # which needs `window_width * frame_stride` raw frames (one window past
        # the last kept frame). This is >= the pixel span `(W-1)*stride+1` and
        # equals `window_width` when frame_stride == 1, so it is a no-op for the
        # contiguous case. See MetaWorldTranslator.load_clip / _read_summed_actions.
        span = window_width * frame_stride
        self._span = span
        self._episodes: list[EpisodeRef] = [
            ep for ep in translator.list_episodes() if ep.length >= span
        ]

        if sampling == SAMPLING_EXHAUSTIVE:
            cumulative: list[int] = []
            running = 0
            for ep in self._episodes:
                running += ep.length - span + 1
                cumulative.append(running)
            self._cumulative = cumulative
            self._length = running
        else:
            self._cumulative = None
            self._length = len(self._episodes)

    def __len__(self) -> int:
        return self._length

    @property
    def episodes(self) -> list[EpisodeRef]:
        return list(self._episodes)

    def __getitem__(self, idx: int) -> dict[str, object]:
        if idx < 0:
            idx = self._length + idx
        if idx < 0 or idx >= self._length:
            raise IndexError(f"index {idx} out of range for dataset of length {self._length}")

        if self.sampling == SAMPLING_EXHAUSTIVE:
            ep_idx = bisect_left(self._cumulative, idx + 1)
            prior = self._cumulative[ep_idx - 1] if ep_idx > 0 else 0
            start = idx - prior
            ep = self._episodes[ep_idx]
        else:
            ep = self._episodes[idx]
            if self.num_windows is not None:
                # Random augmentation restricted to the episode's fixed K-window pool
                # (so every sampled clip has a cached latent).
                starts = self._fixed_starts(ep)
                start = starts[int(torch.randint(0, len(starts), ()).item())]
            else:
                max_start = ep.length - self._span
                start = 0 if max_start <= 0 else int(torch.randint(0, max_start + 1, ()).item())

        return self.translator.load_clip(
            ep,
            start=start,
            length=self.window_width,
            stride=self.frame_stride,
        )

    def _fixed_starts(self, ep: EpisodeRef) -> list[int]:
        """Deterministic pool of up to ``num_windows`` evenly-spaced start indices for
        an episode (inclusive of both ends). Fewer than ``num_windows`` when the
        episode admits fewer distinct starts. Stable across runs/processes, so the
        latent cache keyed on ``start_idx`` hits."""
        max_start = ep.length - self._span
        if max_start <= 0:
            return [0]
        k = min(int(self.num_windows), max_start + 1)
        if k == 1:
            return [0]
        return sorted({round(i * max_start / (k - 1)) for i in range(k)})

    def fixed_window_enumeration(self) -> "FixedWindowEnumeration":
        """A deterministic ``Dataset`` over *every* window in the K-window pool
        (one item per (episode, fixed-start) pair) — the exact set the random
        sampler can draw. Iterate it once to precompute latents for all of them."""
        return FixedWindowEnumeration(self)


class FixedWindowEnumeration(Dataset):
    """Flattened, deterministic view of a :class:`TranslatedClipDataset`'s fixed
    K-window pool: ``__len__`` = total (episode, start) pairs, in a stable order.
    Used by the offline latent-precompute pass so it caches exactly the windows
    training will sample (no more, no less)."""

    def __init__(self, base: TranslatedClipDataset) -> None:
        if base.num_windows is None:
            raise ValueError("fixed-window enumeration needs num_windows set on the base dataset.")
        self.base = base
        self._pairs: list[tuple[EpisodeRef, int]] = [
            (ep, start) for ep in base._episodes for start in base._fixed_starts(ep)
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        ep, start = self._pairs[idx]
        return self.base.translator.load_clip(
            ep, start=start, length=self.base.window_width, stride=self.base.frame_stride
        )
