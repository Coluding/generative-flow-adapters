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

        self.translator = translator
        self.window_width = window_width
        self.frame_stride = frame_stride
        self.sampling = sampling

        span = (window_width - 1) * frame_stride + 1
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
            max_start = ep.length - self._span
            if max_start <= 0:
                start = 0
            else:
                start = int(torch.randint(0, max_start + 1, ()).item())

        return self.translator.load_clip(
            ep,
            start=start,
            length=self.window_width,
            stride=self.frame_stride,
        )
