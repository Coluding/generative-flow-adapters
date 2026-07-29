from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeRef:
    identifier: tuple[str, ...]
    length: int


class Translator(ABC):
    @abstractmethod
    def list_episodes(self) -> list[EpisodeRef]:
        raise NotImplementedError

    @abstractmethod
    def load_clip(
        self,
        ref: EpisodeRef,
        start: int,
        length: int,
        stride: int = 1,
    ) -> dict[str, object]:
        raise NotImplementedError

    def load_clip_meta(
        self,
        ref: EpisodeRef,
        start: int,
        length: int,
        stride: int = 1,
    ) -> dict[str, object] | None:
        """Everything :meth:`load_clip` returns **except** the decoded frames, or
        ``None`` if this translator can't produce it without decoding.

        Lets a caller that already holds the clip's VAE latent (see
        ``data/latent_prefetch.py``) skip the video decode entirely — for a
        97-frame 480x640 window that is ~89 MB of pixels per sample that would
        otherwise be decoded, collated and shipped only to be discarded on a cache
        hit. Implementations must emit ``source_hw`` (the un-resized ``(H, W)``)
        so the latent-cache key can still be built.
        """
        return None

    def close(self) -> None:
        return None
