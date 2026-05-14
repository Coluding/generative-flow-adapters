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

    def close(self) -> None:
        return None
