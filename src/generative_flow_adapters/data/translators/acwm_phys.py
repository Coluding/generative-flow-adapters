"""Translator for the ACWM-Phys benchmark (arXiv:2605.08567, HF ``t1an/ACWM-Phys``).

On-disk layout (one directory per environment split, e.g.
``<root>/rigid_dynamics/push_block/ind_train/``):

    episode_{i}.mp4     RGB video (released at 1024x1024, 10 fps)
    metadata.pt         list[dict] with per-episode
                        ``video_path`` (str), ``actions`` (FloatTensor [T, A],
                        normalized to [-1, 1]), ``length`` (int, == T),
                        and optionally ``seed``.

Verified against the release 2026-07-22: ``push_block`` (the paper's Push
Cube, A=2 pusher-target actions) and ``pushcube_2`` (two-pusher ablation,
A=4) both have fixed 66-frame episodes with action rows == frame count.

The translator emits the same clip dict as :class:`MetaWorldTranslator`
(``video`` uint8 [T, H, W, C], ``act`` float32 [T, A] summed per stride
window, plus the identity fields ``env_name``/``episode_idx``/``start_idx``/
``frame_stride`` the latent cache keys on), so the whole downstream stack —
``TranslatedClipDataset``, the Wan2.2 diffusion-forcing preprocessor, the
latent cache, eval — works unchanged.

Selected precisely because its actions are *informative*: the pusher's
commanded target decides the future, so I(action; future | anchor frame) is
high by construction — the property MetaWorld scripted demos lack (see
thesis-vault 50_Decisions/open/second-dataset-action-informativeness.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from generative_flow_adapters.data.translators.base import EpisodeRef, Translator


class ACWMPhysTranslator(Translator):
    def __init__(
        self,
        data_dir: str,
        env_name: str | None = None,
        fs_value: int = 1,
        fps: int = 10,
        caption_template: str = "a robot pushing objects on a table, {env_name}",
    ) -> None:
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "metadata.pt"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.pt not found in {self.data_dir} — point data_dir at one split "
                "directory of the HF release (e.g. .../rigid_dynamics/push_block/ind_train)."
            )
        # Default env identity: "<env>-<split>" from the path, e.g.
        # "push_block-ind_train". Part of every latent-cache key, so keep it
        # stable once a cache exists.
        self.env_name = env_name or f"{self.data_dir.parent.name}-{self.data_dir.name}"
        self.fs_value = int(fs_value)
        self.fps = int(fps)
        self.caption = caption_template.format(env_name=self.data_dir.parent.name)
        # metadata.pt is a plain list[dict] of tensors/ints/strs (torch>=2.6
        # defaults weights_only=True which rejects it).
        self._meta: list[dict] = torch.load(meta_path, map_location="cpu", weights_only=False)
        # Lazily opened per process (DataLoader workers fork before first
        # __getitem__, so each worker builds its own readers).
        self._readers: dict[int, object] = {}

    def list_episodes(self) -> list[EpisodeRef]:
        refs: list[EpisodeRef] = []
        for idx, entry in enumerate(self._meta):
            n_frames = int(entry["length"])
            n_actions = int(entry["actions"].shape[0])
            # The usable span is bounded by both streams (release has them
            # equal; min() keeps us safe if a future env pads differently).
            refs.append(EpisodeRef(identifier=(self.env_name, str(idx)), length=min(n_frames, n_actions)))
        return refs

    def _reader(self, episode_idx: int):
        reader = self._readers.get(episode_idx)
        if reader is None:
            from decord import VideoReader  # noqa: PLC0415 — heavy import, worker-local

            path = self.data_dir / str(self._meta[episode_idx]["video_path"])
            reader = VideoReader(str(path))
            # Cap the per-worker open-file set; random sampling revisits
            # episodes rarely enough that re-opening is cheap.
            if len(self._readers) >= 32:
                self._readers.clear()
            self._readers[episode_idx] = reader
        return reader

    def load_clip(self, ref: EpisodeRef, start: int, length: int, stride: int = 1) -> dict[str, object]:
        if length <= 0 or stride <= 0 or start < 0:
            raise ValueError(f"invalid clip request: start={start}, length={length}, stride={stride}")
        episode_idx = int(ref.identifier[1])
        entry = self._meta[episode_idx]
        pixel_span = (length - 1) * stride + 1
        action_span = length * stride  # matches MetaWorld: sum needs one window past the last kept frame
        if start + max(pixel_span, action_span) > ref.length:
            raise IndexError(
                f"Clip exceeds episode: start={start}, length={length}, stride={stride}, "
                f"episode_length={ref.length}"
            )

        frame_indices = list(range(start, start + pixel_span, stride))
        video = self._reader(episode_idx).get_batch(frame_indices).asnumpy()  # uint8 [T, H, W, C]

        actions = entry["actions"][start : start + action_span].to(torch.float32)
        if stride > 1:
            actions = actions.reshape(length, stride, -1).sum(dim=1)

        return {
            "video": np.ascontiguousarray(video),
            "act": actions,
            "caption": self.caption,
            "task_name": self.data_dir.parent.name,
            "fps": self.fps,
            "fs": self.fs_value,
            "frame_stride": int(stride),
            "start_idx": int(start),
            "env_name": self.env_name,
            "episode_idx": episode_idx,
        }

    def close(self) -> None:
        self._readers.clear()
