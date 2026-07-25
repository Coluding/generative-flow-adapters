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
``kinematics/robot_arm`` (A=7) is nominally 128 frames, but ``length`` there
is nominal only — a few mp4s decode shorter, so episode length comes from
probing the videos (see :meth:`ACWMPhysTranslator._probe_frame_counts`).

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
        letterbox_aspect: tuple[int, int] | None = None,
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
        # Optional letterbox onto Wan's native aspect. DEFAULT OFF (2026-07-23):
        # square 768x768 was thought to break the base, but that finding was
        # confounded by a base-corruption bug (--random-init aliased the frozen
        # 5B's weights via the adapter). With that fixed, the frozen TI2V-5B
        # generates coherent video at plain square 768x768 — so we train at
        # max_area 589824 like MetaWorld, no letterbox. Set (1280, 704) to
        # re-enable native-aspect padding if a quality benefit is ever shown.
        self.letterbox_aspect = letterbox_aspect
        self.caption = caption_template.format(env_name=self.data_dir.parent.name)
        # metadata.pt is a plain list[dict] of tensors/ints/strs (torch>=2.6
        # defaults weights_only=True which rejects it).
        self._meta: list[dict] = torch.load(meta_path, map_location="cpu", weights_only=False)
        # Convert per-episode action tensors to numpy ONCE: spawn-based
        # DataLoader workers pickle the dataset, and torch shares each tensor
        # through its own file descriptor — 1500 episodes blow the fd ulimit
        # ("unable to open shared memory object ... Too many open files").
        # Numpy arrays pickle by value (~1 MB total for an env) — no fds.
        for entry in self._meta:
            actions = entry.get("actions")
            if isinstance(actions, torch.Tensor):
                entry["actions"] = actions.to(torch.float32).numpy()
        # Lazily opened per process (DataLoader workers fork before first
        # __getitem__, so each worker builds its own readers).
        self._readers: dict[int, object] = {}
        # metadata.pt's `length` is NOMINAL, not the decodable frame count — see
        # _probe_frame_counts. Probed once here (list[int], pickles by value to
        # workers), so the sampler never asks decord for a frame past EOF.
        self._video_frames: list[int] = self._probe_frame_counts()

    def _probe_frame_counts(self) -> list[int]:
        """Decodable frame count per episode, probed from the mp4s themselves.

        The release's ``metadata.pt`` reports a *nominal* per-episode ``length``
        (128 for ``kinematics/robot_arm``) and always ships ``actions`` with that
        many rows, but a few exported videos are genuinely shorter — in
        ``robot_arm/ind_train`` 6 of 2002 episodes decode to 75/79/79/80/96/121
        frames (verified 2026-07-25: local sha256 matches the upstream HF
        blob hash and ``ffprobe -count_frames`` agrees with decord, so the files
        are complete, not truncated downloads). Trusting ``length`` makes the
        window sampler emit starts past the end of the video and decord raises
        ``IndexError: Out of bound indices``.

        Result is cached to a JSON sidecar next to ``metadata.pt`` (best effort —
        a read-only data dir just means re-probing, ~7 s for 2000 episodes).
        """
        import json  # noqa: PLC0415 — only needed on this path
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        cache_path = self.data_dir / "frame_counts.json"
        cached: dict[str, int] = {}
        if cache_path.exists():
            try:
                cached = {str(k): int(v) for k, v in json.loads(cache_path.read_text()).items()}
            except (OSError, ValueError):
                cached = {}

        paths = [str(entry["video_path"]) for entry in self._meta]
        missing = [p for p in dict.fromkeys(paths) if p not in cached]
        if missing:
            from decord import VideoReader  # noqa: PLC0415 — heavy import

            def probe(rel: str) -> tuple[str, int]:
                return rel, len(VideoReader(str(self.data_dir / rel)))

            with ThreadPoolExecutor(max_workers=8) as pool:
                cached.update(dict(pool.map(probe, missing)))
            try:
                cache_path.write_text(json.dumps(cached, sort_keys=True))
            except OSError:
                pass  # read-only dataset dir: just re-probe next time

        return [cached[p] for p in paths]

    def list_episodes(self) -> list[EpisodeRef]:
        refs: list[EpisodeRef] = []
        short: list[tuple[int, int, int]] = []
        for idx, entry in enumerate(self._meta):
            n_frames = int(entry["length"])
            n_actions = int(entry["actions"].shape[0])
            n_video = int(self._video_frames[idx])
            # The usable span is bounded by all three streams: nominal length,
            # action rows, and what the mp4 actually decodes to (the last one is
            # smaller for a handful of episodes — see _probe_frame_counts).
            usable = min(n_frames, n_actions, n_video)
            if n_video < min(n_frames, n_actions):
                short.append((idx, min(n_frames, n_actions), n_video))
            refs.append(EpisodeRef(identifier=(self.env_name, str(idx)), length=usable))
        if short:
            preview = ", ".join(f"ep{i}: {claimed}->{actual}" for i, claimed, actual in short[:5])
            more = f" (+{len(short) - 5} more)" if len(short) > 5 else ""
            print(
                f"[acwm_phys] {self.env_name}: {len(short)}/{len(refs)} episodes have fewer video "
                f"frames than metadata claims; clipped to the decodable length [{preview}{more}]"
            )
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
        if self.letterbox_aspect is not None:
            video = self._letterbox(video)

        actions = torch.as_tensor(entry["actions"][start : start + action_span], dtype=torch.float32)
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

    def _letterbox(self, video: np.ndarray) -> np.ndarray:
        """Pad uint8 [T, H, W, C] frames onto a white canvas of
        ``letterbox_aspect`` (content centered, no scaling here — the
        preprocessor's aspect-preserving resize handles the final size)."""
        t, h, w, c = video.shape
        aw, ah = self.letterbox_aspect
        target_w = max(w, int(round(h * aw / ah)))
        target_h = max(h, int(round(w * ah / aw)))
        if target_w == w and target_h == h:
            return video
        canvas = np.full((t, target_h, target_w, c), 255, dtype=video.dtype)
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        canvas[:, y0 : y0 + h, x0 : x0 + w] = video
        return canvas

    def close(self) -> None:
        self._readers.clear()

    def __getstate__(self) -> dict:
        # decord VideoReaders are neither picklable nor fork-safe; spawn-based
        # DataLoader workers re-open their own lazily via _reader().
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state
