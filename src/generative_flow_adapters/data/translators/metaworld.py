from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from generative_flow_adapters.data.schema import METAWORLD_OPTIONAL_KEYS
from generative_flow_adapters.data.translators.base import EpisodeRef, Translator

CAPTION_MODE_EMPTY = "empty"
CAPTION_MODE_TEMPLATE = "template"
DEFAULT_CAPTION_TEMPLATE = "Robot arm performs the task: {task_name}"
DEFAULT_METAWORLD_FPS = 5


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for MetaWorldTranslator. "
            "Install with `pip install h5py` or `pip install -e .[dev]`."
        ) from exc
    return h5py


class MetaWorldTranslator(Translator):
    """Reads HDF5 files produced by ``collect_metaworld_avid.py``.

    Layout it expects (one file)::

        file.h5/<env_name>/episode_<i>/
          pixels         (T, H, W, 3) uint8
          action         (T, A)       float32
          proprio, depth, tactile, force_torque, gripper,
          ee_xyz, object_1_xyz, object_2_xyz, bool_contact
          attrs: policy_type, task_name, ...

    The file handle is opened lazily, so each DataLoader worker holds its own
    handle after fork. The episode index is built once at construction time.
    """

    def __init__(
        self,
        path: str | Path,
        caption_mode: str = CAPTION_MODE_EMPTY,
        caption_template: str = DEFAULT_CAPTION_TEMPLATE,
        fps: int = DEFAULT_METAWORLD_FPS,
    ) -> None:
        if caption_mode not in {CAPTION_MODE_EMPTY, CAPTION_MODE_TEMPLATE}:
            raise ValueError(
                f"caption_mode must be 'empty' or 'template', got {caption_mode!r}"
            )
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"MetaWorld HDF5 not found: {self.path}")
        self.caption_mode = caption_mode
        self.caption_template = caption_template
        self.fps = fps

        self._file = None  # opened lazily, per process
        self._episode_meta: dict[tuple[str, ...], dict[str, Any]] = {}
        self._episodes: list[EpisodeRef] = self._build_index()

    def _build_index(self) -> list[EpisodeRef]:
        h5py = _require_h5py()
        episodes: list[EpisodeRef] = []
        with h5py.File(self.path, "r") as f:
            for env_name in sorted(f.keys()):
                env_group = f[env_name]
                task_name = str(env_group.attrs.get("task_name", env_name))
                for ep_key in sorted(env_group.keys()):
                    ep_group = env_group[ep_key]
                    if "pixels" not in ep_group:
                        continue
                    length = int(ep_group["pixels"].shape[0])
                    identifier = (env_name, ep_key)
                    self._episode_meta[identifier] = {
                        "task_name": task_name,
                        "episode_idx": _parse_episode_idx(ep_key),
                        "policy_type": str(ep_group.attrs.get("policy_type", "unknown")),
                    }
                    episodes.append(EpisodeRef(identifier=identifier, length=length))
        return episodes

    def list_episodes(self) -> list[EpisodeRef]:
        return list(self._episodes)

    def _open(self):
        if self._file is None:
            h5py = _require_h5py()
            self._file = h5py.File(self.path, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def load_clip(
        self,
        ref: EpisodeRef,
        start: int,
        length: int,
        stride: int = 1,
    ) -> dict[str, object]:
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        span = (length - 1) * stride + 1
        if start + span > ref.length:
            raise IndexError(
                f"Clip exceeds episode: start={start}, length={length}, stride={stride}, "
                f"episode_length={ref.length}"
            )

        env_name, ep_key = ref.identifier
        meta = self._episode_meta[ref.identifier]
        ep_group = self._open()[env_name][ep_key]
        sl = slice(start, start + span, stride)

        clip: dict[str, object] = {
            "video": _read(ep_group, "pixels", sl),
            "act": _read(ep_group, "action", sl),
            "caption": self._caption_for(meta["task_name"]),
            "fps": int(self.fps),
            "frame_stride": int(stride),
            "start_idx": int(start),
            "env_name": env_name,
            "episode_idx": int(meta["episode_idx"]),
            "policy_type": meta["policy_type"],
        }
        for key in METAWORLD_OPTIONAL_KEYS:
            if key in ep_group:
                clip[key] = _read(ep_group, key, sl)
        return clip

    def _caption_for(self, task_name: str) -> str:
        if self.caption_mode == CAPTION_MODE_EMPTY:
            return ""
        return self.caption_template.format(task_name=task_name)


def _parse_episode_idx(ep_key: str) -> int:
    try:
        return int(ep_key.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return -1


def _read(ep_group, name: str, sl: slice) -> torch.Tensor:
    arr = np.asarray(ep_group[name][sl])
    return torch.from_numpy(arr.copy())
