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
    """Reads HDF5 files of MetaWorld rollouts. Two layouts are auto-detected.

    **Flat layout** (legacy ``collect_metaworld_avid.py``)::

        file.h5/<env_name>/episode_<i>/
          pixels, action, proprio, depth, tactile, ...    # everything together
          attrs: policy_type, task_name, ...

    **Camera layout** (current) — images split per camera, sensors stored once::

        file.h5/<env_name>/                 # attrs: task_name, camera_names,
          │                                 #        frame_stride, frame_aggregation
          ├── <camera>/episode_<i>/         # one group per camera (e.g. corner2)
          │     pixels (T,H,W,3) uint8, depth (T,H,W) float32
          └── sensors/episode_<i>/          # camera-independent, stored ONCE
                action, proprio, tactile, force_torque, gripper,
                ee_xyz, object_*_xyz, bool_contact
                attrs: policy_type, task_name, ...

    A view is paired with its state by **matching episode index**:
    ``<env>/corner2/episode_5`` ↔ ``<env>/sensors/episode_5``. ``T`` is the same
    across all cameras and the sensors branch (the collector already applied its
    own ``frame_stride``/``frame_aggregation`` — see the note in ``load_clip``).

    ``envs`` / ``cameras`` select which environments / camera angles to train on
    (str, list of str, or ``None`` = all). Each (env, camera, episode) becomes
    its own clip source, so multiple cameras multiply the sample count (same
    rollout seen from several views). ``cameras`` is ignored for the flat layout.

    The file handle is opened lazily, so each DataLoader worker holds its own
    handle after fork. The episode index is built once at construction time.
    """

    def __init__(
        self,
        path: str | Path,
        caption_mode: str = CAPTION_MODE_EMPTY,
        caption_template: str = DEFAULT_CAPTION_TEMPLATE,
        fps: int = DEFAULT_METAWORLD_FPS,
        fs_value: int = 1,
        envs: str | list[str] | None = None,
        cameras: str | list[str] | None = None,
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
        # None -> keep everything; otherwise a frozenset of names to keep.
        self._env_filter = _as_filter(envs)
        self._camera_filter = _as_filter(cameras)
        # The constant value emitted as ``fs`` for the frozen base's fps channel,
        # DECOUPLED from the actual slice stride. We subsample frames for a longer
        # temporal window (see ``load_clip``) but deliberately hold ``fs`` fixed
        # (default 1) rather than scaling it with the stride: MetaWorld's large
        # per-frame disparity makes the base's fps prior a poor fit, so we let the
        # adapter absorb the motion and keep ``fs`` a constant, non-conditioning
        # input. See thesis-vault decision metaworld-frame-stride-load-time.
        self.fs_value = int(fs_value)

        self._file = None  # opened lazily, per process
        self._episode_meta: dict[tuple[str, ...], dict[str, Any]] = {}
        self._episodes: list[EpisodeRef] = self._build_index()

    def _build_index(self) -> list[EpisodeRef]:
        h5py = _require_h5py()
        episodes: list[EpisodeRef] = []
        with h5py.File(self.path, "r") as f:
            for env_name in sorted(f.keys()):
                if self._env_filter is not None and env_name not in self._env_filter:
                    continue
                env_group = f[env_name]
                if _is_camera_layout(env_group):
                    episodes.extend(self._index_camera_env(env_name, env_group))
                else:
                    episodes.extend(self._index_flat_env(env_name, env_group))
        return episodes

    def _index_flat_env(self, env_name: str, env_group) -> list[EpisodeRef]:
        task_name = str(env_group.attrs.get("task_name", env_name))
        episodes: list[EpisodeRef] = []
        for ep_key in sorted(env_group.keys()):
            ep_group = env_group[ep_key]
            if "pixels" not in ep_group:
                continue
            identifier = (env_name, ep_key)
            self._episode_meta[identifier] = {
                "task_name": task_name,
                "episode_idx": _parse_episode_idx(ep_key),
                "policy_type": str(ep_group.attrs.get("policy_type", "unknown")),
            }
            episodes.append(EpisodeRef(identifier=identifier, length=int(ep_group["pixels"].shape[0])))
        return episodes

    def _index_camera_env(self, env_name: str, env_group) -> list[EpisodeRef]:
        task_name = str(env_group.attrs.get("task_name", env_name))
        frame_stride_attr = int(env_group.attrs.get("frame_stride", 1))
        sensors = env_group["sensors"] if "sensors" in env_group else None
        episodes: list[EpisodeRef] = []
        for camera in self._resolve_cameras(env_group):
            cam_group = env_group[camera]
            for ep_key in sorted(cam_group.keys()):
                if "pixels" not in cam_group[ep_key]:
                    continue
                # Per-episode policy metadata lives on the sensors branch.
                ep_attrs = sensors[ep_key].attrs if (sensors is not None and ep_key in sensors) else {}
                identifier = (env_name, camera, ep_key)
                self._episode_meta[identifier] = {
                    "task_name": str(ep_attrs.get("task_name", task_name)),
                    "episode_idx": _parse_episode_idx(ep_key),
                    "policy_type": str(ep_attrs.get("policy_type", "unknown")),
                    "frame_stride_attr": frame_stride_attr,
                }
                episodes.append(
                    EpisodeRef(identifier=identifier, length=int(cam_group[ep_key]["pixels"].shape[0]))
                )
        return episodes

    def _resolve_cameras(self, env_group) -> list[str]:
        names = env_group.attrs.get("camera_names")
        if names is not None:
            cameras = _decode_names(names)
        else:
            cameras = [k for k in env_group.keys() if k != "sensors"]
        if self._camera_filter is not None:
            cameras = [c for c in cameras if c in self._camera_filter]
        return sorted(cameras)

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
        # Pixels/state keys are sampled at the `length` kept frames (every
        # `stride`-th). Actions are SUMMED over each stride-window, which needs
        # one extra window past the last kept frame -> `length * stride` raw
        # frames. The action span is the binding one (>= the pixel span).
        # NOTE: the camera layout already applied a frame_stride at collection
        # time (env attr), so a config there usually wants stride=1; the SUM
        # still composes correctly on pre-aggregated deltas if stride>1.
        pixel_span = (length - 1) * stride + 1
        action_span = length * stride
        span = max(pixel_span, action_span)  # == action_span for stride >= 1
        if start + span > ref.length:
            raise IndexError(
                f"Clip exceeds episode: start={start}, length={length}, stride={stride}, "
                f"span={span}, episode_length={ref.length}"
            )
        sl = slice(start, start + pixel_span, stride)

        if len(ref.identifier) == 3:
            return self._load_clip_camera(ref, start=start, length=length, stride=stride, sl=sl)
        return self._load_clip_flat(ref, start=start, length=length, stride=stride, sl=sl)

    def _load_clip_flat(self, ref, *, start, length, stride, sl) -> dict[str, object]:
        env_name, ep_key = ref.identifier
        meta = self._episode_meta[ref.identifier]
        ep_group = self._open()[env_name][ep_key]
        clip = self._base_clip(meta, start=start, stride=stride, env_name=env_name)
        clip["video"] = _read(ep_group, "pixels", sl)
        clip["act"] = _read_summed_actions(ep_group, start=start, length=length, stride=stride)
        for key in METAWORLD_OPTIONAL_KEYS:
            if key in ep_group:
                # State keys (proprio, ee_xyz, ...) are SAMPLED at the kept frames,
                # never summed — summing absolute positions would be meaningless.
                clip[key] = _read(ep_group, key, sl)
        return clip

    def _load_clip_camera(self, ref, *, start, length, stride, sl) -> dict[str, object]:
        env_name, camera, ep_key = ref.identifier
        meta = self._episode_meta[ref.identifier]
        env_group = self._open()[env_name]
        cam_group = env_group[camera][ep_key]
        sensors_group = env_group["sensors"][ep_key]

        clip = self._base_clip(meta, start=start, stride=stride, env_name=env_name)
        clip["camera"] = camera
        clip["video"] = _read(cam_group, "pixels", sl)
        clip["act"] = _read_summed_actions(sensors_group, start=start, length=length, stride=stride)
        # Depth lives alongside pixels in the camera group; the rest of the
        # state channels are camera-independent under sensors/.
        if "depth" in cam_group:
            clip["depth"] = _read(cam_group, "depth", sl)
        for key in METAWORLD_OPTIONAL_KEYS:
            if key == "depth":
                continue
            if key in sensors_group:
                clip[key] = _read(sensors_group, key, sl)
        return clip

    def _base_clip(self, meta, *, start: int, stride: int, env_name: str) -> dict[str, object]:
        return {
            "caption": self._caption_for(meta["task_name"]),
            "fps": int(self.fps),
            # `fs` is the value fed to the frozen base's fps channel — a fixed
            # constant decoupled from the real slice stride. `frame_stride`
            # records the actual subsample stride for provenance/debugging.
            "fs": int(self.fs_value),
            "frame_stride": int(stride),
            "start_idx": int(start),
            "env_name": env_name,
            "episode_idx": int(meta["episode_idx"]),
            "policy_type": meta["policy_type"],
        }

    def _caption_for(self, task_name: str) -> str:
        if self.caption_mode == CAPTION_MODE_EMPTY:
            return ""
        return self.caption_template.format(task_name=task_name)


def _as_filter(value: str | list[str] | None) -> frozenset[str] | None:
    """Normalize an env/camera selector to a frozenset (or None = keep all)."""
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(str(v) for v in value)


def _decode_names(value) -> list[str]:
    """Decode an HDF5 ``camera_names`` attr (bytes / numpy array / list) to str."""
    out: list[str] = []
    for item in np.atleast_1d(value):
        out.append(item.decode() if isinstance(item, bytes) else str(item))
    return out


def _is_camera_layout(env_group) -> bool:
    """Camera layout = images split per camera with sensors stored once. Marked
    by a ``sensors`` subgroup or a ``camera_names`` env attr; otherwise the flat
    legacy layout (episode groups directly under the env)."""
    return "sensors" in env_group or "camera_names" in env_group.attrs


def _parse_episode_idx(ep_key: str) -> int:
    try:
        return int(ep_key.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return -1


def _read(ep_group, name: str, sl: slice) -> torch.Tensor:
    arr = np.asarray(ep_group[name][sl])
    return torch.from_numpy(arr.copy())


def _read_summed_actions(ep_group, *, start: int, length: int, stride: int) -> torch.Tensor:
    """Net delta-action per kept frame.

    MetaWorld actions are per-step deltas ``(Δx, Δy, Δz, Δgripper)``. When we
    subsample frames by ``stride``, the transition between two kept frames is
    driven by *all* ``stride`` intermediate actions, so the action attached to a
    kept frame is their SUM — the net control applied over that window. This
    reduces to the raw per-frame action when ``stride == 1``.

    Reads ``length * stride`` contiguous raw actions from ``start`` and sums each
    consecutive ``stride``-block, returning ``(length, action_dim)``.
    """
    n = length * stride
    arr = np.asarray(ep_group["action"][start:start + n])          # (n, A)
    action_dim = arr.shape[-1]
    arr = arr.reshape(length, stride, action_dim).sum(axis=1)      # (length, A)
    return torch.from_numpy(arr.copy())
