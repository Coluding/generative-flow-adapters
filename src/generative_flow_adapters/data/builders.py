"""Config-driven construction of the MetaWorld clip dataset.

Keeps the translator/dataset wiring in one place so the training scripts read
dataset shape — frame stride especially — from the YAML ``data:`` block rather
than from CLI flags. CLI overrides are still honoured when explicitly passed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from generative_flow_adapters.data.dataset import TranslatedClipDataset
from generative_flow_adapters.data.translators.acwm_phys import ACWMPhysTranslator
from generative_flow_adapters.data.translators.rt1 import RT1Translator
from generative_flow_adapters.data.translators.openvid import OpenVidTranslator
from generative_flow_adapters.data.translators.metaworld import MetaWorldTranslator

if TYPE_CHECKING:
    from generative_flow_adapters.config import DataConfig


def build_metaworld_clip_dataset(
    data: "DataConfig",
    *,
    default_window_width: int,
    hdf5: str | None = None,
    frame_stride: int | None = None,
    sampling: str | None = None,
    num_windows: int | None = None,
) -> tuple[MetaWorldTranslator, TranslatedClipDataset]:
    """Build ``(translator, dataset)`` from a :class:`DataConfig`.

    The config supplies the values; the keyword overrides (e.g. from CLI) win
    only when not ``None``. ``default_window_width`` is the fallback used when
    ``data.window_width`` is unset (typically ``model.extra.temporal_length``).

    The translator feeds a constant ``fs = data.fs_value`` to the frozen base,
    decoupled from the real subsample stride (``data.frame_stride`` / the
    ``frame_stride`` override) — see the load-time-frame-stride decision.
    """
    resolved_hdf5 = hdf5 or data.hdf5
    if not resolved_hdf5:
        raise ValueError(
            "MetaWorld dataset needs an HDF5 path: set `data.hdf5` in the config or pass --hdf5."
        )
    resolved_stride = data.frame_stride if frame_stride is None else int(frame_stride)
    resolved_sampling = sampling or data.sampling
    resolved_window = int(data.window_width) if data.window_width else int(default_window_width)
    resolved_num_windows = num_windows if num_windows is not None else getattr(data, "num_windows", None)

    translator = MetaWorldTranslator(
        resolved_hdf5,
        caption_mode=data.caption_mode,
        fs_value=data.fs_value,
        envs=data.env,
        cameras=data.camera,
    )
    dataset = TranslatedClipDataset(
        translator,
        window_width=resolved_window,
        frame_stride=resolved_stride,
        sampling=resolved_sampling,
        num_windows=resolved_num_windows,
    )
    return translator, dataset


def build_acwmphys_clip_dataset(
    data: "DataConfig",
    *,
    default_window_width: int,
    data_dir: str,
    env_name: str | None = None,
    frame_stride: int | None = None,
    sampling: str | None = None,
    num_windows: int | None = None,
    letterbox_aspect: tuple[int, int] | None = None,
) -> tuple[ACWMPhysTranslator, TranslatedClipDataset]:
    """ACWM-Phys twin of :func:`build_metaworld_clip_dataset`. ``data_dir`` is
    one split directory of the HF release (contains ``metadata.pt`` +
    ``episode_*.mp4``); everything else mirrors the MetaWorld builder so the
    training/precompute scripts can switch datasets with one flag.

    ``letterbox_aspect`` pads frames to Wan's native aspect (default 1280:704);
    None disables it (raw square frames — see the aspect-vs-random-init test)."""
    if not data_dir:
        raise ValueError("ACWM-Phys dataset needs --data-dir (a split directory of the HF release).")
    resolved_stride = data.frame_stride if frame_stride is None else int(frame_stride)
    resolved_sampling = sampling or data.sampling
    resolved_window = int(data.window_width) if data.window_width else int(default_window_width)
    resolved_num_windows = num_windows if num_windows is not None else getattr(data, "num_windows", None)

    translator = ACWMPhysTranslator(data_dir, env_name=env_name, fs_value=data.fs_value,
                                    letterbox_aspect=letterbox_aspect)
    dataset = TranslatedClipDataset(
        translator,
        window_width=resolved_window,
        frame_stride=resolved_stride,
        sampling=resolved_sampling,
        num_windows=resolved_num_windows,
    )
    return translator, dataset


def build_rt1_clip_dataset(
    data: "DataConfig",
    *,
    default_window_width: int,
    data_dir: str,
    env_name: str | None = None,
    frame_stride: int | None = None,
    sampling: str | None = None,
    num_windows: int | None = None,
    letterbox_aspect: tuple[int, int] | None = None,
) -> tuple[RT1Translator, TranslatedClipDataset]:
    """RT-1 (Open X-Embodiment) twin of :func:`build_acwmphys_clip_dataset`.

    ``data_dir`` is a directory produced by
    ``jobs/experiments_cluster/avid_official/convert_rt1_to_mp4meta.py``
    (``metadata.pt`` + ``episode_*.mp4``, ACWM-identical schema). Real-world
    robot video, in-distribution for the Wan/SkyReels priors — the control for
    the ACWM action-blindness. Everything mirrors the ACWM builder; only the
    translator (and thus caption/fps identity) differs."""
    if not data_dir:
        raise ValueError("RT-1 dataset needs --data-dir (a converted RT-1 split directory).")
    resolved_stride = data.frame_stride if frame_stride is None else int(frame_stride)
    resolved_sampling = sampling or data.sampling
    resolved_window = int(data.window_width) if data.window_width else int(default_window_width)
    resolved_num_windows = num_windows if num_windows is not None else getattr(data, "num_windows", None)

    translator = RT1Translator(data_dir, env_name=env_name, fs_value=data.fs_value,
                               letterbox_aspect=letterbox_aspect)
    dataset = TranslatedClipDataset(
        translator,
        window_width=resolved_window,
        frame_stride=resolved_stride,
        sampling=resolved_sampling,
        num_windows=resolved_num_windows,
    )
    return translator, dataset


def build_openvid_clip_dataset(
    data: "DataConfig",
    *,
    default_window_width: int,
    data_dir: str,
    env_name: str | None = None,
    frame_stride: int | None = None,
    sampling: str | None = None,
    num_windows: int | None = None,
    letterbox_aspect: tuple[int, int] | None = None,
) -> tuple[OpenVidTranslator, TranslatedClipDataset]:
    """OpenVid-1M captioned real-world video (D3 few-step shortcut in-distribution
    test). ``data_dir`` is produced by ``scripts/download_openvid.py`` (mp4 +
    metadata.pt with a PER-CLIP ``caption`` + ``clip_id``, no real actions).
    Mirrors the RT-1/ACWM builders; the translator emits per-clip caption +
    task_name=clip_id so the per-clip text-context table conditions the base on
    each clip's own caption (TI2V)."""
    if not data_dir:
        raise ValueError("OpenVid dataset needs --data-dir (a converted OpenVid split directory).")
    resolved_stride = data.frame_stride if frame_stride is None else int(frame_stride)
    resolved_sampling = sampling or data.sampling
    resolved_window = int(data.window_width) if data.window_width else int(default_window_width)
    resolved_num_windows = num_windows if num_windows is not None else getattr(data, "num_windows", None)

    translator = OpenVidTranslator(data_dir, env_name=env_name, fs_value=data.fs_value,
                                   letterbox_aspect=letterbox_aspect)
    dataset = TranslatedClipDataset(
        translator,
        window_width=resolved_window,
        frame_stride=resolved_stride,
        sampling=resolved_sampling,
        num_windows=resolved_num_windows,
    )
    return translator, dataset
