"""RT-1 (fractal20220817_data, Open X-Embodiment) dataset translator.

RT-1 episodes are converted to our mp4 + metadata.pt schema by
``jobs/experiments_cluster/avid_official/convert_rt1_to_mp4meta.py`` (streamed
from ``gs://gresearch/robotics``). That schema is IDENTICAL to ACWM-Phys
(``metadata.pt`` = list[dict] with ``video_path`` / ``actions`` [T,7] / ``length``,
plus ``episode_*.mp4``), so this translator reuses
:class:`ACWMPhysTranslator`'s reading + windowing logic verbatim and only
overrides the caption / frame-rate identity.

Actions are RT-1's canonical **7-DoF end-effector delta**:
``world_vector[3] + rotation_delta[3] + gripper_closedness_action[1]``, already
in ~[-1, 1] in the source (no extra normalization applied at conversion). The
stride-summing in ``load_clip`` (inherited) correctly composes consecutive
deltas into a net window delta.

Why it exists: RT-1 is real-world robot video that is *in-distribution* for the
Wan / SkyReels priors — the control for the ACWM action-blindness finding
(thesis-vault ``30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind``).
"""
from __future__ import annotations

from generative_flow_adapters.data.translators.acwm_phys import ACWMPhysTranslator
from generative_flow_adapters.data.translators.base import EpisodeRef


class RT1Translator(ACWMPhysTranslator):
    """Open X-Embodiment RT-1, read through the ACWM mp4+metadata.pt path.

    Surfaces each episode's ``natural_language_instruction`` as the PER-CLIP
    caption + ``task_name=clip_id`` (stored by the converter), so the per-clip
    text-context table conditions the base on this episode's instruction rather
    than a generic robot-arm prompt. Falls back to the ``caption_template`` for
    older converted data without per-episode captions."""

    def __init__(
        self,
        data_dir: str,
        env_name: str | None = None,
        fs_value: int = 1,
        fps: int = 3,  # RT-1 is ~3 Hz (vs ACWM's 10)
        caption_template: str = "a robot arm manipulating objects on a table, {env_name}",
        letterbox_aspect: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            data_dir,
            env_name=env_name,
            fs_value=fs_value,
            fps=fps,
            caption_template=caption_template,
            letterbox_aspect=letterbox_aspect,
        )

    def load_clip(self, ref: EpisodeRef, start: int, length: int, stride: int = 1) -> dict[str, object]:
        clip = super().load_clip(ref, start, length, stride)
        entry = self._meta[int(clip["episode_idx"])]
        caption = entry.get("caption")
        if caption:  # per-episode instruction present -> use it (else keep template caption)
            clip["caption"] = str(caption)
            clip["task_name"] = str(entry.get("clip_id", clip["task_name"]))
        return clip
