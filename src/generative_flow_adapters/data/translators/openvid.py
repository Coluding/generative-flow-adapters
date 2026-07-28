"""OpenVid-1M dataset translator — real-world CAPTIONED video for the D3
(few-step shortcut) in-distribution test. NO actions (shortcut is action-free);
each clip carries its OWN caption, unlike ACWM's fixed prompt pool.

OpenVid clips are converted to our mp4 + metadata.pt schema by
``scripts/download_openvid.py``. Each ``metadata.pt`` entry is
``{video_path, caption, clip_id, actions[T,1] (dummy zeros), length}`` — schema-
compatible with :class:`ACWMPhysTranslator` (which processes ``actions``), so
this reuses its reading/windowing verbatim and only overrides the emitted
identity: a PER-CLIP ``caption`` and ``task_name = clip_id``.

The ``task_name = clip_id`` is the key the per-clip text-context table
(``scripts/precompute_clip_captions.py`` → ``positive: {clip_id: T5(caption)}``)
is looked up by in :class:`PromptContextProvider` (positive/task mode). So the
frozen base is conditioned on THIS clip's caption during training — the TI2V
setup (frame-0 image anchor + per-clip text), with the shortcut adapter making
it few-step. Diffusion (DynamiCrafter) vs flow (Wan) few-step on the same
captioned real-world data is the flow-vs-diffusion trajectory comparison.

Actions are dummy zeros [T,1] (there are none) — the shortcut configs are
action-free (drop_condition_prob 1.0 / conditions []), so the value is nulled;
the dummy just keeps the ACWM reading path (which touches ``actions``) working.
"""
from __future__ import annotations

from generative_flow_adapters.data.translators.acwm_phys import ACWMPhysTranslator
from generative_flow_adapters.data.translators.base import EpisodeRef


class OpenVidTranslator(ACWMPhysTranslator):
    """OpenVid-1M captioned real-world video, read through the ACWM path.

    Overrides ``load_clip`` to emit the clip's own caption + a ``task_name``
    equal to its ``clip_id`` (the per-clip text-context lookup key)."""

    def __init__(
        self,
        data_dir: str,
        env_name: str | None = None,
        fs_value: int = 1,
        fps: int = 8,  # OpenVid clips are ~8 fps after our subsampling
        caption_template: str = "{env_name}",  # unused (per-clip caption overrides)
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
        caption = str(entry.get("caption", ""))
        clip_id = str(entry.get("clip_id", f"clip_{clip['episode_idx']}"))
        # PER-CLIP identity: caption is this clip's real caption; task_name is the
        # stable key the positive-mode text-context table is indexed by.
        clip["caption"] = caption
        clip["task_name"] = clip_id
        return clip
