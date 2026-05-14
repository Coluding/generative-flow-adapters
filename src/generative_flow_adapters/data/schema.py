"""Canonical clip schema produced by :class:`TranslatedClipDataset`.

Every translator must return a dict with at least the ``CORE_KEYS``. Sensor
keys are optional and translator-specific. Names mirror AVID's RTX loader
(``prepare_rtx_example``) so that downstream code can stay source-agnostic.

Shapes (T = clip length, after stride):

- ``video``         uint8,   (T, H, W, 3)         raw pixels, no normalization
- ``act``           float32, (T, A)               per-frame action
- ``caption``       str                            text condition (may be "")
- ``fps``           int                            placeholder when source has no fps
- ``frame_stride``  int                            stride used to slice the clip
- ``start_idx``     int                            absolute start index in the episode

Optional MetaWorld sensor channels (present when stored by the collector):

- ``proprio``       float32, (T, 7)
- ``depth``         float32, (T, H, W)
- ``tactile``       float32, (T, 2, H_t, W_t)
- ``force_torque``  float32, (T, F)
- ``gripper``       float32, (T,)
- ``ee_xyz``        float32, (T, 3)
- ``object_1_xyz``  float32, (T, 3)
- ``object_2_xyz``  float32, (T, 3)
- ``bool_contact``  bool,    (T,)

Provenance:

- ``env_name``      str
- ``episode_idx``   int
- ``policy_type``   str

No image normalization, latent encoding, or rearrange happens in the dataset.
Downstream code is responsible for any such processing.
"""

from __future__ import annotations

CORE_KEYS: tuple[str, ...] = (
    "video",
    "act",
    "caption",
    "fps",
    "frame_stride",
    "start_idx",
)

PROVENANCE_KEYS: tuple[str, ...] = (
    "env_name",
    "episode_idx",
    "policy_type",
)

METAWORLD_OPTIONAL_KEYS: tuple[str, ...] = (
    "proprio",
    "depth",
    "tactile",
    "force_torque",
    "gripper",
    "ee_xyz",
    "object_1_xyz",
    "object_2_xyz",
    "bool_contact",
)
