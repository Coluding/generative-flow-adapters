from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from generative_flow_adapters.data import (
    EpisodeRef,
    MetaWorldTranslator,
    TranslatedClipDataset,
)

h5py = pytest.importorskip("h5py")


ENVS = ("assembly-v3", "pick-place-v3")
EPISODE_LENGTHS = {
    ("assembly-v3", "episode_0"): 12,
    ("assembly-v3", "episode_1"): 10,
    ("pick-place-v3", "episode_0"): 8,
    ("pick-place-v3", "episode_1"): 14,
}
H = W = 4
ACTION_DIM = 4
PROPRIO_DIM = 7
FT_DIM = 6
TACTILE = (2, 4, 4)


def _write_episode(group, length: int, env_name: str, episode_seed: int, policy_type: str) -> None:
    # encode (episode_seed, frame_idx, channel) into each pixel so we can verify slicing exactly
    pixels = np.zeros((length, H, W, 3), dtype=np.uint8)
    for t in range(length):
        pixels[t, :, :, 0] = episode_seed
        pixels[t, :, :, 1] = t
        pixels[t, :, :, 2] = (episode_seed * 100 + t) % 256
    actions = np.arange(length * ACTION_DIM, dtype=np.float32).reshape(length, ACTION_DIM) + episode_seed * 1000
    proprio = np.arange(length * PROPRIO_DIM, dtype=np.float32).reshape(length, PROPRIO_DIM)
    depth = np.zeros((length, H, W), dtype=np.float32)
    tactile = np.zeros((length,) + TACTILE, dtype=np.float32)
    force_torque = np.zeros((length, FT_DIM), dtype=np.float32)
    gripper = np.zeros(length, dtype=np.float32)
    ee_xyz = np.zeros((length, 3), dtype=np.float32)
    object_1_xyz = np.zeros((length, 3), dtype=np.float32)
    object_2_xyz = np.zeros((length, 3), dtype=np.float32)
    bool_contact = np.zeros(length, dtype=np.bool_)

    group.create_dataset("pixels", data=pixels)
    group.create_dataset("action", data=actions)
    group.create_dataset("proprio", data=proprio)
    group.create_dataset("depth", data=depth)
    group.create_dataset("tactile", data=tactile)
    group.create_dataset("force_torque", data=force_torque)
    group.create_dataset("gripper", data=gripper)
    group.create_dataset("ee_xyz", data=ee_xyz)
    group.create_dataset("object_1_xyz", data=object_1_xyz)
    group.create_dataset("object_2_xyz", data=object_2_xyz)
    group.create_dataset("bool_contact", data=bool_contact)
    group.attrs["policy_type"] = policy_type
    group.attrs["task_name"] = env_name


@pytest.fixture
def fake_metaworld_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "fake.hdf5"
    seed = 0
    with h5py.File(path, "w") as f:
        for env_name in ENVS:
            env_group = f.create_group(env_name)
            env_group.attrs["task_name"] = env_name
            env_group.attrs["randomize_every_reset"] = True
            for ep_key in ("episode_0", "episode_1"):
                length = EPISODE_LENGTHS[(env_name, ep_key)]
                ep_group = env_group.create_group(ep_key)
                _write_episode(
                    ep_group,
                    length=length,
                    env_name=env_name,
                    episode_seed=seed + 1,
                    policy_type="noisy_expert" if ep_key.endswith("0") else "random_walk",
                )
                seed += 1
    return path


def test_translator_indexes_all_episodes(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    episodes = translator.list_episodes()
    assert len(episodes) == 4
    by_id = {ep.identifier: ep.length for ep in episodes}
    assert by_id == EPISODE_LENGTHS


def test_translator_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        MetaWorldTranslator("/does/not/exist.hdf5")


def test_translator_rejects_bad_caption_mode(fake_metaworld_hdf5: Path) -> None:
    with pytest.raises(ValueError):
        MetaWorldTranslator(fake_metaworld_hdf5, caption_mode="wat")


def test_load_clip_schema(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    ref = translator.list_episodes()[0]
    clip = translator.load_clip(ref, start=0, length=4, stride=1)
    assert clip["video"].dtype == torch.uint8
    assert clip["video"].shape == (4, H, W, 3)
    assert clip["act"].dtype == torch.float32
    assert clip["act"].shape == (4, ACTION_DIM)
    assert clip["proprio"].shape == (4, PROPRIO_DIM)
    assert clip["depth"].shape == (4, H, W)
    assert clip["tactile"].shape == (4,) + TACTILE
    assert clip["force_torque"].shape == (4, FT_DIM)
    assert clip["gripper"].shape == (4,)
    assert clip["ee_xyz"].shape == (4, 3)
    assert clip["bool_contact"].dtype == torch.bool
    assert clip["caption"] == ""  # default = empty
    assert clip["fps"] == 5
    assert clip["frame_stride"] == 1
    assert clip["start_idx"] == 0
    assert clip["env_name"] in ENVS
    assert clip["episode_idx"] in {0, 1}
    assert clip["policy_type"] in {"noisy_expert", "random_walk"}


def test_load_clip_slices_window_correctly(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    ref = next(ep for ep in translator.list_episodes() if ep.identifier == ("assembly-v3", "episode_0"))
    clip = translator.load_clip(ref, start=3, length=4, stride=1)
    # The fixture encodes the frame index into channel 1 of every pixel.
    assert clip["video"][:, 0, 0, 1].tolist() == [3, 4, 5, 6]
    # Action values are arange-encoded; row t starts at t * ACTION_DIM + seed_offset.
    first_row = clip["act"][0]
    second_row = clip["act"][1]
    assert torch.allclose(second_row - first_row, torch.full((ACTION_DIM,), float(ACTION_DIM)))


def test_load_clip_with_stride(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    ref = next(ep for ep in translator.list_episodes() if ep.identifier == ("assembly-v3", "episode_0"))
    clip = translator.load_clip(ref, start=0, length=4, stride=2)
    assert clip["video"][:, 0, 0, 1].tolist() == [0, 2, 4, 6]
    assert clip["frame_stride"] == 2


def test_load_clip_rejects_out_of_range(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    ref = next(ep for ep in translator.list_episodes() if ep.identifier == ("pick-place-v3", "episode_0"))
    with pytest.raises(IndexError):
        translator.load_clip(ref, start=5, length=4, stride=1)  # 5 + 4 > 8


def test_caption_template_mode(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5, caption_mode="template")
    ref = translator.list_episodes()[0]
    clip = translator.load_clip(ref, start=0, length=2, stride=1)
    assert clip["caption"] == f"Robot arm performs the task: {clip['env_name']}"


def test_translator_handles_missing_optional_fields(tmp_path: Path) -> None:
    path = tmp_path / "minimal.hdf5"
    with h5py.File(path, "w") as f:
        env_group = f.create_group("assembly-v3")
        env_group.attrs["task_name"] = "assembly-v3"
        ep_group = env_group.create_group("episode_0")
        pixels = np.zeros((6, H, W, 3), dtype=np.uint8)
        ep_group.create_dataset("pixels", data=pixels)
        ep_group.create_dataset(
            "action", data=np.zeros((6, ACTION_DIM), dtype=np.float32)
        )
    translator = MetaWorldTranslator(path)
    clip = translator.load_clip(translator.list_episodes()[0], start=0, length=4, stride=1)
    for opt in ("proprio", "depth", "tactile", "force_torque"):
        assert opt not in clip
    assert "video" in clip
    assert "act" in clip


def test_dataset_exhaustive_length(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    dataset = TranslatedClipDataset(translator, window_width=4, sampling="exhaustive")
    expected = sum(length - 4 + 1 for length in EPISODE_LENGTHS.values())
    assert len(dataset) == expected


def test_dataset_exhaustive_mapping_round_trip(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    dataset = TranslatedClipDataset(translator, window_width=4, sampling="exhaustive")
    # The fixture puts frame_idx in channel 1; first valid (episode, start) is
    # (assembly-v3/episode_0, 0). With sorted index order, idx=0 -> start=0,
    # idx=1 -> start=1, etc.
    sample_0 = dataset[0]
    sample_1 = dataset[1]
    assert sample_0["start_idx"] == 0
    assert sample_1["start_idx"] == 1
    assert sample_0["video"][:, 0, 0, 1].tolist() == [0, 1, 2, 3]
    assert sample_1["video"][:, 0, 0, 1].tolist() == [1, 2, 3, 4]


def test_dataset_random_sampling_gives_one_per_episode(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    dataset = TranslatedClipDataset(translator, window_width=4, sampling="random")
    assert len(dataset) == 4
    torch.manual_seed(0)
    seen_starts: set[tuple[tuple[str, ...], int]] = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        ident = (sample["env_name"], f"episode_{sample['episode_idx']}")
        ep_len = EPISODE_LENGTHS[ident]
        assert 0 <= sample["start_idx"] <= ep_len - 4
        seen_starts.add((ident, int(sample["start_idx"])))
    assert len(seen_starts) == 4  # one per episode


def test_dataset_drops_short_episodes(tmp_path: Path) -> None:
    path = tmp_path / "short.hdf5"
    with h5py.File(path, "w") as f:
        env_group = f.create_group("assembly-v3")
        env_group.attrs["task_name"] = "assembly-v3"
        ep_short = env_group.create_group("episode_0")
        ep_short.create_dataset("pixels", data=np.zeros((3, H, W, 3), dtype=np.uint8))
        ep_short.create_dataset("action", data=np.zeros((3, ACTION_DIM), dtype=np.float32))
        ep_long = env_group.create_group("episode_1")
        ep_long.create_dataset("pixels", data=np.zeros((6, H, W, 3), dtype=np.uint8))
        ep_long.create_dataset("action", data=np.zeros((6, ACTION_DIM), dtype=np.float32))
    translator = MetaWorldTranslator(path)
    dataset = TranslatedClipDataset(translator, window_width=4, sampling="exhaustive")
    assert len(dataset) == 6 - 4 + 1


def test_dataset_works_with_dataloader(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    dataset = TranslatedClipDataset(translator, window_width=4, sampling="exhaustive")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["video"].shape == (2, 4, H, W, 3)
    assert batch["video"].dtype == torch.uint8
    assert batch["act"].shape == (2, 4, ACTION_DIM)
    assert isinstance(batch["caption"], list) and len(batch["caption"]) == 2
    assert isinstance(batch["env_name"], list) and len(batch["env_name"]) == 2


def test_dataset_rejects_bad_args(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    with pytest.raises(ValueError):
        TranslatedClipDataset(translator, window_width=0)
    with pytest.raises(ValueError):
        TranslatedClipDataset(translator, window_width=4, frame_stride=0)
    with pytest.raises(ValueError):
        TranslatedClipDataset(translator, window_width=4, sampling="nope")


def test_dataset_index_bounds(fake_metaworld_hdf5: Path) -> None:
    translator = MetaWorldTranslator(fake_metaworld_hdf5)
    dataset = TranslatedClipDataset(translator, window_width=4, sampling="exhaustive")
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_episode_ref_dataclass() -> None:
    ref = EpisodeRef(identifier=("env", "episode_0"), length=10)
    assert ref.identifier == ("env", "episode_0")
    assert ref.length == 10
