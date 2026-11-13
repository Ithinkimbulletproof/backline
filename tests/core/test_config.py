# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

import pytest
import yaml

from backline.config import (
    BacklineConfig,
    ConfigError,
    PathSettings,
    deep_merge,
    load_config,
)
from backline.music import DYNAMICS_ORDER, Dynamic
from backline.score import NoteFlag

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _write_profile(tmp_path: Path, override: dict[str, object]) -> Path:
    data = {"extends": str(CONFIGS / "base.yaml"), **override}
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_base_profile_loads() -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.name == "base"
    assert cfg.model.d_model % cfg.model.n_heads == 0


def test_tiny_extends_base(tiny_config: BacklineConfig) -> None:
    base = load_config(CONFIGS / "base.yaml")
    assert tiny_config.name == "tiny"
    assert tiny_config.model.d_model < base.model.d_model
    assert tiny_config.synth.n_songs < base.synth.n_songs
    assert tiny_config.instruments == base.instruments
    assert tiny_config.drums == base.drums


def test_pitch_names_and_bass_profiles(tiny_config: BacklineConfig) -> None:
    bass = tiny_config.instruments.bass
    assert bass.profiles["bass4_standard"].tuning == (28, 33, 38, 43)
    assert bass.profiles["bass5_standard"].lowest == 23
    assert bass.active.highest == 43 + 24
    assert bass.train_range == (23, 67)
    assert tiny_config.instruments.guitar.tuning[0] == 40


def test_output_map_path_is_resolved(tiny_config: BacklineConfig) -> None:
    path = tiny_config.drums.output_map
    assert path.is_absolute()
    assert path.is_file()
    assert tiny_config.drums.load_output_map().name == "gm"


def test_deep_merge_merges_mappings_and_replaces_the_rest() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    merged = deep_merge(base, {"a": {"y": [3]}, "c": 2})
    assert merged == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}


def test_paths_are_relative_to_the_file_that_sets_them(tmp_path: Path) -> None:
    inherited = load_config(_write_profile(tmp_path, {"name": "elsewhere"}))
    assert inherited.drums.output_map == (CONFIGS / "drum_maps" / "gm.yaml").resolve()
    (tmp_path / "kit.yaml").write_text(
        (CONFIGS / "drum_maps" / "gm.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    own = load_config(_write_profile(tmp_path, {"drums": {"output_map": "kit.yaml"}}))
    assert own.drums.output_map == (tmp_path / "kit.yaml").resolve()


def test_user_profile_can_switch_bass_profile(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path, {"name": "mine", "instruments": {"bass": {"profile": "bass5_standard"}}}
    )
    cfg = load_config(path)
    assert cfg.instruments.bass.active.lowest == 23
    assert cfg.instruments.guitar.frets == 24


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"drums": {"input_map": {39: "clap"}}}, "unknown classes"),
        ({"drums": {"coarse": {"cymbals": ["crash1"]}}}, "exactly once"),
        ({"instruments": {"bass": {"profile": "bass7"}}}, "unknown bass profile"),
        (
            {"instruments": {"bass": {"profiles": {"bad": {"tuning": ["G2", "E1"]}}}}},
            "lowest to the highest",
        ),
        ({"velocity": {"levels": {"accent": {"min": 40, "out": 45}}}}, "strictly increase"),
        ({"model": {"d_model": 30, "n_heads": 4}}, "divisible"),
        ({"unknown_section": 1}, "Extra inputs"),
    ],
)
def test_invalid_profiles_are_rejected(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    path = _write_profile(tmp_path, override)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_extends_cycle_is_detected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="extends itself"):
        load_config(tmp_path / "a.yaml")


def test_missing_profile_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml")


def test_velocity_levels(tiny_config: BacklineConfig) -> None:
    v = tiny_config.velocity
    assert v.level_of(30) is Dynamic.GHOST
    assert v.level_of(90) is Dynamic.NORMAL
    assert v.level_of(100) is Dynamic.ACCENT
    assert v.level_of(127) is Dynamic.MAX
    for dynamic in DYNAMICS_ORDER:
        assert v.level_of(v.velocity_of(dynamic)) is dynamic


def test_flags_override_velocity(tiny_config: BacklineConfig) -> None:
    v = tiny_config.velocity
    assert v.level_of(120, NoteFlag.GHOST) is Dynamic.GHOST
    assert v.level_of(60, NoteFlag.HEAVY_ACCENT) is Dynamic.MAX
    assert v.level_of(60, NoteFlag.ACCENT) is Dynamic.ACCENT
    assert v.level_of(127, NoteFlag.ACCENT) is Dynamic.MAX


def test_drum_vocabulary(tiny_config: BacklineConfig) -> None:
    drums = tiny_config.drums
    assert drums.class_of(36) == "kick"
    assert drums.class_of(37) == "rim"
    assert drums.class_of(56) is None  # cowbell is dropped
    assert drums.canonical_note("kick") == 36
    assert drums.coarse_of("rim") == "snare"
    assert drums.coarse_of("china") == "cymbals"
    assert {drums.class_of(n) for n in (41, 43)} == {"tom_floor"}
    assert {drums.class_of(n) for n in (45, 47)} == {"tom_mid"}
    assert {drums.class_of(n) for n in (48, 50)} == {"tom_high"}


def test_paths_come_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKLINE_DATA_DIR", str(tmp_path / "tabs"))
    monkeypatch.setenv("BACKLINE_CONFIG", str(CONFIGS / "tiny.yaml"))
    paths = PathSettings(_env_file=None)  # type: ignore[call-arg]
    assert paths.data_dir == tmp_path / "tabs"
    assert load_config().name == "tiny"
