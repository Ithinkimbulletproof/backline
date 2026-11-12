# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

import pytest

from backline.config import BacklineConfig
from backline.drums import DrumMapError, Substitution, load_drum_map
from backline.music import Dynamic

MAPS = Path(__file__).resolve().parents[2] / "configs" / "drum_maps"


@pytest.mark.parametrize("name", ["gm.yaml", "sd3_signature_part2.yaml"])
def test_shipped_maps_cover_every_class(tiny_config: BacklineConfig, name: str) -> None:
    load_drum_map(MAPS / name).check_covers(tiny_config.drums.classes)


def test_gm_map_has_no_substitutions() -> None:
    gm = load_drum_map(MAPS / "gm.yaml")
    assert gm.substitutions() == []
    assert gm.note_for("snare", Dynamic.MAX) == 38


def test_sd3_substitutions_name_the_replacement_sound() -> None:
    sd3 = load_drum_map(MAPS / "sd3_signature_part2.yaml")
    assert sd3.substitutions() == [
        Substitution(missing="china", played_as="crash2", note=57),
        Substitution(missing="splash", played_as="crash1", note=49),
    ]


def test_sd3_rimshot_and_kit_specific_notes() -> None:
    sd3 = load_drum_map(MAPS / "sd3_signature_part2.yaml")
    assert sd3.note_for("snare", Dynamic.NORMAL) == 38
    assert sd3.note_for("snare", Dynamic.MAX) == 40
    assert sd3.note_for("kick", Dynamic.MAX) == 36
    assert sd3.note_for("tom_high") == 48


def test_incomplete_map_is_rejected(tiny_config: BacklineConfig, tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("name: partial\nclasses:\n  kick: {note: 36}\n", encoding="utf-8")
    with pytest.raises(DrumMapError, match="missing classes: china"):
        load_drum_map(path).check_covers(tiny_config.drums.classes)


def test_malformed_map_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\nclasses:\n  kick: {note: 200}\n", encoding="utf-8")
    with pytest.raises(DrumMapError, match="invalid"):
        load_drum_map(path)


def test_unknown_class_lookup_is_an_error() -> None:
    gm = load_drum_map(MAPS / "gm.yaml")
    with pytest.raises(DrumMapError, match="cowbell"):
        gm.note_for("cowbell")
