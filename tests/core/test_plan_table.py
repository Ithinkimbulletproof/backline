# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from backline.music import ChordQuality, Mode
from backline.tokenizer.encode import PlanChord
from backline.tokenizer.plan_table import (
    PlanTableError,
    function_cells,
    plan_to_table,
    table_to_plan,
)

P, MAJ = ChordQuality.POWER, ChordQuality.MAJOR
E_MINOR = (4, Mode.MINOR)


def test_table_round_trip() -> None:
    plan = [
        PlanChord(0, 0, 0, P, *E_MINOR),
        PlanChord(0, 72, 3, P, *E_MINOR),
        PlanChord(1, 0, 5, P, *E_MINOR),
        PlanChord(2, 0, 0, P, *E_MINOR),
        PlanChord(2, 48, 10, MAJ, *E_MINOR),
        PlanChord(3, 0, None, None, *E_MINOR),
    ]
    table = plan_to_table(plan, 4, 4, 96)
    assert table == [
        ["E5", "", "", "G5"],
        ["A5", "", "", ""],
        ["E5", "", "D", ""],
        ["N.C.", "", "", ""],
    ]
    assert table_to_plan(table, 4, Mode.MINOR, 96) == plan


def test_empty_bar_continues_the_previous_chord() -> None:
    plan = table_to_plan([["E5", "", "", ""], ["", "", "G5", ""]], 4, Mode.MINOR, 96)
    assert [(c.bar, c.pos, c.degree) for c in plan] == [(0, 0, 0), (1, 0, 0), (1, 48, 3)]


def test_errors_name_the_cell() -> None:
    with pytest.raises(PlanTableError, match="такт 1, доля 1"):
        table_to_plan([["", "", "", ""]], 4, Mode.MINOR, 96)
    with pytest.raises(PlanTableError, match="такт 2, доля 3") as caught:
        table_to_plan([["E5", "", "", ""], ["A5", "", "Hm", ""]], 4, Mode.MINOR, 96)
    assert (caught.value.bar, caught.value.beat) == (1, 2)


def test_function_labels() -> None:
    cells = function_cells([["E5", "", "D", "C5"], ["A5", "", "B7", "Xx"]], tonic=4)
    assert [c.function for c in cells[0]] == ["T", "", "bVII", "bVI"]
    assert [c.function for c in cells[1]] == ["S", "", "D", "?"]
