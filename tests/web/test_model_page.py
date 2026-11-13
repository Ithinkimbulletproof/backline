# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""The read-only pipeline status page."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from django.test import Client

from web.library.status import sparkline


@pytest.mark.django_db
def test_model_page(client: Client, workspace: Path, use_workspace: Callable[[Path], None]) -> None:
    use_workspace(workspace)
    page = client.get("/library/model/").content.decode()
    for text in (
        "Модель и данные",
        "Датасет",
        "Индекс оригинальности",
        "Базовая модель",
        "train_base",
    ):
        assert text in page
    assert page.count("готово") >= 3  # dataset, index, model
    assert '<svg class="spark"' in page
    assert "пороги (бас)" in page


@pytest.mark.django_db
def test_model_page_when_nothing_is_built(
    client: Client, use_workspace: Callable[[Path], None], tmp_path: Path
) -> None:
    use_workspace(tmp_path)
    page = client.get("/library/model/").content.decode()
    assert "не сделано" in page
    assert "Адаптеров пока нет" in page
    assert client.get("/").status_code == 200


def test_sparkline() -> None:
    assert sparkline([1.0]) == ""
    svg = sparkline([3.0, 2.0, float("nan"), 1.0])
    assert svg.startswith('<svg class="spark"')
    assert "<polyline" in svg
