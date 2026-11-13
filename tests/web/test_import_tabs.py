# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import CommandError, call_command
from django.test import Client

from backline.config import BacklineConfig
from tests.library import build_library
from web.library.models import Song, Track

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BACKLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BACKLINE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BACKLINE_CONFIG", str(CONFIGS / "tiny.yaml"))
    return tmp_path


@pytest.mark.django_db
def test_import_tabs_fills_the_library(
    env: Path, tiny_config: BacklineConfig, client: Client
) -> None:
    root = env / "tabs"
    build_library(root, tiny_config)
    out = StringIO()
    call_command("import_tabs", str(root), "--workers", "1", stdout=out)
    assert "web library: 18 new" in out.getvalue()
    assert Song.objects.count() == 18
    assert Song.objects.filter(status="ok").count() == 10
    assert Track.objects.filter(role="bass", ambiguous=True).exists()

    (root / "Night Drive.gp").unlink()
    out = StringIO()
    call_command("import_tabs", str(root), "--workers", "1", stdout=out)
    assert "1 removed" in out.getvalue()

    page = client.get("/library/").content.decode()
    assert "в датасете" in page
    assert "Grey Harbor" in page


@pytest.mark.django_db
def test_library_is_sorted_for_fixing(
    env: Path, tiny_config: BacklineConfig, client: Client
) -> None:
    paths = build_library(env / "tabs", tiny_config)
    out = StringIO()
    call_command("import_tabs", str(env / "tabs"), "--workers", "1", stdout=out)
    assert "triage       fixable=2, rejected=2" in out.getvalue()
    gpx = Song.objects.get(path=paths["gpx"])
    assert gpx.category == Song.Category.FIXABLE
    assert gpx.problems[0]["code"] == "old_format"
    assert Song.objects.get(path=paths["no_bass"]).category == Song.Category.REJECTED

    page = client.get("/library/").content.decode()
    assert page.index("Можно поправить") < page.index("Годится")
    assert "откройте в Guitar Pro 7 или 8 и сохраните как .gp" in page
    fixable = client.get("/library/", {"category": "fixable"}).content.decode()
    assert "legacy.gpx" in fixable
    assert "Bassless" not in fixable
    rejected = client.get("/library/", {"category": "rejected"}).content.decode()
    assert "Нет партии баса" in rejected
    assert "Бас есть только в 10% тактов" in rejected
    found = client.get("/library/", {"q": "cowbell"}).content.decode()
    assert "другая перкуссия" in found
    assert "Grey Harbor" not in found

    export = client.get("/library/export.csv", {"category": "rejected"})
    assert export["Content-Type"].startswith("text/csv")
    text = export.content.decode("utf-8")
    assert text.startswith("\N{BYTE ORDER MARK}Категория;Файл;")
    lines = text.strip().splitlines()
    assert len(lines) == 3
    assert all(line.startswith("не подходит;") for line in lines[1:])

    status = client.get("/library/model/").content.decode()
    assert "можно поправить" in status


@pytest.mark.django_db
def test_import_tabs_reports_missing_folder(env: Path) -> None:
    with pytest.raises(CommandError, match="not found"):
        call_command("import_tabs", str(env / "missing"))
