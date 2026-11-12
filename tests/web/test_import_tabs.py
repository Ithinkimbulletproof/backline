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
    assert "web library: 17 new" in out.getvalue()
    assert Song.objects.count() == 17
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
def test_import_tabs_reports_missing_folder(env: Path) -> None:
    with pytest.raises(CommandError, match="not found"):
        call_command("import_tabs", str(env / "missing"))
