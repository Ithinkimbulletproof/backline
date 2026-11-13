# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import pytest
from django.test import Client
from pytest_django.fixtures import Settings


@pytest.mark.django_db
def test_studio_page(client: Client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    page = response.content.decode()
    assert "Студия" in page
    assert 'lang="ru"' in page


@pytest.mark.django_db
def test_library_page(client: Client) -> None:
    response = client.get("/library/")
    assert response.status_code == 200
    assert "Библиотека" in response.content.decode()


@pytest.mark.django_db
def test_site_name_is_configurable(client: Client, settings: Settings) -> None:
    settings.SITE_NAME = "My Studio"
    page = client.get("/library/").content.decode()
    assert '<a class="brand" href="/">My Studio</a>' in page
    assert "<title>Библиотека · My Studio</title>" in page


def test_site_name_defaults_to_the_label(monkeypatch: pytest.MonkeyPatch) -> None:
    from web.settings import UiSettings

    monkeypatch.delenv("BACKLINE_SITE_NAME", raising=False)
    assert UiSettings(_env_file=None).site_name == "Fazanka Records"
