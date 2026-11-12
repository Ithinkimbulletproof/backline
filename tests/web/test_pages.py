# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
import pytest
from django.test import Client


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
