# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Django settings for the local-only Backline UI.

Paths come from the same ``BACKLINE_*`` variables (or ``.env``) as the core package.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from backline.config import PathSettings

BASE_DIR = Path(__file__).resolve().parent.parent
PATHS = PathSettings()


class UiSettings(BaseSettings):
    """Look of the web UI, from ``BACKLINE_*`` variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="BACKLINE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    site_name: str = "Backline"
    """Shown in the header and the page titles."""


SITE_NAME = UiSettings().site_name


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG = _env_bool("DJANGO_DEBUG", True)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or "insecure-local-only-key"
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]", "testserver"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "huey.contrib.djhuey",
    "web.library",
    "web.studio",
]

# Background jobs: Huey on SQLite (no broker), thread workers (process workers are not
# available on Windows). Run the consumer next to the dev server: `python manage.py run_huey`.
HUEY_FILE = Path(os.environ.get("HUEY_FILE", PATHS.data_dir / "huey.sqlite3"))
HUEY_FILE.parent.mkdir(parents=True, exist_ok=True)
HUEY = {
    "huey_class": "huey.SqliteHuey",
    "name": "backline",
    "filename": str(HUEY_FILE),
    "immediate": _env_bool("HUEY_IMMEDIATE", False),
    "consumer": {"workers": 1, "worker_type": "thread"},
}

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "web.urls"
WSGI_APPLICATION = "web.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "web" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.template.context_processors.media",
                "web.context_processors.site",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": Path(os.environ.get("DJANGO_DB_PATH", BASE_DIR / "db.sqlite3")),
    }
}

LANGUAGE_CODE = "ru"
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "web" / "static"]
MEDIA_URL = "media/"
MEDIA_ROOT = PATHS.media_dir.resolve()

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
