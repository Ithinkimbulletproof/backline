# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
from django.apps import AppConfig


class LibraryConfig(AppConfig):
    name = "web.library"
    label = "library"
    verbose_name = "Библиотека"
