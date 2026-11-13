# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Russian texts for triage problems (the core describes them in English)."""

from __future__ import annotations

from typing import Any

ROLES = {"guitar": "гитара", "bass": "бас", "drums": "барабаны", "other": "другой инструмент"}


def _share(value: Any) -> str:
    return f"{float(value):.0%}" if value is not None else "?"


def describe(problem: dict[str, Any]) -> str:
    """The problem and what to do about it, in Russian."""
    code = problem.get("code", "")
    detail = str(problem.get("detail") or "")
    value, limit = problem.get("value"), problem.get("limit")
    track = f"Дорожка {int(problem.get('track') or 0) + 1} «{problem.get('track_name') or ''}»"
    role = ROLES.get(detail, detail)
    share, need = _share(value), _share(limit)
    thin = f": дорожка есть, но в ней всего {int(value)} нот" if value else ""
    texts = {
        "old_format": "Guitar Pro 6 (.gpx): откройте в Guitar Pro 7 или 8 и сохраните как .gp",
        "read_error": f"Файл не читается ({detail}). Попробуйте пересохранить его в Guitar Pro",
        "bass_not_recognised": f"{track} похожа на бас, но распознана как {role}. "
        "Если это бас — назовите её Bass",
        "drums_not_recognised": f"{track} похожа на барабаны, но распознана как {role}. "
        "Если это барабаны — назовите её Drums",
        "guitar_not_recognised": f"{track} похожа на гитару, но распознана как {role}. "
        "Если это гитара — назовите её Guitar",
        "bass_coverage_low": f"Бас звучит только в {share} тактов с гитарой (нужно {need}): "
        "допишите недостающие такты",
        "drums_coverage_low": f"Барабаны звучат только в {share} тактов с гитарой "
        f"(нужно {need}): допишите недостающие такты",
        "unsupported_format": f"Формат не поддерживается: {detail}",
        "no_bass": "Нет партии баса" + thin,
        "no_drums": "Нет партии барабанов" + thin,
        "no_guitar": "Нет партии гитары" + thin,
        "bass_mostly_missing": f"Бас есть только в {share} тактов с гитарой — "
        "проще найти другую версию",
        "drums_mostly_missing": f"Барабаны есть только в {share} тактов с гитарой — "
        "проще найти другую версию",
        "ambiguous_role": f"{track} распознана как {role} неуверенно — проверьте",
        "unmapped_percussion": f"{share} барабанных нот — другая перкуссия, она не используется "
        f"({detail})",
        "no_artist": "Не указан исполнитель: назовите файл «Исполнитель - Название»",
        "extra_bass_ignored": "Басовых дорожек несколько, взята самая полная "
        f"(лишних: {int(value or 0)}) — проверьте",
        "version_of": f"Другая версия: в датасете {detail}",
    }
    return texts.get(code, code)
