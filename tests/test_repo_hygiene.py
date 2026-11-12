# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Guards for the repository itself: licensing headers and ignore rules."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "web", "tests", "configs")
LICENSE_LINE = "SPDX-License-Identifier: AGPL-3.0-only"


def _files(*suffixes: str) -> list[Path]:
    found = []
    for folder in SOURCE_DIRS:
        for path in (REPO_ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in suffixes:
                found.append(path)
    return found


def test_python_files_carry_the_license_header() -> None:
    missing = []
    for path in [*_files(".py"), REPO_ROOT / "manage.py"]:
        head = path.read_text(encoding="utf-8").splitlines()[:4]
        if not any(LICENSE_LINE in line for line in head):
            missing.append(path.relative_to(REPO_ROOT).as_posix())
    assert missing == []


def test_gitignore_never_hides_source_files() -> None:
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    candidates = [
        p.relative_to(REPO_ROOT).as_posix() for p in _files(".py", ".yaml", ".html", ".css")
    ]
    result = subprocess.run(
        [git, "check-ignore", "--no-index", "--stdin"],
        input="\n".join(candidates),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.stdout.split() == []


def test_core_has_its_own_folder_for_adapters() -> None:
    """The LoRA package directory name must survive the `/adapters/` ignore rule."""
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    probe = "src/backline/adapters/__init__.py"
    result = subprocess.run(
        [git, "check-ignore", "--no-index", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.stdout.strip() == ""
