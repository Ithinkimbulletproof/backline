# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Run adapter training in a separate process and mirror its progress into the database.

A child process (``backline adapter train``) keeps GPU memory and crashes away from the
web server and the task consumer; it appends progress as JSON lines that are read here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from web.studio.models import AdapterJob

POLL_SECONDS = 1.0


def adapter_command(references: Sequence[str], progress_file: Path) -> list[str]:
    command = [sys.executable, "-m", "backline.cli", "adapter", "train"]
    for reference in references:
        command += ["--ref", reference]
    return [*command, "--progress-file", str(progress_file)]


def _apply(job: AdapterJob, line: str) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if "error" in event:
        job.status, job.message = AdapterJob.Status.FAILED, str(event["error"])
    elif event.get("done"):
        job.status = AdapterJob.Status.CACHED if event.get("cached") else AdapterJob.Status.DONE
        job.key, job.progress, job.message = str(event.get("key", "")), 1.0, ""
    elif "step" in event and "total" in event:
        job.progress = min(float(event["step"]) / max(float(event["total"]), 1.0), 0.99)
        loss = event.get("loss")
        job.message = f"шаг {event['step']} из {event['total']}" + (
            f", loss {loss:.3f}" if loss else ""
        )


def run_adapter_job(job_id: int, *, poll: float = POLL_SECONDS) -> AdapterJob:
    """Train the job's adapter and keep the job row up to date until the child exits."""
    job = AdapterJob.objects.get(pk=job_id)
    job.status, job.progress, job.message = AdapterJob.Status.RUNNING, 0.0, ""
    job.save()
    with tempfile.TemporaryDirectory() as folder:
        progress_file = Path(folder) / "progress.jsonl"
        progress_file.touch()
        error_file = Path(folder) / "stderr.txt"
        with error_file.open("w", encoding="utf-8") as errors:
            process = subprocess.Popen(
                adapter_command(job.references, progress_file),
                stdout=subprocess.DEVNULL,
                stderr=errors,
            )
        read = 0
        while True:
            finished = process.poll() is not None
            lines = progress_file.read_text(encoding="utf-8").splitlines()
            for line in lines[read:]:
                _apply(job, line)
            read = len(lines)
            job.save()
            if finished:
                break
            time.sleep(poll)
        error_lines = error_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if process.returncode != 0 and job.status != AdapterJob.Status.FAILED:
        job.status = AdapterJob.Status.FAILED
        job.message = error_lines[-1] if error_lines else "adapter training failed"
    job.save()
    return job
