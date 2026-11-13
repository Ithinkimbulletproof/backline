# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Huey tasks (discovered by ``huey.contrib.djhuey``)."""

from huey.contrib.djhuey import db_task

from web.studio.jobs import run_adapter_job


@db_task()
def train_adapter_task(job_id: int) -> None:
    run_adapter_job(job_id)


@db_task()
def generate_task(job_id: int) -> None:
    from web.studio.services import run_generation

    run_generation(job_id)
