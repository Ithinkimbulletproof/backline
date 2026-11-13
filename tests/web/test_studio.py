# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""The studio end to end on a tiny model trained on a synthetic library."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from pytest_django.fixtures import Settings

from backline.config import BacklineConfig
from backline.drums import load_drum_map
from backline.formats.midi import write_midi
from backline.score import Score, TrackRole
from backline.synth import generate_corpus
from web.library.models import Song
from web.studio import services
from web.studio.models import GenerationJob, Riff, Variant
from web.studio.render import piano_roll


@pytest.fixture
def studio(
    workspace: Path,
    use_workspace: Callable[[Path], None],
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tmp_path: Path,
) -> Path:
    use_workspace(workspace)
    settings.MEDIA_ROOT = tmp_path / "media"
    # Run generation in the request instead of the Huey queue.
    monkeypatch.setattr("web.studio.views.generate_task", services.run_generation)
    return tmp_path / "media"


def _riff_file(config: BacklineConfig, folder: Path) -> SimpleUploadedFile:
    """A guitar-only MIDI riff that is not in the training library."""
    song = generate_corpus(config, n_songs=1, seed=101)[0].score
    guitar = song.tracks_with_role(TrackRole.GUITAR)[0]
    path = folder / "riff.mid"
    write_midi(Score(song.n_bars, song.time_signatures, song.tempos, [guitar], song.tpq), path)
    return SimpleUploadedFile("Мой рифф.mid", path.read_bytes(), content_type="audio/midi")


@pytest.mark.django_db
def test_upload_rejects_other_files(client: Client, settings: Settings, tmp_path: Path) -> None:
    settings.MEDIA_ROOT = tmp_path
    text = SimpleUploadedFile("notes.txt", b"hello")
    assert client.post("/riffs/", {"riff": text}).status_code == 400
    broken = SimpleUploadedFile("broken.mid", b"not a midi file")
    response = client.post("/riffs/", {"riff": broken})
    assert response.status_code == 400
    assert "Не удалось прочитать файл" in response.content.decode()
    assert not Riff.objects.exists()


@pytest.mark.django_db
def test_studio_flow(client: Client, studio: Path, tiny_config: BacklineConfig) -> None:
    response = client.post("/riffs/", {"riff": _riff_file(tiny_config, studio.parent)})
    assert response.status_code == 302
    riff = Riff.objects.get()
    assert response["Location"] == f"/riffs/{riff.pk}/"
    assert riff.name == "Мой рифф"
    assert 1 <= riff.n_bars <= services.max_bars()
    assert len(riff.plan) == riff.n_bars

    page = client.get(f"/riffs/{riff.pk}/").content.decode()
    for text in (
        "Гармонический план",
        "Сгенерировать партии под план",
        "Референсы",
        'id="cell-0-0"',
    ):
        assert text in page
    assert "Модель ещё не обучена" not in page
    assert f'value="{riff.tempo}"' in page  # not localized to "120,0"
    assert 'hx-trigger="load"' in page  # the first plan starts by itself

    # Settings are clamped and ask the plan to refresh.
    response = client.post(
        f"/riffs/{riff.pk}/settings/", {"tempo": "500", "n_bars": "2", "start_bar": "1"}
    )
    assert response["HX-Trigger"] == "plan-updated"
    riff.refresh_from_db()
    assert (riff.tempo, riff.n_bars, len(riff.plan)) == (300.0, 2, 2)
    client.post(f"/riffs/{riff.pk}/settings/", {"tempo": "120"})

    # A generated plan fills every bar.
    shown = client.post(f"/riffs/{riff.pk}/generate/", {"kind": "plan"}).content.decode()
    job = GenerationJob.objects.get(kind="plan")
    assert job.status == "done", job.message
    riff.refresh_from_db()
    assert all(row[0] for row in riff.plan)
    assert f'id="cell-0-0" name="cell-0-0" value="{riff.plan[0][0]}"' in shown
    response = client.get(f"/generation/{job.pk}/")
    assert response["HX-Trigger"] == "plan-updated"

    # Edits are validated cell by cell, with the place of the mistake.
    cells = {
        f"cell-{b}-{i}": value for b, row in enumerate(riff.plan) for i, value in enumerate(row)
    }
    bad = client.post(f"/riffs/{riff.pk}/plan/save/", {**cells, "cell-1-0": "Hx7"}).content.decode()
    assert "такт 2, доля 1" in bad
    blocked = client.post(f"/riffs/{riff.pk}/generate/", {"kind": "parts"})
    assert "такт 2, доля 1" in blocked.content.decode()
    assert not GenerationJob.objects.filter(kind="parts").exists()
    good = client.post(f"/riffs/{riff.pk}/plan/save/", {**cells, "cell-0-0": "E5"}).content.decode()
    assert 'class="error"' not in good
    riff.refresh_from_db()
    assert riff.plan[0][0] == "E5"

    # One pass, no regeneration until green: parts under the edited plan.
    assert client.post(f"/riffs/{riff.pk}/strength/", {"strength": "0.5"}).content == b"0.5"
    client.post(f"/riffs/{riff.pk}/strength/", {"until_green": ["off"]})
    riff.refresh_from_db()
    assert (riff.strength, riff.until_green) == (0.5, False)
    response = client.post(f"/riffs/{riff.pk}/generate/", {"kind": "parts"})
    assert response["HX-Trigger"] == "variants-updated"
    job = GenerationJob.objects.get(kind="parts")
    assert job.status == "done", job.message
    variants = list(Variant.objects.filter(job=job))
    assert len(variants) == tiny_config.generation.variants
    for variant in variants:
        assert variant.roll.startswith('<svg class="roll"')
        assert variant.verdict in ("green", "yellow", "red")
        assert variant.plan[0][0] == "E5"
        for name in (variant.bass_file, variant.drums_file, variant.mix_file):
            assert (studio / name).is_file()
    assert client.get(f"/generation/{job.pk}/")["HX-Trigger"] == "variants-updated"
    listing = client.get(f"/riffs/{riff.pk}/variants/").content.decode()
    assert "Вариант 1" in listing
    assert f'src="/media/{variants[0].mix_file}"' in listing
    assert "эвристика" in listing


@pytest.mark.django_db
def test_references_and_adapter(
    client: Client, studio: Path, tiny_config: BacklineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    Song.objects.create(id="a1", path="a.gp5", title="Первая", artist="Группа", status="ok")
    Song.objects.create(id="b2", path="b.gp5", title="Second", artist="Band", status="ok")
    Song.objects.create(id="c3", path="c.gp5", title="Hidden", artist="Band", status="duplicate")
    client.post("/riffs/", {"riff": _riff_file(tiny_config, studio.parent)})
    riff = Riff.objects.get()

    found = client.get(f"/riffs/{riff.pk}/references/", {"q": "band"}).content.decode()
    assert "Second" in found
    assert "Hidden" not in found  # other versions are not offered
    assert "Первая" not in found

    response = client.post(f"/riffs/{riff.pk}/references/toggle/", {"song": "b2"})
    assert response["HX-Trigger"] == "references-changed"
    assert "Подготовить адаптер" in response.content.decode()
    client.post(f"/riffs/{riff.pk}/references/toggle/", {"song": "nope"})
    riff.refresh_from_db()
    assert riff.references == ["b2"]

    queued: list[int] = []
    monkeypatch.setattr("web.studio.views.train_adapter_task", queued.append)
    status = client.post(f"/riffs/{riff.pk}/adapter/").content.decode()
    riff.refresh_from_db()
    assert queued == [riff.adapter_job_id]
    assert "every 2s" in status

    client.post(f"/riffs/{riff.pk}/references/toggle/", {"song": "b2"})
    riff.refresh_from_db()
    assert riff.references == []
    assert riff.adapter_job is None


def test_piano_roll_marks_matched_bars(tiny_config: BacklineConfig) -> None:
    score = generate_corpus(tiny_config, n_songs=1, seed=3)[0].score
    groups = {36: "kick", 38: "snare", 42: "hats"}
    svg = piano_roll(score, groups, highlight=[0, 99])
    assert svg.startswith('<svg class="roll"')
    assert svg.endswith("</svg>")
    assert svg.count('class="match"') == 1
    assert 'class="bass"' in svg
    assert 'class="hit kick"' in svg


def test_drum_kit_substitutions_are_reported() -> None:
    kit = load_drum_map(
        Path(__file__).resolve().parents[2] / "configs" / "drum_maps" / "sd3_signature_part2.yaml"
    )
    warnings = services.substitution_warnings(kit, {"kick", "snare", "china"})
    assert warnings == ["china в ките нет — сыграно Crash 2"]
    assert services.substitution_warnings(kit, {"kick", "crash1"}) == []
