# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Transformer, constrained generation, metrics and the training loop (tiny, on CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backline.config import BacklineConfig, load_config
from backline.dataset.build import build_dataset
from backline.dataset.reader import SegmentDataset
from backline.formats.guitarpro import write_gp5
from backline.ingest.pipeline import LibraryPaths, run_ingest
from backline.model.generate import GenerationRequest, generate, generated_part
from backline.model.grammar import Grammar
from backline.model.metrics import WHOLE_NOTE_STEPS, segment_metrics
from backline.model.train import Trainer, collate, load_model
from backline.model.transformer import Backline, ModelShape
from backline.music import ChordQuality, Mode
from backline.originality.build import build_originality_index
from backline.synth import generate_corpus
from backline.tokenizer.encode import PlanChord, decode
from backline.tokenizer.render import plan_tokens, to_score
from backline.tokenizer.vocab import Vocab

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="module")
def workspace(
    tmp_path_factory: pytest.TempPathFactory, tiny_config: BacklineConfig
) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("model")
    tabs = root / "tabs"
    tabs.mkdir()
    for index, song in enumerate(generate_corpus(tiny_config, n_songs=24, seed=91)):
        write_gp5(song.score, tabs / f"{index:02d} {song.score.artist} - {song.score.title}.gp5")
    config = tiny_config.model_copy(
        update={"ingest": tiny_config.ingest.model_copy(update={"val_fraction": 0.25})}
    )
    library = LibraryPaths(root / "catalog.sqlite", root / "scores", root / "reports")
    run_ingest(tabs, config, library, workers=1)
    build_dataset(tiny_config, library, root / "dataset")
    build_originality_index(tiny_config, library, root / "index")
    return {"root": root, "dataset": root / "dataset", "index": root / "index"}


def _tiny_model(config: BacklineConfig) -> tuple[Backline, Vocab]:
    vocab = Vocab.from_config(config)
    torch.manual_seed(0)
    return Backline(ModelShape.from_config(config.model, len(vocab))), vocab


def test_base_model_size() -> None:
    config = load_config(CONFIGS / "base.yaml")
    model = Backline(ModelShape.from_config(config.model, len(Vocab.from_config(config))))
    assert 30e6 <= model.parameter_count() <= 80e6


def test_small_model_size() -> None:
    config = load_config(CONFIGS / "small.yaml")
    model = Backline(ModelShape.from_config(config.model, len(Vocab.from_config(config))))
    assert 10e6 <= model.parameter_count() <= 25e6


def test_cache_matches_full_forward(tiny_config: BacklineConfig) -> None:
    model, vocab = _tiny_model(tiny_config)
    model.eval()
    tokens = torch.randint(0, len(vocab), (2, 12))
    full, _ = model(tokens)
    prefix, cache = model(tokens[:, :8])
    step_logits = [prefix[:, -1]]
    for t in range(8, 12):
        logits, cache = model(tokens[:, t : t + 1], cache)
        step_logits.append(logits[:, -1])
    assert torch.allclose(full[:, 7:12], torch.stack(step_logits, dim=1), atol=1e-4)


def test_loss_ignores_the_condition(tiny_config: BacklineConfig) -> None:
    model, vocab = _tiny_model(tiny_config)
    tokens = torch.randint(5, len(vocab), (1, 20))
    targets = torch.randint(5, len(vocab), (1, 20))
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, 10:] = True
    changed = targets.clone()
    changed[:, :10] = 1
    assert torch.isclose(model.loss(tokens, targets, mask), model.loss(tokens, changed, mask))


def test_collate_masks_from_the_plan(
    workspace: dict[str, Path], tiny_config: BacklineConfig
) -> None:
    dataset = SegmentDataset(workspace["dataset"], "train", tiny_config)
    samples = [dataset[0], dataset[1]]
    inputs, targets, mask = collate(samples, dataset.vocab.pad)
    first = samples[0]
    position = first.target_start - 1
    assert targets[0, position] == dataset.vocab.id("PLAN")
    assert bool(mask[0, position])
    assert not bool(mask[0, position - 1])
    assert inputs.shape == targets.shape == mask.shape


def test_untrained_constrained_generation_is_valid(
    workspace: dict[str, Path], tiny_config: BacklineConfig
) -> None:
    model, vocab = _tiny_model(tiny_config)
    dataset = SegmentDataset(workspace["dataset"], "val", tiny_config)
    record = dataset.index[0]
    sample = dataset[0]
    prompt = sample.tokens[: sample.target_start].tolist()
    meter = decode(prompt, vocab).meter
    profile = tiny_config.instruments.bass.active
    request = GenerationRequest(
        prompt,
        int(record["n_bars"]),
        meter[0] * WHOLE_NOTE_STEPS // meter[1],
        (profile.lowest, profile.highest),
    )
    for sequence in generate(model, vocab, request, variants=3, seed=1):
        decoded = decode(sequence, vocab)
        metrics = segment_metrics(decoded, tiny_config, int(record["n_bars"]))
        assert metrics.valid
        assert vocab.token(sequence[-1]) == "EOS"
        part = generated_part(sequence, vocab)
        assert vocab.token(part[0]) == "PLAN"
        score = to_score(decoded, tiny_config, int(record["n_bars"]))
        assert score.tracks[0].notes or score.tracks[1].notes


def test_forced_plan_is_followed(workspace: dict[str, Path], tiny_config: BacklineConfig) -> None:
    model, vocab = _tiny_model(tiny_config)
    dataset = SegmentDataset(workspace["dataset"], "val", tiny_config)
    record = dataset.index[0]
    n_bars = int(record["n_bars"])
    sample = dataset[0]
    prompt = sample.tokens[: sample.target_start].tolist()
    header = decode(prompt, vocab)
    plan = [
        PlanChord(bar, 0, (0, 5)[bar % 2], ChordQuality.POWER, header.tonic, header.mode)
        for bar in range(n_bars)
    ]
    forced = plan_tokens(plan, n_bars, (header.tonic, header.mode), vocab)
    profile = tiny_config.instruments.bass.active
    request = GenerationRequest(
        prompt, n_bars, header.meter[0] * 24, (profile.lowest, profile.highest), forced
    )
    (sequence,) = generate(model, vocab, request, variants=1, seed=2)
    decoded = decode(sequence, vocab)
    assert [(c.bar, c.degree) for c in decoded.plan] == [(c.bar, c.degree) for c in plan]


def test_grammar_rejects_broken_forced_tokens(tiny_config: BacklineConfig) -> None:
    vocab = Vocab.from_config(tiny_config)
    grammar = Grammar(vocab, n_bars=1, bar_steps=96, bass_range=(28, 67))
    with pytest.raises(ValueError, match="not allowed"):
        grammar.accepts(grammar.start(), [vocab.id("PLAN"), vocab.id("BASS")])
    with pytest.raises(ValueError, match="first beat"):
        plan_tokens(
            [PlanChord(0, 24, 0, ChordQuality.MAJOR, 4, Mode.MAJOR)], 1, (4, Mode.MAJOR), vocab
        )


def test_cli_train_and_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_config: BacklineConfig
) -> None:
    from typer.testing import CliRunner

    from backline.cli import app
    from backline.formats.midi import write_midi
    from backline.originality.build import slice_score

    tiny = str(CONFIGS / "tiny.yaml")
    for name, folder in (
        ("DATA", "data"),
        ("ARTIFACTS", "artifacts"),
        ("CHECKPOINTS", "ckpt"),
        ("MEDIA", "media"),
    ):
        monkeypatch.setenv(f"BACKLINE_{name}_DIR", str(tmp_path / folder))
    runner = CliRunner()
    tabs = tmp_path / "tabs"
    tabs.mkdir()
    songs = generate_corpus(tiny_config, n_songs=10, seed=17)
    for index, song in enumerate(songs):
        write_gp5(song.score, tabs / f"{index:02d} {song.score.artist} - {song.score.title}.gp5")
    riff = slice_score(songs[0].score, 0, 2)
    riff.tracks = [riff.tracks[0]]
    riff_path = tmp_path / "riff.mid"
    write_midi(riff, riff_path)

    assert runner.invoke(app, ["generate", str(riff_path), "-c", tiny]).exit_code == 1
    for command in (
        ["ingest", str(tabs), "--workers", "1"],
        ["build-dataset"],
        ["build-originality-index"],
    ):
        result = runner.invoke(app, [*command, "-c", tiny])
        assert result.exit_code == 0, result.output
    trained = runner.invoke(app, ["train", "-c", tiny, "--device", "cpu"])
    assert trained.exit_code == 0, trained.output
    assert "best" in trained.output
    generated = runner.invoke(
        app,
        ["generate", str(riff_path), "-c", tiny, "--device", "cpu", "--seed", "3", "--until-green"],
    )
    assert generated.exit_code == 0, generated.output
    assert "variant 2" in generated.output
    assert "originality" in generated.output
    files = sorted((tmp_path / "media" / "generated" / "riff").glob("*.mid"))
    assert [f.name for f in files] == [
        "variant_1_bass.mid",
        "variant_1_drums.mid",
        "variant_2_bass.mid",
        "variant_2_drums.mid",
    ]


def test_training_reduces_loss_and_resumes(
    workspace: dict[str, Path], tiny_config: BacklineConfig
) -> None:
    from backline.originality.index import OriginalityIndex

    checkpoints = workspace["root"] / "checkpoints"
    index = OriginalityIndex.load(workspace["index"])
    trainer = Trainer(tiny_config, workspace["dataset"], checkpoints, device="cpu", index=index)
    summary = trainer.train(resume=False)
    assert summary.steps == tiny_config.train.max_steps
    losses = [h["train_loss"] for h in summary.history]
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
    evaluated = [h for h in summary.history if "val_loss" in h]
    assert evaluated
    assert {"memorization", "plan_adherence", "validity"} <= evaluated[-1].keys()
    assert evaluated[-1]["validity"] == 1.0
    assert (checkpoints / "last.pt").is_file()
    assert (checkpoints / "best.pt").is_file()
    assert (checkpoints / "logs" / "metrics.jsonl").is_file()

    longer = tiny_config.model_copy(
        update={
            "train": tiny_config.train.model_copy(
                update={"max_steps": tiny_config.train.max_steps + 5}
            )
        }
    )
    resumed = Trainer(longer, workspace["dataset"], checkpoints, device="cpu")
    assert resumed.resume()
    assert resumed.step == tiny_config.train.max_steps
    assert resumed.train().steps == tiny_config.train.max_steps + 5

    model, state = load_model(checkpoints / "best.pt", device="cpu")
    assert state["vocab"] == trainer.vocab.fingerprint()
    assert isinstance(model, Backline)
    lines = (checkpoints / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["step"] == tiny_config.train.max_steps + 5
