# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""From a riff file to generated variants: the pipeline the CLI and the web UI share."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backline.config import BacklineConfig
from backline.drums import DrumMap
from backline.formats.midi import write_midi
from backline.ingest.roles import assign_roles
from backline.model.generate import GenerationRequest, generate
from backline.model.metrics import WHOLE_NOTE_STEPS, SegmentMetrics, segment_metrics
from backline.model.transformer import Backline
from backline.originality.check import OriginalityChecker, OriginalityReport
from backline.score import Score, TrackRole
from backline.theory.annotate import annotate_score
from backline.theory.key import KeySpan
from backline.tokenizer.encode import Decoded, PlanChord, SegmentEncoder, decode
from backline.tokenizer.render import plan_tokens, to_score
from backline.tokenizer.vocab import Vocab


@dataclass(slots=True)
class Riff:
    """A riff ready to prompt the model."""

    score: Score
    guitar: int
    bars: range
    key: KeySpan

    @property
    def n_bars(self) -> int:
        return len(self.bars)


@dataclass(slots=True)
class Variant:
    decoded: Decoded
    score: Score
    metrics: SegmentMetrics
    originality: OriginalityReport | None = None
    attempts: int = 1


@dataclass(slots=True)
class GenerationResult:
    riff: Riff
    variants: list[Variant] = field(default_factory=list)


def prepare_riff(
    score: Score, config: BacklineConfig, *, bars: range | None = None, track: int | None = None
) -> Riff:
    """Pick the guitar, the bars (at most the longest segment length) and detect the key."""
    labelled = assign_roles(score, config.ingest.name_hints)
    guitars = [i for i, t in enumerate(labelled.tracks) if t.role is TrackRole.GUITAR]
    if track is not None:
        guitar = track
    elif guitars:
        guitar = guitars[0]
    elif labelled.tracks:
        guitar = 0
    else:
        raise ValueError("the file has no tracks")
    longest = max(config.tokenizer.segment_bars)
    chosen = bars if bars is not None else range(0, min(labelled.n_bars, longest))
    riff_only = Score(
        labelled.n_bars,
        labelled.time_signatures,
        labelled.tempos,
        [labelled.tracks[guitar]],
        labelled.tpq,
    )
    riff_only.tracks[0].role = TrackRole.GUITAR
    key = annotate_score(riff_only).key_at_bar(chosen.start)
    return Riff(labelled, guitar, chosen, key)


def generate_variants(
    model: Backline,
    vocab: Vocab,
    config: BacklineConfig,
    riff: Riff,
    *,
    plan: list[PlanChord] | None = None,
    variants: int | None = None,
    checker: OriginalityChecker | None = None,
    references: list[str] | None = None,
    until_green: bool = False,
    seed: int | None = None,
    drum_map: DrumMap | None = None,
) -> GenerationResult:
    """Generate variants for a riff; optionally force an edited plan and retry red ones."""
    gen = config.generation
    encoder = SegmentEncoder(config, vocab)
    prompt = encoder.condition(riff.score, riff.guitar, riff.bars, riff.key)
    meter = riff.score.time_signature_at(riff.bars.start)
    bar_steps = meter.numerator * WHOLE_NOTE_STEPS // meter.denominator
    profile = config.instruments.bass.active
    forced = plan_tokens(plan, riff.n_bars, (riff.key.tonic, riff.key.mode), vocab) if plan else []
    request = GenerationRequest(
        prompt, riff.n_bars, bar_steps, (profile.lowest, profile.highest), forced
    )
    tempo = riff.score.tempo_at(riff.score.bar_starts()[riff.bars.start])
    count = variants or gen.variants
    sequences = generate(
        model,
        vocab,
        request,
        variants=count,
        temperature=gen.temperature,
        top_p=gen.top_p,
        constrained=gen.constrained,
        seed=seed,
    )
    result = GenerationResult(riff)
    retries = config.originality.max_regenerations if until_green else 0
    for index, sequence in enumerate(sequences):
        variant = _variant(sequence, vocab, config, riff, tempo, checker, references, drum_map)
        attempt_seed = (seed or 0) + 1000 * (index + 1)
        while (
            variant.originality is not None
            and variant.originality.verdict != "green"
            and variant.attempts <= retries
        ):
            attempt_seed += 1
            (again,) = generate(
                model,
                vocab,
                request,
                variants=1,
                temperature=gen.temperature,
                top_p=gen.top_p,
                constrained=gen.constrained,
                seed=attempt_seed,
            )
            attempts = variant.attempts + 1
            variant = _variant(again, vocab, config, riff, tempo, checker, references, drum_map)
            variant.attempts = attempts
        result.variants.append(variant)
    return result


def _variant(
    sequence: list[int],
    vocab: Vocab,
    config: BacklineConfig,
    riff: Riff,
    tempo: float,
    checker: OriginalityChecker | None,
    references: list[str] | None,
    drum_map: DrumMap | None,
) -> Variant:
    decoded = decode(sequence, vocab)
    score = to_score(decoded, config, riff.n_bars, tempo=tempo, drum_map=drum_map)
    report = None
    if checker is not None:
        report = checker.check(
            to_score(decoded, config, riff.n_bars, tempo=tempo), references=references or []
        )
    return Variant(decoded, score, segment_metrics(decoded, config, riff.n_bars), report)


def export_variant(variant: Variant, folder: Path, stem: str) -> list[Path]:
    """Separate MIDI files for the bass and the drums of a variant."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for track in variant.score.tracks:
        part = track.role.value
        single = Score(
            variant.score.n_bars, variant.score.time_signatures, variant.score.tempos, [track]
        )
        path = folder / f"{stem}_{part}.mid"
        write_midi(single, path)
        paths.append(path)
    return paths
