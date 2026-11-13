# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""Typed configuration.

Behaviour comes from YAML profiles (``configs/base.yaml``, ``configs/tiny.yaml``, your
own ``configs/*.local.yaml``); a profile may ``extends:`` another one, dictionaries are
merged recursively and every other value is replaced. Locations of data and weights
come from environment variables or ``.env`` (see :class:`PathSettings`), so the same
profile works on any machine.

Each phase of the pipeline adds its own section here as it lands.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backline.drums import DrumMap, load_drum_map
from backline.music import DYNAMICS_ORDER, Dynamic, parse_pitch
from backline.score import NoteFlag

Pitch = Annotated[int, BeforeValidator(parse_pitch), Field(ge=0, le=127)]
"""A MIDI pitch that may be written in YAML as a note name (``E1``) or a number."""


class ConfigError(ValueError):
    """Raised when a config profile cannot be read or fails validation."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- paths


class PathSettings(BaseSettings):
    """Where data, artifacts and weights live. Read from ``BACKLINE_*`` env vars or ``.env``.

    Relative paths are resolved against the current working directory, so run commands
    from the repository root.
    """

    model_config = SettingsConfigDict(
        env_prefix="BACKLINE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    config: Path = Path("configs/base.yaml")
    data_dir: Path = Path("data")
    tabs_dir: Path = Path("data/tabs")
    artifacts_dir: Path = Path("artifacts")
    checkpoints_dir: Path = Path("checkpoints")
    adapters_dir: Path = Path("adapters")
    media_dir: Path = Path("media")


# ---------------------------------------------------------------------- instruments


class StringInstrument(_Frozen):
    """A fretted instrument: open strings from the lowest up, and the number of frets."""

    tuning: tuple[Pitch, ...] = Field(min_length=1)
    frets: int = Field(default=24, ge=1, le=36)

    @property
    def lowest(self) -> int:
        """Lowest playable pitch."""
        return min(self.tuning)

    @property
    def highest(self) -> int:
        """Highest playable pitch."""
        return max(self.tuning) + self.frets

    @model_validator(mode="after")
    def _strings_ascend(self) -> Self:
        if list(self.tuning) != sorted(self.tuning):
            raise ValueError("tuning must list strings from the lowest to the highest")
        return self


class GuitarConfig(StringInstrument):
    """Rhythm guitar that carries the riff."""

    input_range: tuple[Pitch, Pitch]
    """Notes outside this range are dropped when reading a riff (e.g. keyswitch notes)."""


class BassConfig(_Frozen):
    """Bass profiles: the active one bounds generation, ``train_range`` bounds the corpus."""

    profile: str
    train_range: tuple[Pitch, Pitch]
    profiles: dict[str, StringInstrument] = Field(min_length=1)

    @property
    def active(self) -> StringInstrument:
        """The instrument the generated parts are written for."""
        return self.profiles[self.profile]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.profile not in self.profiles:
            known = ", ".join(sorted(self.profiles))
            raise ValueError(f"unknown bass profile {self.profile!r} (known: {known})")
        low, high = self.train_range
        if low > high:
            raise ValueError("bass train_range must be (low, high)")
        return self


class InstrumentsConfig(_Frozen):
    """The riff guitar and the bass the parts are written for."""

    guitar: GuitarConfig
    bass: BassConfig


# ------------------------------------------------------------------------- velocity


class VelocityLevel(_Frozen):
    """A loudness level: the lowest MIDI velocity that falls into it, and the one written out."""

    min: int = Field(ge=1, le=127)
    out: int = Field(ge=1, le=127)


class VelocityConfig(_Frozen):
    """Quantization of MIDI velocity into ghost / normal / accent / max."""

    levels: dict[Dynamic, VelocityLevel]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.levels) != set(DYNAMICS_ORDER):
            raise ValueError("velocity levels must define exactly: ghost, normal, accent, max")
        mins = [self.levels[d].min for d in DYNAMICS_ORDER]
        if mins != sorted(set(mins)):
            raise ValueError("velocity level minimums must strictly increase from ghost to max")
        for dynamic in DYNAMICS_ORDER:
            level = self.levels[dynamic]
            upper = self._upper_bound(dynamic)
            if not level.min <= level.out <= upper:
                raise ValueError(f"velocity level {dynamic}: out must lie within its own range")
        return self

    def _upper_bound(self, dynamic: Dynamic) -> int:
        index = DYNAMICS_ORDER.index(dynamic)
        if index + 1 == len(DYNAMICS_ORDER):
            return 127
        return self.levels[DYNAMICS_ORDER[index + 1]].min - 1

    def level_of(self, velocity: int, flags: NoteFlag = NoteFlag.NONE) -> Dynamic:
        """Dynamic level of a note. Articulation flags win over the raw velocity.

        Ghost notes stay ghost even when written loud, so snare ghosts are never lost.
        """
        if flags & NoteFlag.GHOST:
            return Dynamic.GHOST
        if flags & NoteFlag.HEAVY_ACCENT:
            return Dynamic.MAX
        level = Dynamic.GHOST
        for dynamic in DYNAMICS_ORDER:
            if velocity >= self.levels[dynamic].min:
                level = dynamic
        if flags & NoteFlag.ACCENT and DYNAMICS_ORDER.index(level) < 2:
            return Dynamic.ACCENT
        return level

    def velocity_of(self, dynamic: Dynamic) -> int:
        """MIDI velocity written for a dynamic level."""
        return self.levels[dynamic].out


# ---------------------------------------------------------------------------- drums


class DrumsConfig(_Frozen):
    """Drum vocabulary: classes, how input notes map to them, and the output map."""

    classes: tuple[str, ...] = Field(min_length=1)
    input_map: dict[int, str]
    """General MIDI note -> class. Unmapped notes are dropped and counted."""
    coarse: dict[str, tuple[str, ...]]
    """Coarser groups used by the originality check; every class belongs to one group."""
    output_map: Path
    """Drum map file; a relative path is resolved against the profile's directory."""
    other_percussion_warn_ratio: float = Field(default=0.02, ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        classes = set(self.classes)
        if len(classes) != len(self.classes):
            raise ValueError("drum classes must be unique")
        bad_notes = [note for note in self.input_map if not 0 <= note <= 127]
        if bad_notes:
            raise ValueError(f"drum input_map has notes outside 0..127: {bad_notes}")
        unknown = sorted(set(self.input_map.values()) - classes)
        if unknown:
            raise ValueError(f"drum input_map refers to unknown classes: {unknown}")
        unmapped = sorted(classes - set(self.input_map.values()))
        if unmapped:
            raise ValueError(f"drum classes without any input note: {unmapped}")
        grouped = [c for group in self.coarse.values() for c in group]
        if sorted(grouped) != sorted(classes):
            raise ValueError("drum coarse groups must contain every class exactly once")
        return self

    def class_of(self, note: int) -> str | None:
        """Drum class for an input note, or ``None`` if the note is not mapped."""
        return self.input_map.get(note)

    def canonical_note(self, drum_class: str) -> int:
        """First input note listed for a class: how it is written in a General MIDI tab."""
        for note, mapped in self.input_map.items():
            if mapped == drum_class:
                return note
        raise KeyError(drum_class)

    def coarse_of(self, drum_class: str) -> str:
        """Coarse group of a drum class."""
        for group, members in self.coarse.items():
            if drum_class in members:
                return group
        raise KeyError(drum_class)

    def load_output_map(self) -> DrumMap:
        """Load the output drum map and check it covers every class."""
        drum_map = load_drum_map(self.output_map)
        drum_map.check_covers(self.classes)
        return drum_map


# -------------------------------------------------------------------------- ingest


class NameHints(_Frozen):
    """Lower-case fragments of track names that hint at a role (any language)."""

    guitar: tuple[str, ...] = ("guitar", "gtr", "гитар")
    bass: tuple[str, ...] = ("bass", "бас")
    drums: tuple[str, ...] = ("drum", "kit", "perc", "барабан", "ударн")
    rhythm: tuple[str, ...] = ("rhythm", "rhy", "ритм")
    lead: tuple[str, ...] = ("lead", "solo", "соло", "melody", "мелод")


class IngestConfig(_Frozen):
    """Import of a tab library: filtering, metadata, deduplication and the split."""

    min_notes_guitar: int = Field(default=16, ge=1)
    min_notes_bass: int = Field(default=16, ge=1)
    min_notes_drums: int = Field(default=32, ge=1)
    min_coverage: float = Field(default=0.6, ge=0, le=1)
    """Bass and drums must play in at least this share of the bars where a guitar plays."""
    filename_patterns: tuple[str, ...] = ("{artist} - {title}",)
    """Patterns tried on the file name when the tab has no artist/title fields."""
    folder_is_artist: bool = True
    """Use the parent folder as the artist when nothing else names one."""
    val_fraction: float = Field(default=0.1, ge=0, lt=1)
    test_fraction: float = Field(default=0.0, ge=0, lt=1)
    keep_duplicates: bool = False
    """Keep every version of a song (all in the same split) instead of one canonical."""
    workers: int = Field(default=0, ge=0)
    """Parallel reader processes; 0 picks one per CPU core (at most 8)."""
    name_hints: NameHints = NameHints()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be below 1")
        for pattern in self.filename_patterns:
            if "{title}" not in pattern:
                raise ValueError(f"filename pattern {pattern!r} must contain {{title}}")
        return self


# ----------------------------------------------------------------------- tokenizer


class TokenizerConfig(_Frozen):
    """How songs become token sequences."""

    segment_bars: tuple[int, ...] = Field(default=(4, 8), min_length=1)
    """Segment lengths in bars; every length is cut from every song."""
    stride_fraction: float = Field(default=0.5, gt=0, le=1)
    """Hop between segments as a share of the segment length."""
    max_positions: int = Field(default=192, ge=24)
    """Longest bar in grid steps (24 per quarter): 192 = 8/4."""
    tempo_buckets: tuple[int, ...] = (60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 165, 180, 200)
    """Upper edges of tempo buckets in bpm; faster tempos fall into the last bucket."""
    bass_role_tokens: bool = True
    """Write each bass note's harmonic role before its pitch."""
    extra_guitars: bool = True
    """Also cut segments conditioned on other non-lead guitars."""
    min_notes_guitar: int = Field(default=4, ge=1)
    min_notes_bass: int = Field(default=4, ge=1)
    min_notes_drums: int = Field(default=8, ge=1)
    transpose: tuple[int, int] = (-5, 5)
    """Semitone range for transposition augmentation (kept within instrument ranges)."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        low, high = self.transpose
        if low > 0 or high < 0:
            raise ValueError("transpose range must include 0")
        if list(self.tempo_buckets) != sorted(set(self.tempo_buckets)):
            raise ValueError("tempo buckets must strictly increase")
        if any(bars < 1 for bars in self.segment_bars):
            raise ValueError("segment lengths must be at least one bar")
        return self


# --------------------------------------------------------------------- originality


class OriginalityConfig(_Frozen):
    """Copy detection against the corpus. A heuristic, not a legal guarantee."""

    ngram: int = Field(default=8, ge=4, le=16)
    """Bass events per n-gram for the rare-pattern share."""
    seed: int = Field(default=5, ge=3, le=12)
    """Exact bass events that start a candidate match."""
    max_mismatches: int = Field(default=2, ge=0, le=4)
    """Differing events tolerated in any n-gram window of a match."""
    drum_seed_beats: int = Field(default=4, ge=2)
    """Exact drum beats (coarse classes) that start a candidate match."""
    drum_beat_similarity: float = Field(default=0.75, gt=0, le=1)
    """Jaccard similarity of a beat's onsets for it to count as copied."""
    common_seed_share: float = Field(default=0.05, gt=0, le=1)
    """Seeds found in more than this share of song groups are common vocabulary: not searched."""
    red_bars: float = Field(default=2.0, gt=0)
    """A continuous copy at least this long (bars) from one song is red, whatever the score."""
    yellow_bars: float = Field(default=1.0, gt=0)
    yellow_percentile: float = Field(default=95.0, gt=0, lt=100)
    red_percentile: float = Field(default=99.5, gt=0, lt=100)
    """Percentiles of the between-songs similarity distribution that set the thresholds."""
    calibration_songs: int = Field(default=300, ge=10)
    max_regenerations: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.seed > self.ngram:
            raise ValueError("originality seed must not be longer than the n-gram")
        if self.yellow_percentile >= self.red_percentile:
            raise ValueError("yellow percentile must be below the red one")
        if self.yellow_bars > self.red_bars:
            raise ValueError("yellow_bars must not exceed red_bars")
        return self


# --------------------------------------------------------------------------- synth


class SynthConfig(_Frozen):
    """Synthetic corpus used by tests and CI instead of real tabs."""

    n_songs: int = Field(default=200, ge=1)
    n_artists: int = Field(default=40, ge=1)
    sections_per_song: tuple[int, int] = (3, 6)
    time_signatures: dict[str, float] = Field(default_factory=lambda: {"4/4": 0.9, "3/4": 0.1})
    tempo_range: tuple[int, int] = (80, 180)
    shuffle_probability: float = Field(default=0.1, ge=0, le=1)
    full_chord_probability: float = Field(default=0.3, ge=0, le=1)
    """Share of sections where the guitar voices the third instead of power chords."""
    key_change_probability: float = Field(default=0.2, ge=0, le=1)
    lead_guitar_probability: float = Field(default=0.3, ge=0, le=1)
    """Share of songs with a second, lead guitar track."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        low, high = self.sections_per_song
        if not 1 <= low <= high:
            raise ValueError("sections_per_song must be (low, high) with 1 <= low <= high")
        tlow, thigh = self.tempo_range
        if not 20 <= tlow <= thigh:
            raise ValueError("tempo_range must be (low, high) in bpm")
        for meter in self.time_signatures:
            if meter not in ("4/4", "3/4"):
                raise ValueError(f"synthetic meter {meter!r} is not supported (4/4, 3/4)")
        return self


# --------------------------------------------------------------------------- model


class ModelConfig(_Frozen):
    """Transformer size. Used from the model phase on; ``tiny`` runs on a CPU in CI."""

    d_model: int = Field(ge=8)
    n_layers: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    context_length: int = Field(ge=16)
    dropout: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


class TrainConfig(_Frozen):
    """Base-model training."""

    batch_size: int = Field(default=8, ge=1)
    grad_accum: int = Field(default=4, ge=1)
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    warmup_steps: int = Field(default=500, ge=0)
    max_steps: int = Field(default=20000, ge=1)
    """Optimizer steps (each covers ``batch_size * grad_accum`` segments)."""
    eval_every: int = Field(default=1000, ge=1)
    checkpoint_every: int = Field(default=1000, ge=1)
    eval_batches: int = Field(default=50, ge=1)
    grad_clip: float = Field(default=1.0, gt=0)
    augment: bool = True
    memorization_prompts: int = Field(default=32, ge=0)
    """Train and val prompts generated at every evaluation for the memorization rate."""
    memorization_weight: float = Field(default=1.0, ge=0)
    """Checkpoint selection: val loss + weight * memorization rate."""
    bf16: bool = True
    """Mixed precision on CUDA (ignored on CPU)."""
    num_workers: int = Field(default=0, ge=0)


class AdapterConfig(_Frozen):
    """LoRA adapters trained on a set of reference songs."""

    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: tuple[str, ...] = ("qkv", "proj", "gate", "up", "down")
    steps: int = Field(default=600, ge=1)
    batch_size: int = Field(default=8, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    reference_epochs: float = Field(default=4.0, gt=0)
    """Passes over the reference segments; sets the reference share of each batch."""
    reference_share: float | None = Field(default=None, gt=0, le=1)
    """Fixed share of reference segments per batch; ``None`` derives it from reference_epochs."""
    min_share: float = Field(default=0.2, gt=0, le=1)
    max_share: float = Field(default=0.8, gt=0, le=1)
    eval_every: int = Field(default=100, ge=1)
    reference_prompts: int = Field(default=8, ge=0)
    stop_memorization: float = Field(default=0.25, ge=0, le=1)
    """Stop early once this share of generated reference prompts copies the references."""
    strengths: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    """Strengths tried (strongest first) to recommend a default for the slider."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min_share > self.max_share:
            raise ValueError("adapter min_share must not exceed max_share")
        if not self.target_modules:
            raise ValueError("adapter target_modules must not be empty")
        return self


class GenerationConfig(_Frozen):
    """Sampling settings and the number of variants per request."""

    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    variants: int = Field(default=4, ge=1, le=32)
    constrained: bool = True
    """Mask tokens that would break the sequence grammar (bar counts, ranges, order)."""


# -------------------------------------------------------------------------- root


class BacklineConfig(_Frozen):
    """The whole behaviour profile."""

    name: str
    seed: int = 1337
    instruments: InstrumentsConfig
    velocity: VelocityConfig
    drums: DrumsConfig
    ingest: IngestConfig = IngestConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    originality: OriginalityConfig = OriginalityConfig()
    synth: SynthConfig = SynthConfig()
    model: ModelConfig
    train: TrainConfig = TrainConfig()
    adapter: AdapterConfig = AdapterConfig()
    generation: GenerationConfig = GenerationConfig()


_PATH_KEYS = (("drums", "output_map"),)


def load_config(path: Path | str | None = None) -> BacklineConfig:
    """Load a profile, following ``extends:`` chains. Defaults to ``BACKLINE_CONFIG``."""
    config_path = Path(path) if path is not None else PathSettings().config
    config_path = config_path.resolve()
    raw = _read_layered(config_path, seen=())
    _resolve_paths(raw, config_path.parent)
    try:
        return BacklineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"config {config_path} is invalid:\n{exc}") from exc


def _read_layered(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ConfigError(f"config extends itself: {chain}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a mapping at the top level")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _read_layered((path.parent / str(parent)).resolve(), (*seen, path))
    return deep_merge(base, data)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``: mappings merge recursively, other values replace."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(raw: dict[str, Any], base_dir: Path) -> None:
    for section, key in _PATH_KEYS:
        block = raw.get(section)
        if isinstance(block, dict) and isinstance(block.get(key), str):
            candidate = Path(block[key])
            block[key] = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
