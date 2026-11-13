# SPDX-FileCopyrightText: 2026 Ithinkimbulletproof
# SPDX-License-Identifier: AGPL-3.0-only
"""LoRA adapters: mixed sampling, caching, strength and generation (tiny, on CPU)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from backline.adapters import lora
from backline.adapters.cache import AdapterStore, adapter_key, file_fingerprint
from backline.adapters.train import (
    AdapterError,
    load_for_generation,
    reference_share,
    train_adapter,
)
from backline.config import BacklineConfig
from backline.dataset.build import build_dataset
from backline.formats.guitarpro import write_gp5
from backline.ingest.catalog import Catalog
from backline.ingest.pipeline import LibraryPaths, run_ingest
from backline.model.service import generate_variants, prepare_riff
from backline.model.train import Trainer, load_model
from backline.originality.build import build_originality_index, slice_score
from backline.originality.index import OriginalityIndex
from backline.synth import generate_corpus
from backline.tokenizer.vocab import Vocab


@pytest.fixture(scope="module")
def setup(
    tmp_path_factory: pytest.TempPathFactory, tiny_config: BacklineConfig
) -> dict[str, object]:
    root = tmp_path_factory.mktemp("adapters")
    tabs = root / "tabs"
    tabs.mkdir()
    songs = generate_corpus(tiny_config, n_songs=20, seed=55)
    for index, song in enumerate(songs):
        write_gp5(song.score, tabs / f"{index:02d} {song.score.artist} - {song.score.title}.gp5")
    library = LibraryPaths(root / "catalog.sqlite", root / "scores", root / "reports")
    run_ingest(tabs, tiny_config, library, workers=1)
    build_dataset(tiny_config, library, root / "dataset")
    index = build_originality_index(tiny_config, library, root / "index")
    Trainer(tiny_config, root / "dataset", root / "ckpt", device="cpu").train(resume=False)
    with Catalog(library.catalog) as catalog:
        ids = [s.id for s in catalog.songs("ok") if s.split == "train"]
    return {
        "root": root,
        "dataset": root / "dataset",
        "checkpoint": root / "ckpt" / "best.pt",
        "index": index,
        "ids": ids,
        "riff": songs[0].score,
    }


def test_reference_share_follows_epochs(tiny_config: BacklineConfig) -> None:
    cfg = tiny_config.adapter
    small = reference_share(tiny_config, 1)
    large = reference_share(tiny_config, 10_000)
    assert small == cfg.min_share
    assert large == cfg.max_share
    fixed = tiny_config.model_copy(
        update={"adapter": cfg.model_copy(update={"reference_share": 0.5})}
    )
    assert reference_share(fixed, 1) == 0.5


def test_key_depends_on_references_settings_and_base(tiny_config: BacklineConfig) -> None:
    cfg = tiny_config.adapter
    key = adapter_key(["b", "a"], cfg, "base1")
    assert key == adapter_key(["a", "b", "a"], cfg, "base1")
    assert key != adapter_key(["a"], cfg, "base1")
    assert key != adapter_key(["a", "b"], cfg, "base2")
    assert key != adapter_key(["a", "b"], cfg.model_copy(update={"rank": cfg.rank + 1}), "base1")


def test_fingerprint_is_memoised(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"0123456789" * 1000)
    first = file_fingerprint(path)
    assert (tmp_path / "weights.pt.sha256").is_file()
    assert file_fingerprint(path) == first
    path.write_bytes(b"other")
    assert file_fingerprint(path) != first


def test_train_cache_and_strength(
    setup: dict[str, object], tiny_config: BacklineConfig, tmp_path: Path
) -> None:
    store = AdapterStore(tmp_path / "adapters")
    refs = setup["ids"][:2]  # type: ignore[index]
    seen: list[int] = []
    result = train_adapter(
        tiny_config,
        setup["checkpoint"],  # type: ignore[arg-type]
        setup["dataset"],  # type: ignore[arg-type]
        refs,
        store,
        index=setup["index"],  # type: ignore[arg-type]
        device="cpu",
        progress=lambda step, total, record: seen.append(step),
    )
    assert not result.cached
    assert seen[0] == 1
    assert result.info.references == sorted(refs)
    assert 0 < result.info.reference_share <= 1
    assert result.info.recommended_strength in tiny_config.adapter.strengths
    assert (result.path / "adapter_model.safetensors").is_file()

    started = time.perf_counter()
    again = train_adapter(
        tiny_config,
        setup["checkpoint"],
        setup["dataset"],
        list(reversed(refs)),
        store,
        device="cpu",  # type: ignore[arg-type]
    )
    assert again.cached
    assert again.info.key == result.info.key
    assert time.perf_counter() - started < 5
    assert [info.key for info in store.list()] == [result.info.key]

    model, info = load_for_generation(tiny_config, setup["checkpoint"], refs, store, device="cpu")  # type: ignore[arg-type]
    assert info is not None
    base, _ = load_model(setup["checkpoint"], device="cpu")  # type: ignore[arg-type]
    tokens = torch.randint(0, len(Vocab.from_config(tiny_config)), (1, 12))
    with torch.no_grad(), lora.strength(model, 0.0):
        at_zero, _ = model(tokens)
    expected, _ = base(tokens)
    assert torch.allclose(at_zero, expected, atol=1e-5)

    riff_score = slice_score(setup["riff"], 0, 2)  # type: ignore[arg-type]
    riff_score.tracks = [riff_score.tracks[0]]
    riff = prepare_riff(riff_score, tiny_config)
    generated = generate_variants(
        model, Vocab.from_config(tiny_config), tiny_config, riff, strength=0.5, seed=4
    )
    assert len(generated.variants) == tiny_config.generation.variants
    assert all(v.metrics.valid for v in generated.variants)


def test_unknown_references_are_rejected(
    setup: dict[str, object], tiny_config: BacklineConfig, tmp_path: Path
) -> None:
    store = AdapterStore(tmp_path / "adapters")
    with pytest.raises(AdapterError, match="no usable segments"):
        train_adapter(
            tiny_config, setup["checkpoint"], setup["dataset"], ["missing"], store, device="cpu"
        )  # type: ignore[arg-type]
    with pytest.raises(AdapterError, match="at least one"):
        train_adapter(tiny_config, setup["checkpoint"], setup["dataset"], [], store, device="cpu")  # type: ignore[arg-type]
    base, info = load_for_generation(
        tiny_config, setup["checkpoint"], ["missing"], store, device="cpu"
    )  # type: ignore[arg-type]
    assert info is None
    assert not isinstance(base, OriginalityIndex)
