"""Step 07 training data: reverberant rendering, the sharded cache, datasets, loss, training.

Everything runs on the tiny fake dataset of ``conftest.py`` (no downloads, CPU only).
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import json
from typing import Any

import numpy as np
import pytest
import torch

from soundspy.cli.main import main
from soundspy.config import WaveformerConfig
from soundspy.data import cache as C
from soundspy.data.mixtures import (
    ClipLibrary,
    RecipeError,
    iter_recipes,
    recipes_path,
    render_with_config,
    write_recipes,
)
from soundspy.data.prepare import load_activity, load_clip_index
from soundspy.data.reverb import Room, recipe_rirs, recipe_rng, sample_room, unit_energy

SR = 16_000
COUNTS = {"train": 11, "val": 5, "test": 4}


@pytest.fixture(scope="module")
def cache_env(shared_prepared_env: Any, tmp_path_factory):
    """Recipes for the three splits and a config whose neural extractor is tiny: 16 kHz model,
    4 recipes per shard, reverberation off (the simulation has its own test)."""
    config = shared_prepared_env.config
    index = load_clip_index(config.paths["processed"])
    activity = load_activity(config.paths["processed"])
    for split, count in COUNTS.items():
        write_recipes(index, activity, config, split, count=count)
    neural = config.extraction.neural_tse
    model = WaveformerConfig(sample_rate=SR, L=4, enc_dim=16, num_enc_layers=3, dec_dim=8,
                             num_dec_layers=1, dec_buf_len=3, dec_chunk_size=3, out_buf_len=2,
                             use_pos_enc=True)
    data = dataclasses.replace(neural.data, shard_size=4, min_free_gb=0.0,
                               reverb=dataclasses.replace(neural.data.reverb, prob=0.0))
    training = dataclasses.replace(
        neural.training, epochs=2, train_recipes=11, epoch_recipes=8, train_crop_s=1.0,
        batch_size=2, max_batch_size=2, effective_batch_size=4, num_workers=0, val_examples=5,
        warmup_epochs=0.5, amp=False, smoke_epochs=1, smoke_train_recipes=4,
        smoke_val_examples=2)
    neural = dataclasses.replace(neural, model=model, data=data, training=training,
                                 checkpoint=None)
    experiments = tmp_path_factory.mktemp("experiments")
    config = dataclasses.replace(
        config, paths={**config.paths, "experiments": experiments},
        extraction=dataclasses.replace(config.extraction, neural_tse=neural))
    return config


def recipes_of(config, split: str = "train"):
    return list(iter_recipes(recipes_path(config.paths["mixtures"], split)))


# --- reverberation ----------------------------------------------------------------------------


def test_reverberant_render_is_still_target_plus_rest(cache_env) -> None:
    config = cache_env
    library = ClipLibrary.from_config(config)
    recipe = next(r for r in recipes_of(config) if not r.empty_target and len(r.events) >= 2)
    dry = render_with_config(recipe, SR, library, config)
    rng = np.random.default_rng(0)
    decay = np.exp(-np.arange(SR // 4) / (0.05 * SR))
    rirs = [unit_energy(rng.standard_normal(SR // 4) * decay) for _ in recipe.events]
    wet = render_with_config(recipe, SR, library, config, rirs=rirs)
    for rendered in (dry, wet):
        np.testing.assert_allclose(rendered.mixture.samples,
                                   rendered.target.samples + rendered.rest.samples, atol=1e-6)
    assert wet.mixture.samples.shape == dry.mixture.samples.shape  # same length: tail is cut
    assert wet.meta["reverb"] is True and dry.meta["reverb"] is False
    assert not np.allclose(wet.target.samples, dry.target.samples, atol=1e-3)
    np.testing.assert_array_equal(wet.target_hot, dry.target_hot)
    # the target is the REVERBERANT target: it rings on after the dry one has stopped
    quiet = np.abs(dry.target.samples.ravel()) < 1e-6
    if quiet.any():
        assert np.abs(wet.target.samples.ravel()[quiet]).max() > 1e-4
    with pytest.raises(RecipeError, match="impulse responses"):
        render_with_config(recipe, SR, library, config, rirs=rirs[:-1])


def test_rooms_are_deterministic_per_recipe(config) -> None:
    cfg = config.extraction.neural_tse.data.reverb
    a = sample_room(recipe_rng(7, "train-000001"), cfg, 3)
    b = sample_room(recipe_rng(7, "train-000001"), cfg, 3)
    c = sample_room(recipe_rng(7, "train-000002"), cfg, 3)
    assert a == b and a != c and isinstance(a, Room) and len(a.sources) == 3
    assert cfg.rt60_s[0] <= a.rt60_s <= cfg.rt60_s[1]
    for (low, high), size, mic in zip(cfg.room_m, a.dims, a.mic, strict=True):
        assert low <= size <= high
        assert cfg.wall_margin_m <= mic <= size - cfg.wall_margin_m
    never = dataclasses.replace(cfg, prob=0.0)
    assert recipe_rirs("train-000001", 3, SR, never, 7) is None  # dry: nothing is simulated
    assert recipe_rirs("train-000001", 0, SR, cfg, 7) is None
    energy = float(np.sum(unit_energy(np.array([3.0, 4.0])) ** 2))
    assert energy == pytest.approx(1.0)
    # about `prob` of the recipes are reverberant (decided from the id alone)
    wet = sum(recipe_rng(7, f"train-{i:06d}").random() < cfg.prob for i in range(400))
    assert 150 < wet < 250


def test_simulated_impulse_responses() -> None:
    pytest.importorskip("pyroomacoustics")
    from soundspy.config import ReverbConfig
    from soundspy.data.reverb import simulate_rirs

    cfg = ReverbConfig(prob=1.0, rt60_s=(0.3, 0.3), room_m=((4.0, 4.0), (3.0, 3.0), (2.5, 2.5)),
                       wall_margin_m=0.5, max_order=6)
    first = recipe_rirs("r1", 2, SR, cfg, seed=1)
    again = recipe_rirs("r1", 2, SR, cfg, seed=1)
    assert first is not None and len(first) == 2
    for rir, same in zip(first, again, strict=True):
        np.testing.assert_array_equal(rir, same)  # rebuilt identically
        assert rir.dtype == np.float32 and float(np.sum(rir.astype(np.float64) ** 2)) == \
            pytest.approx(1.0, abs=1e-4)
    # the earliest direct sound is at the start: no common delay is added to the events
    assert min(int(np.argmax(np.abs(r))) for r in first) <= 4
    room = sample_room(recipe_rng(1, "r1"), cfg, 2)
    assert len(simulate_rirs(room, SR, 2)[0]) < len(first[0])  # fewer reflections: shorter


# --- the cache --------------------------------------------------------------------------------


def test_cache_is_written_read_and_resumed(cache_env, monkeypatch) -> None:
    config = cache_env
    logs: list[str] = []
    summary = C.build_extraction_cache(config, "train", workers=1, progress=logs.append)
    directory = C.split_dir(config, SR, "train")
    assert summary.directory == directory and directory.is_relative_to(config.paths["processed"])
    assert (summary.recipes, summary.shards, summary.built) == (11, 3, 3)
    samples = round(config.data.mixtures.duration_s * SR)
    assert summary.bytes >= 11 * 2 * samples * 2  # float16
    assert sorted(p.name for p in directory.iterdir()) == [
        "index.json", "shard_000.meta.npz", "shard_000.npy", "shard_001.meta.npz",
        "shard_001.npy", "shard_002.meta.npz", "shard_002.npy"]
    assert np.load(directory / "shard_002.npy", mmap_mode="r").shape == (3, 2, samples)
    assert any("3 of 3 shards" in m for m in logs)

    cache = C.ExtractionCache(directory)
    recipes = recipes_of(config)
    library = ClipLibrary.from_config(config)
    assert len(cache) == 11 and cache.recipe_ids == [r.id for r in recipes]
    assert cache.sample_rate == SR and cache.samples == samples
    assert cache.labels == list(config.labels)
    for i in (0, 5, 10):  # first, across a shard border, last
        rendered = render_with_config(recipes[i], SR, library, config)
        pair = cache.pair(i)
        assert pair.dtype == np.float16 and pair.shape == (2, samples)
        assert not pair.flags.writeable  # a view into the memory-mapped file
        np.testing.assert_allclose(pair[0], rendered.mixture.samples.ravel(), atol=2e-3)
        np.testing.assert_allclose(pair[1], rendered.target.samples.ravel(), atol=2e-3)
        np.testing.assert_array_equal(cache.target_hot[i], rendered.target_hot)
        np.testing.assert_array_equal(cache.present_hot[i], rendered.present)
        assert bool(cache.empty[i]) == recipes[i].empty_target and not cache.reverb[i]
        if not cache.empty[i]:
            start, end = int(cache.target_start[i]), int(cache.target_end[i])
            assert 0 <= start < end <= samples
            assert np.abs(pair[1][start:end]).max() > 0
            assert not pair[1][:start].any() or np.abs(pair[1][:start]).max() < \
                1e-3 * np.abs(pair[1]).max() + 1e-4
    with pytest.raises(IndexError):
        cache.pair(11)
    import pickle

    assert len(pickle.loads(pickle.dumps(cache))) == 11  # goes to DataLoader workers

    # idempotent: everything is there, nothing is rendered
    rendered_shards: list[int] = []
    original = C._render_shard
    monkeypatch.setattr(C, "_render_shard",
                        lambda *a: rendered_shards.append(a[1]) or original(*a))
    again = C.build_extraction_cache(config, "train", workers=1, progress=logs.append)
    assert again.built == 0 and not rendered_shards and again.bytes == summary.bytes

    # resumable: a missing shard and a half-written one are rendered, the others are kept
    reference = np.array(cache.pair(5))
    del pair  # Windows cannot delete a file that is still memory-mapped
    cache.close()
    gc.collect()
    before = (directory / "shard_000.npy").stat().st_mtime_ns
    (directory / "shard_001.npy").unlink()
    (directory / "shard_009.npy.part").write_bytes(b"half")
    with pytest.raises(C.CacheError, match="incomplete"):
        C.ExtractionCache(directory)
    resumed = C.build_extraction_cache(config, "train", workers=1, progress=logs.append)
    assert resumed.built == 1 and rendered_shards == [1]
    assert not list(directory.glob("*.part"))
    assert (directory / "shard_000.npy").stat().st_mtime_ns == before
    np.testing.assert_array_equal(C.ExtractionCache(directory).pair(5), reference)


def test_cache_refuses_other_settings_and_a_full_disk(cache_env, monkeypatch) -> None:
    config = cache_env
    C.build_extraction_cache(config, "val", workers=1, progress=lambda m: None)
    neural = config.extraction.neural_tse
    changed = dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, neural_tse=dataclasses.replace(neural, data=dataclasses.replace(
            neural.data, seed=neural.data.seed + 1))))
    with pytest.raises(C.CacheError, match="other settings"):
        C.build_extraction_cache(changed, "val", workers=1, progress=lambda m: None)
    rebuilt = C.build_extraction_cache(changed, "val", workers=1, force=True,
                                       progress=lambda m: None)
    assert rebuilt.built == rebuilt.shards == 2
    C.build_extraction_cache(config, "val", workers=1, force=True, progress=lambda m: None)

    # free space is checked before anything is written
    monkeypatch.setattr(C, "free_bytes", lambda path: 100_000)
    with pytest.raises(C.CacheError, match="not enough space"):
        C.build_extraction_cache(config, "test", workers=1, progress=lambda m: None)
    assert not list(C.split_dir(config, SR, "test").glob("shard_*"))
    needy = dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, neural_tse=dataclasses.replace(neural, data=dataclasses.replace(
            neural.data, min_free_gb=50.0))))
    monkeypatch.setattr(C, "free_bytes", lambda path: 40_000_000_000)
    with pytest.raises(C.CacheError, match="50 GB must stay free"):
        C.build_extraction_cache(needy, "test", workers=1, progress=lambda m: None)

    with pytest.raises(C.CacheError, match="soundspy data mix"):
        C.build_extraction_cache(dataclasses.replace(
            config, paths={**config.paths, "mixtures": config.paths["mixtures"] / "none"}),
            "train", progress=lambda m: None)
    with pytest.raises(C.CacheError, match="soundspy data cache"):
        C.ExtractionCache(config.paths["processed"] / "cache" / "nothing")


def test_cache_in_parallel_and_from_the_cli(cache_env, monkeypatch, capsys) -> None:
    config = cache_env
    monkeypatch.undo()
    serial = C.build_extraction_cache(config, "test", workers=1, progress=lambda m: None)
    reference = np.array(C.ExtractionCache(serial.directory).pair(3))
    parallel = C.build_extraction_cache(config, "test", workers=2, limit=3, force=True,
                                        progress=lambda m: None)
    assert parallel.recipes == 3 and parallel.built == 1
    C.build_extraction_cache(config, "test", workers=2, force=True, progress=lambda m: None)
    np.testing.assert_array_equal(C.ExtractionCache(serial.directory).pair(3), reference)

    import soundspy.config

    monkeypatch.setattr(soundspy.config, "load_config", lambda *a, **k: config)
    assert main(["data", "cache", "--task", "extraction", "--split", "all", "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "train" in out and "val" in out and "test" in out and "GB" in out


# --- datasets and the loss --------------------------------------------------------------------


def test_training_crops_and_validation_items(cache_env) -> None:
    from soundspy.training.extraction_data import (
        ExtractionEvalSet,
        ExtractionTrainSet,
        collate,
        crop_start,
    )

    config = cache_env
    C.build_extraction_cache(config, "train", workers=1, progress=lambda m: None)
    cache = C.ExtractionCache(C.split_dir(config, SR, "train"))
    train = ExtractionTrainSet(cache, range(len(cache)), crop_s=1.0, crop_target_prob=1.0,
                               seed=3)
    assert len(train) == 11 and train.crop == SR
    mixture, target, label = train[(2, 0)]
    assert mixture.shape == target.shape == (SR,) and mixture.dtype == torch.float32
    assert label.shape == (len(config.labels),)
    same = train[(2, 0)]
    other = train[(2, 1)]
    torch.testing.assert_close(mixture, same[0])  # (index, epoch) fixes the crop
    assert not torch.equal(mixture, other[0])  # another epoch: another crop
    # with crop_target_prob = 1 a crop of an example with a target always contains some of it
    for i in range(len(cache)):
        if not cache.empty[i]:
            for epoch in range(4):
                assert train[(i, epoch)][1].abs().max() > 0
    batch = collate([train[(0, 0)], train[(1, 0)]])
    assert batch[0].shape == (2, SR) and batch[2].shape == (2, len(config.labels))

    val = ExtractionEvalSet(cache, [4, 0])
    mixture, target, label = val[0]
    assert mixture.shape == (cache.samples,)
    np.testing.assert_allclose(mixture.numpy(), np.asarray(cache.pair(4)[0], dtype=np.float32))

    rng = np.random.default_rng(0)
    for _ in range(200):  # the crop overlaps the target's stretch by at least 0.25 s
        start = crop_start(rng, 96_000, 16_000, 40_000, 60_000, SR, 1.0)
        assert start + 16_000 >= 40_000 + 4_000 and start <= 60_000 - 4_000
    assert crop_start(rng, 16_000, 16_000, 0, 100, SR, 1.0) == 0  # nothing to choose
    assert 0 <= crop_start(rng, 96_000, 16_000, 0, 0, SR, 1.0) <= 80_000  # no target: anywhere


def test_loss_terms_and_the_floor(config) -> None:
    from soundspy.training.extraction_data import (
        extraction_loss,
        is_silent,
        si_snr_db,
        snr_db,
    )

    cfg = dataclasses.replace(config.extraction.neural_tse.training.loss, snr_weight=0.9,
                              si_snr_weight=0.1, empty_weight=0.5, empty_floor_db=-40.0)
    torch.manual_seed(0)
    target = torch.randn(1, 8000) * 0.1
    noise = torch.randn(1, 8000) * 0.1
    mixture = target + noise

    # 1. the two ratios: known values
    estimate = target + 0.1 * noise  # error 20 dB below the target
    assert float(snr_db(estimate, target)) == pytest.approx(20.0, abs=0.2)
    assert float(si_snr_db(estimate, target)) == pytest.approx(20.0, abs=0.2)
    assert float(snr_db(0.5 * target, target)) == pytest.approx(6.02, abs=0.01)  # level matters
    assert float(si_snr_db(0.5 * target, target)) > 60.0  # ... and here it does not
    loss, parts = extraction_loss(estimate, target, mixture, cfg)
    expected = -(0.9 * snr_db(estimate, target) + 0.1 * si_snr_db(estimate, target))
    assert float(loss) == pytest.approx(float(expected), abs=1e-4)
    assert parts["silent"] == 0.0 and parts["quality"] == pytest.approx(float(expected), abs=1e-4)
    better, _ = extraction_loss(target + 0.01 * noise, target, mixture, cfg)
    assert float(better) < float(loss)  # a better estimate has a lower loss

    # 2. a silent target: the output's energy relative to the mixture, in dB, floored
    silent = torch.zeros(1, 8000)
    assert bool(is_silent(silent)) and not bool(is_silent(target))
    for gain, expected_db in ((1.0, 0.0), (0.1, -20.0), (0.01, -40.0), (1e-4, -40.0)):
        loss, parts = extraction_loss(gain * mixture, silent, mixture, cfg)
        assert parts["leak_db"] == pytest.approx(expected_db, abs=0.01)
        assert float(loss) == pytest.approx(0.5 * expected_db, abs=0.01)  # empty_weight
        assert parts["silent"] == 1.0
    loss, _ = extraction_loss(torch.zeros(1, 8000), silent, mixture, cfg)
    assert torch.isfinite(loss) and float(loss) == pytest.approx(-20.0, abs=0.01)  # the floor

    # 3. a batch with both kinds: the weighted mean over the examples
    est = torch.cat([estimate, 0.1 * mixture])
    tgt = torch.cat([target, silent])
    mix = torch.cat([mixture, mixture])
    loss, parts = extraction_loss(est, tgt, mix, cfg)
    assert float(loss) == pytest.approx((float(expected) + 0.5 * -20.0) / 2, abs=1e-3)
    assert parts["silent"] == 0.5

    # 4. gradients: none below the floor, some above it, and never NaN
    quiet = (1e-4 * mixture).requires_grad_()
    extraction_loss(quiet, silent, mixture, cfg)[0].backward()
    assert not quiet.grad.any()
    loud = (0.1 * mixture).requires_grad_()
    extraction_loss(loud, silent, mixture, cfg)[0].backward()
    assert loud.grad.abs().sum() > 0
    zero = torch.zeros(2, 8000, requires_grad=True)
    extraction_loss(zero, tgt, mix, cfg)[0].backward()
    assert torch.isfinite(zero.grad).all()


# --- training ---------------------------------------------------------------------------------


def test_training_runs_resumes_and_writes_the_experiment(cache_env, monkeypatch, capsys
                                                         ) -> None:
    from soundspy.extraction.neural_model import load_checkpoint
    from soundspy.training.train import TrainingError
    from soundspy.training.train_extraction import (
        ExtractionTrainOptions,
        RotatingSampler,
        train_extraction,
    )

    config = cache_env
    for split in ("train", "val"):
        C.build_extraction_cache(config, split, workers=1, progress=lambda m: None)
    logs: list[str] = []
    opts = ExtractionTrainOptions(name="tse_test", device="cpu", allow_cpu=True, workers=0,
                                  pretrained=False, epochs=1)
    result = train_extraction(config, opts, log=logs.append)
    exp = config.paths["experiments"] / "tse_test"
    assert result.epochs_run == 1 and result.best_epoch == 1 and not result.stopped_early
    assert (result.batch_size, result.accumulate) == (2, 2)  # 2 x 2 = effective 4
    for name in ("config.json", "log.csv", "metrics.json", "checkpoints/best.pt",
                 "checkpoints/last.pt", "plots/training_curves.png"):
        assert (exp / name).is_file(), name
    with (exp / "log.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and float(rows[0]["val_si_sdri"]) == pytest.approx(
        result.best_si_sdri, abs=1e-3)
    assert {"train_loss", "val_snri", "lr_backbone", "lr_label", "batch_size"} <= set(rows[0])
    saved = json.loads((exp / "config.json").read_text("utf-8"))
    assert saved["task"] == "extraction" and saved["labels"] == list(config.labels)
    assert saved["epoch_recipes"] == 8 and saved["train_recipes"] == 11
    net, model, labels, extra = load_checkpoint(exp / "checkpoints" / "best.pt")
    assert model == config.extraction.neural_tse.model and labels == list(config.labels)
    assert extra["epoch"] == 1
    assert any("random weights" in m for m in logs) and any("8 crops" in m for m in logs)

    with pytest.raises(TrainingError, match="--resume"):
        train_extraction(config, opts, log=logs.append)  # never overwrites a run silently
    resumed = train_extraction(config, dataclasses.replace(opts, resume=True, epochs=2),
                               log=logs.append)
    assert resumed.epochs_run == 1 and any("resuming tse_test after epoch 1" in m for m in logs)
    with (exp / "log.csv").open(encoding="utf-8") as fh:
        assert [r["epoch"] for r in csv.DictReader(fh)] == ["1", "2"]
    with pytest.raises(TrainingError, match="nothing to resume"):
        train_extraction(config, dataclasses.replace(opts, name="other", resume=True),
                         log=logs.append)

    # the rotating sampler: epochs are short, but every recipe is used before any repeats
    sampler = RotatingSampler(20, 10, seed=1)
    seen: list[int] = []
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        items = list(sampler)
        assert len(items) == len(sampler) == 10 and {e for _, e in items} == {epoch}
        seen += [i for i, _ in items]
    assert sorted(seen) == list(range(20))
    sampler.set_epoch(2)
    assert [i for i, _ in sampler] != seen[:10]  # a new shuffle for the next pass

    # the CLI: smoke mode (the published weights are not downloaded in tests)
    import soundspy.config
    import soundspy.training.train_extraction as module

    monkeypatch.setattr(soundspy.config, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(module, "ensure_pretrained", lambda *a, **k: None)
    assert main(["train", "extraction", "--name", "tse_cli", "--smoke", "--device", "cpu",
                 "--workers", "0"]) == 0
    out = capsys.readouterr().out
    assert "best epoch 1" in out and "4 crops" in out  # smoke: 4 recipes, 1 epoch
    assert (config.paths["experiments"] / "tse_cli" / "checkpoints" / "best.pt").is_file()
    assert main(["train", "extraction", "--name", "tse_cli", "--smoke", "--device", "cpu"]) == 1
    assert "--resume" in capsys.readouterr().out
