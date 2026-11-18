"""Step 07b data: LibriSpeech collector, the recipe sampler's new rules, the room bank, the clip
cache and dynamic mixing. Everything on the tiny fake datasets of ``conftest.py``."""

from __future__ import annotations

import dataclasses
import pickle
import random
from collections import Counter
from typing import Any

import numpy as np
import pytest

from soundspy.data.collectors import CollectorError
from soundspy.data.collectors import librispeech as libri
from soundspy.data.mixtures import (
    ClipLibrary,
    RecipeSampler,
    RecipeSpec,
    generate_recipes,
    lowpass,
    render,
)
from soundspy.data.prepare import load_clip_index

SR = 16_000


def library_of(env: Any) -> ClipLibrary:
    return ClipLibrary.from_config(env.config)


def mixing(env: Any, **changes) -> Any:
    return dataclasses.replace(env.config.data.mixtures, **changes)


# --- LibriSpeech ------------------------------------------------------------------------------


def test_librispeech_rows_use_the_official_subsets_and_speakers(shared_libri_env, kit) -> None:
    index = load_clip_index(shared_libri_env.config.paths["processed"])
    rows = index[index["dataset"] == "librispeech"]
    expected = sum(n for *_, n in kit.LIBRI)
    assert len(rows) == expected
    assert set(rows["role"]) == {"fg:speech"} and set(rows["license"]) == {"CC BY 4.0"}
    assert set(rows["sample_rate"]) == {SR}
    by_split = Counter(rows["split"])
    assert by_split == {"train": 5, "val": 2, "test": 2}  # train-clean-100 / dev / test
    speakers = rows.groupby("group")["split"].nunique()
    assert set(speakers.index) == {"speaker:19", "speaker:26", "speaker:84", "speaker:61"}
    assert (speakers == 1).all()  # a speaker never spans two splits
    one = rows.iloc[0]
    assert one["path"].startswith("librispeech://train-clean-100/19/198/")
    library = library_of(shared_libri_env)
    segment = library.load_segment(one["clip_id"], 0.5, 1.0, 48_000)
    assert segment.shape == (48_000,) and np.abs(segment).max() > 0.05


def test_librispeech_collector_errors(shared_libri_env, tmp_path) -> None:
    paths = {**shared_libri_env.config.paths, "external": tmp_path}
    with pytest.raises(CollectorError, match="not found"):
        libri.collect(paths, {"train": "train-clean-100"}, "speech", ["/m/sp"])
    (tmp_path / "librispeech" / "LibriSpeech" / "dev-clean").mkdir(parents=True)
    with pytest.raises(CollectorError, match="no FLAC"):
        libri.collect(paths, {"val": "dev-clean"}, "speech", ["/m/sp"])


def test_the_frozen_recipes_do_not_change_when_librispeech_is_added(
        shared_prepared_env, shared_libri_env) -> None:
    """With the file settings (LibriSpeech share 0) the same seed gives the same recipes, so
    the val / test files of steps 04-07 stay valid."""
    old, new = shared_prepared_env.config, shared_libri_env.config
    for split in ("train", "val", "test"):
        before = generate_recipes(library_of(shared_prepared_env), old.labels,
                                  old.data.mixtures, split, 40, seed=5)
        after = generate_recipes(library_of(shared_libri_env), new.labels,
                                 new.data.mixtures, split, 40, seed=5)
        assert before == after
        assert not any(e.clip_id.startswith("librispeech:") for r in after for e in r.events)


# --- the sampler ------------------------------------------------------------------------------


def _draw(sampler: RecipeSampler, n: int, seed: int = 0) -> list:
    return [sampler.sample(random.Random(f"{seed}/{i}"), f"t-{i}") for i in range(n)]


def test_empty_and_partial_target_rates(shared_libri_env) -> None:
    env = shared_libri_env
    mix = mixing(env, empty_target_prob=0.3, partial_target_prob=0.4)
    recipes = _draw(RecipeSampler(library_of(env), env.config.labels, mix, "train"), 1500)
    empty = [r for r in recipes if r.empty_target]
    assert 0.25 < len(empty) / len(recipes) < 0.35
    assert all(r.switch_s is None for r in empty)
    partial = [r for r in recipes if r.switch_s is not None]
    assert 0.33 < len(partial) / (len(recipes) - len(empty)) < 0.47
    for r in partial:  # every target event on one side of the switch
        targets = [e for e in r.events if e.role == "fg" and e.cls in r.target]
        after = all(e.onset_s >= r.switch_s - 1e-3 for e in targets)
        before = all(e.onset_s + e.dur_s <= r.switch_s + 1e-3 for e in targets)
        assert after or before, r
        assert mix.partial_switch[0] * mix.duration_s - 1e-3 <= r.switch_s \
            <= mix.partial_switch[1] * mix.duration_s + 1e-3
    none = mixing(env, empty_target_prob=0.3)  # the file settings: never partial
    assert all(r.switch_s is None
               for r in _draw(RecipeSampler(library_of(env), env.config.labels, none, "train"),
                              200))


def test_hard_negatives_put_the_confusable_class_in(shared_libri_env) -> None:
    env = shared_libri_env

    def music_share(prob: float) -> float:
        mix = mixing(env, empty_target_prob=1.0, hard_negative_prob=prob,
                     hard_negative_pairs={"speech": ("music",)})
        recipes = [r for r in _draw(RecipeSampler(library_of(env), env.config.labels, mix,
                                                  "train"), 800) if r.target == ("speech",)]
        assert recipes and all(r.empty_target for r in recipes)
        return sum("music" in r.present for r in recipes) / len(recipes)

    assert music_share(1.0) == 1.0
    assert music_share(0.0) < 0.7  # without the bias music is only sometimes there
    with pytest.raises(Exception, match="unknown"):
        RecipeSampler(library_of(env), env.config.labels,
                      mixing(env, hard_negative_pairs={"speech": ("violin",)}), "train")


def test_speech_source_mix(shared_libri_env) -> None:
    env = shared_libri_env

    def libri_share(share: float) -> float:
        mix = mixing(env, source_mix={"speech": {"librispeech": share}})
        sampler = RecipeSampler(library_of(env), env.config.labels, mix, "train")
        rng = random.Random(1)
        picks = [sampler.pick_clip(rng, "speech") for _ in range(2000)]
        return sum(p.startswith("librispeech:") for p in picks) / len(picks)

    assert 0.45 < libri_share(0.5) < 0.55  # 5 LibriSpeech vs 2 FSD50K clips: still half/half
    assert libri_share(0.0) == 0.0 and libri_share(1.0) == 1.0


def test_a_spec_pins_classes_selection_and_clip_length(shared_libri_env) -> None:
    env = shared_libri_env
    sampler = RecipeSampler(library_of(env), env.config.labels, env.config.data.mixtures, "train")
    rng = random.Random(3)
    spec = RecipeSpec(present=("music",), target=("speech",), n_background=(0, 0),
                      fill=frozenset({"music"}), min_clip_s={"music": 2.0})
    for i in range(20):
        r = sampler.sample(rng, f"b-{i}", spec)
        assert r.present == ("music",) and r.target == ("speech",) and r.empty_target
        assert [e.role for e in r.events] == ["fg"]
        assert sampler.library.info(r.events[0].clip_id).duration_s >= 2.0  # fsd50k:104, 3 s
    # no clip is long enough (the test split's only music clip has 1.2 s): any clip is used
    test = RecipeSampler(library_of(env), env.config.labels, env.config.data.mixtures, "test")
    r = test.sample(rng, "b-y", spec)
    assert r.events[0].clip_id == "fsd50k:302"
    both = RecipeSpec(present=("speech", "music"), target=("music",))
    r = sampler.sample(rng, "b-x", both)
    assert set(r.present) == {"speech", "music"} and r.target == ("music",)
    assert not r.empty_target


# --- rendering: the low-pass augmentation -----------------------------------------------------


def test_lowpass_filter() -> None:
    x = np.random.default_rng(0).standard_normal(SR * 2)
    y = lowpass(x, 1000.0, SR)
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    high = spectrum[freqs > 3000].sum() / spectrum[freqs < 800].sum()
    assert high < 1e-4  # 6th order: > 40 dB down two octaves above the cut-off
    assert lowpass(x, 0.46 * SR, SR) is x  # nothing to do near Nyquist


def test_lowpass_is_applied_to_the_event_so_mixture_and_target_agree(shared_libri_env) -> None:
    env = shared_libri_env
    library = library_of(env)
    sampler = RecipeSampler(library, env.config.labels, env.config.data.mixtures, "train")
    recipe = sampler.sample(random.Random(9), "lp", RecipeSpec(present=("speech", "music"),
                                                               target=("speech",)))
    mix = env.config.data.mixtures
    kw = dict(reference_rms_dbfs=mix.reference_rms_dbfs, peak_limit=mix.peak_limit)
    plain = render(recipe, SR, library, env.config.labels, **kw)
    cutoffs = [1000.0 if e.cls == "speech" else None for e in recipe.events]
    filtered = render(recipe, SR, library, env.config.labels, lowpass_hz=cutoffs, **kw)
    for out in (plain, filtered):
        np.testing.assert_allclose(out.mixture.samples, out.target.samples + out.rest.samples,
                                   atol=1e-6)

    def high_share(x: np.ndarray) -> float:
        """High-frequency share inside the event (its abrupt cut-off end is broadband with or
        without the filter, so only the middle half of the event is measured)."""
        x = x.ravel()
        on = np.flatnonzero(np.abs(x) > 1e-6)
        a, b = on[0], on[-1]
        x = x[a + (b - a) // 4:b - (b - a) // 4]
        spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        freqs = np.fft.rfftfreq(x.size, 1 / SR)
        return spectrum[freqs > 4000].sum() / (spectrum.sum() + 1e-30)

    assert high_share(filtered.target.samples) < 1e-2 * high_share(plain.target.samples)
    with pytest.raises(Exception, match="cut-offs"):
        render(recipe, SR, library, env.config.labels, lowpass_hz=cutoffs[:-1], **kw)


# --- the room bank ----------------------------------------------------------------------------


def _tiny_bank(path, rooms: int = 3, positions: int = 4) -> list[list[np.ndarray]]:
    from soundspy.data.reverb import unit_energy, write_rir_bank

    rng = np.random.default_rng(0)
    rirs = [[unit_energy(rng.standard_normal(200 + 50 * p) * np.exp(-np.arange(200 + 50 * p)
                                                                      / 60.0))
             for p in range(positions)] for _ in range(rooms)]
    write_rir_bank(path, rirs, [(4.0, 3.0, 2.5, 0.4)] * rooms, {"sample_rate": SR})
    return rirs


def test_rir_bank_round_trip_and_draws(tmp_path) -> None:
    from soundspy.data.reverb import RirBank, RirBankError

    path = tmp_path / "rir_bank_16000.npz"
    rirs = _tiny_bank(path)
    bank = RirBank(path)
    assert (bank.rooms, bank.positions, bank.sample_rate) == (3, 4, SR)
    np.testing.assert_allclose(bank.rir(1, 2), rirs[1][2].astype(np.float16), atol=0)
    drawn = bank.draw(np.random.default_rng(5), 3)
    again = bank.draw(np.random.default_rng(5), 3)
    assert all(np.array_equal(a, b) for a, b in zip(drawn, again, strict=True))
    lengths = [len(r) for r in drawn]
    assert len(set(lengths)) == 3  # three different positions of one room
    assert len(bank.draw(np.random.default_rng(1), 6)) == 6  # more events than positions
    clone = pickle.loads(pickle.dumps(bank))  # DataLoader workers reopen the memory map
    assert clone._data is None and np.array_equal(clone.rir(0, 0), bank.rir(0, 0))
    with pytest.raises(IndexError):
        bank.rir(3, 0)
    with pytest.raises(RirBankError, match="no room bank"):
        RirBank(tmp_path / "missing.npz")


def test_rir_bank_simulation(tmp_path, config) -> None:
    pytest.importorskip("pyroomacoustics")
    from soundspy.data.reverb import RirBank, RirBankError, build_rir_bank, trim_tail

    cfg = dataclasses.replace(config.extraction.neural_tse.data.reverb, max_order=4)
    path = tmp_path / "bank.npz"
    info = build_rir_bank(cfg, SR, path, rooms=2, positions=2, seed=1, workers=1,
                          progress=lambda m: None)
    assert info["built"] and RirBank(path).rooms == 2
    assert not build_rir_bank(cfg, SR, path, rooms=2, positions=2, seed=1, workers=1,
                              progress=lambda m: None)["built"]  # same settings: nothing to do
    with pytest.raises(RirBankError, match="--force"):
        build_rir_bank(cfg, SR, path, rooms=3, positions=2, seed=1, workers=1,
                       progress=lambda m: None)
    decay = np.exp(-np.arange(SR) / (0.05 * SR))
    trimmed = trim_tail(decay, -60.0)
    assert len(trimmed) < SR // 2 and float(np.sum(trimmed.astype(np.float64) ** 2)) == \
        pytest.approx(1.0, rel=1e-4)


# --- the clip cache ---------------------------------------------------------------------------


def test_clip_cache_converts_once_and_is_used(shared_libri_env) -> None:
    from soundspy.data.clip_cache import build_clip_cache, clip_cache_dir

    config = shared_libri_env.config
    first = build_clip_cache(config, "librispeech", sample_rate=44_100, workers=1,
                             progress=lambda m: None)
    assert first.clips == first.built == 5  # the train split only
    again = build_clip_cache(config, "librispeech", sample_rate=44_100, workers=1,
                             progress=lambda m: None)
    assert again.built == 0 and again.bytes == first.bytes  # resumable / idempotent
    assert (clip_cache_dir(config, 44_100, "librispeech") / "index.json").is_file()
    library = library_of(shared_libri_env)
    clip = next(c for c in library.clips.values()
                if c.clip_id.startswith("librispeech:") and c.split == "train")
    cached = library.cached_file(clip, 44_100)
    assert cached is not None and cached.suffix == ".wav"
    assert library.cached_file(clip, SR) is None  # the native rate needs no copy
    fast = library.load_segment(clip.clip_id, 0.5, 1.0, 44_100)
    uncached = ClipLibrary(load_clip_index(config.paths["processed"]), library.activity,
                           config.paths, threshold_db=library.threshold_db)
    slow = uncached.load_segment(clip.clip_id, 0.5, 1.0, 44_100)
    assert fast.shape == slow.shape == (44_100,)
    np.testing.assert_allclose(fast[200:-200], slow[200:-200], atol=2e-3)  # 16-bit, edges


# --- dynamic mixing ---------------------------------------------------------------------------


def test_dynamic_examples(shared_libri_env, tmp_path) -> None:
    from soundspy.data.dynamic import DynamicMixer
    from soundspy.data.reverb import RirBank

    env = shared_libri_env
    _tiny_bank(tmp_path / "bank.npz")
    mix = mixing(env, empty_target_prob=0.2, partial_target_prob=0.0, lowpass_prob=0.5,
                 source_mix={"speech": {"librispeech": 0.5}})

    def mixer(**kw) -> DynamicMixer:
        args = dict(sample_rate=SR, crop_s=1.0, crop_target_prob=1.0, seed=3,
                    rir_bank=RirBank(tmp_path / "bank.npz"), reverb_prob=1.0)
        return DynamicMixer(library_of(env), env.config.labels, kw.pop("mix", mix), "train",
                            **{**args, **kw})

    m = mixer()
    a, b, c = m.example(0, 5), m.example(0, 5), m.example(1, 5)
    for ex in (a, c):
        assert ex.mixture.shape == ex.target.shape == ex.rest.shape == (SR,)
        np.testing.assert_allclose(ex.mixture, ex.target + ex.rest, atol=1e-6)
        assert ex.reverb  # reverb_prob 1
        assert ex.target_hot.shape == (len(env.config.labels),)
    np.testing.assert_array_equal(a.mixture, b.mixture)  # (epoch, index) fixes the example
    assert a.recipe == b.recipe and a.recipe != c.recipe  # a new epoch: a new mixture
    examples = [m.example(2, i) for i in range(40)]
    assert {e.recipe.id for e in examples} == {f"dyn-2-{i}" for i in range(40)}
    cut = [x for e in examples for x in e.lowpass_hz if x is not None]
    assert cut and all(mix.lowpass_hz[0] <= x <= mix.lowpass_hz[1] for x in cut)
    # with crop_target_prob = 1 the crop holds some of the target whenever there is one, even
    # when the crop could fall between two target events
    for e in examples:
        if not e.recipe.empty_target:
            assert np.abs(e.target).max() > 0, e.recipe
    dry = mixer(rir_bank=None, reverb_prob=1.0)
    assert dry.reverb_prob == 0.0 and not dry.example(0, 1).reverb
    with pytest.raises(ValueError, match="Hz"):
        mixer(sample_rate=44_100)


def test_partial_targets_switch_inside_the_crop(shared_libri_env) -> None:
    from soundspy.data.dynamic import DynamicMixer

    env = shared_libri_env
    mix = mixing(env, empty_target_prob=0.0, partial_target_prob=1.0)
    m = DynamicMixer(library_of(env), env.config.labels, mix, "train", sample_rate=SR,
                     crop_s=1.0, crop_target_prob=1.0, seed=4)
    edge = SR // 5
    for i in range(20):
        ex = m.example(0, i)
        assert ex.recipe.switch_s is not None
        switch = round(ex.recipe.switch_s * SR) - ex.start
        assert 0.2 * SR <= switch <= 0.8 * SR  # inside the crop, away from its edges
        quiet = min(np.abs(ex.target[:edge]).max(), np.abs(ex.target[-edge:]).max())
        assert quiet < 1e-6 and np.abs(ex.target).max() > 1e-3  # off on one side, on inside
