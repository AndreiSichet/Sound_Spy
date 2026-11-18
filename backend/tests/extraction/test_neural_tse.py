"""``NeuralTSEExtractor`` with a tiny random Waveformer (CPU, no downloads): streaming,
pass-through, residual mixing, state handling, fades."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from soundspy.core.audio import AudioBuffer
from soundspy.core.target import TargetSpec
from soundspy.extraction.neural_tse import (
    NeuralTSEExtractor,
    NeuralUnavailable,
    build_neural_extractor,
    extract_whole,
    load_neural_core,
)

MODEL_SR = 16_000  # the tiny model's rate (conftest.TINY_WAVEFORMER): 12-sample steps
HOP, CONTEXT, LOOKAHEAD = 12, 4, 4


def noise(n: int, seed: int = 0) -> np.ndarray:
    return (0.2 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def target(*labels: str, db: float = 30.0) -> TargetSpec:
    return TargetSpec(labels=frozenset(labels), attenuation_db=db)


def make(config, sample_rate: int, spec: TargetSpec | None, **live) -> NeuralTSEExtractor:
    """An extractor whose target is set **before** ``configure`` (so there is no fade-in)."""
    extractor = build_neural_extractor(config)
    if live:
        extractor.live = dataclasses.replace(extractor.live, **live)
    if spec is not None:
        extractor.set_target(spec)
    extractor.configure(sample_rate, 1)
    return extractor


def run(extractor: NeuralTSEExtractor, x: np.ndarray, sample_rate: int,
        sizes: tuple[int, ...] = (320,)) -> np.ndarray:
    out, start, i = [], 0, 0
    while start < len(x):
        size = sizes[i % len(sizes)]
        chunk = AudioBuffer(x[start:start + size].reshape(-1, 1), sample_rate)
        y = extractor.process(chunk)
        assert y.samples.shape == chunk.samples.shape and y.samples.dtype == np.float32
        out.append(y.samples.reshape(-1))
        start, i = start + size, i + 1
    return np.concatenate(out)


def test_latency_is_reported_and_constant(tiny_neural) -> None:
    same = make(tiny_neural, MODEL_SR, target("speech"))
    # a whole step may still be missing (hop - 1 samples) + the model's lookahead
    assert same.latency_samples == LOOKAHEAD + HOP - 1 == 15
    bridged = make(tiny_neural, 48_000, target("speech"))
    waiting = int(np.ceil(15 * 3))  # the same wait in 48 kHz samples
    assert bridged.latency_samples > waiting  # + the two resampling filters
    assert bridged.latency_samples == waiting + bridged.stream._filter_delay
    info = bridged.status_info()
    assert info["method"] == "neural_tse" and info["device"] == "cpu"
    assert info["latency_ms"] == round(1000 * bridged.latency_samples / 48_000, 2)
    assert info["model"] == "tse_tiny" and info["model_ms"] is None  # nothing processed yet
    blocks = make(tiny_neural, MODEL_SR, target("speech"), block_chunks=3)
    assert blocks.latency_samples == LOOKAHEAD + 3 * HOP - 1  # waits for three steps per call
    with pytest.raises(ValueError, match="one channel"):
        build_neural_extractor(tiny_neural).configure(48_000, 2)


@pytest.mark.parametrize("sample_rate", [MODEL_SR, 48_000])
def test_streaming_equals_the_whole_signal(tiny_neural, sample_rate: int) -> None:
    """Chunk by chunk (any chunk sizes) == the model over the whole signal, delayed by the
    reported latency. At 48 kHz this includes both streaming resamplers."""
    x = noise(sample_rate // 2)
    spec = target("speech", "music", db=30.0)
    extractor = make(tiny_neural, sample_rate, spec)
    extractor.start()  # the warm-up must not leave anything behind
    whole = extract_whole(extractor.core, x, sample_rate, spec.labels, spec.attenuation_db)
    assert whole.shape == x.shape
    delay = extractor.latency_samples
    for sizes in ((sample_rate // 50,), (1, 37, 500, 3, 1024, 7)):
        extractor.reset()
        y = run(extractor, x, sample_rate, sizes)
        assert y.shape == x.shape
        np.testing.assert_allclose(y[delay:], whole[: len(x) - delay], atol=2e-5)
        if sample_rate == MODEL_SR:  # (with resampling the filters ripple a little earlier)
            np.testing.assert_allclose(y[:delay], 0.0, atol=1e-6)  # nothing before the delay
    assert extractor.status_info()["model_ms"] > 0
    # the model really does something (random weights: output != input)
    assert float(np.max(np.abs(whole - x))) > 0.05


def test_block_chunks_only_adds_delay(tiny_neural) -> None:
    x = noise(4000)
    one = make(tiny_neural, MODEL_SR, target("alarm"))
    three = make(tiny_neural, MODEL_SR, target("alarm"), block_chunks=3)
    a, b = run(one, x, MODEL_SR), run(three, x, MODEL_SR)
    shift = three.latency_samples - one.latency_samples
    assert shift == 2 * HOP
    np.testing.assert_allclose(b[shift:], a[: len(a) - shift], atol=2e-5)


@pytest.mark.parametrize("sample_rate", [MODEL_SR, 48_000])
def test_no_selection_passes_the_input_through(tiny_neural, sample_rate: int, monkeypatch
                                               ) -> None:
    x = noise(sample_rate // 4, seed=3)
    for spec in (None, target(), target(db=40.0)):
        extractor = make(tiny_neural, sample_rate, spec)
        calls: list[int] = []

        def never(*args, _calls=calls, **kwargs):
            _calls.append(1)
            pytest.fail("the model was run")

        monkeypatch.setattr(extractor.core, "step", never)
        y = run(extractor, x, sample_rate, (480, 7, 333))
        delay = extractor.latency_samples
        np.testing.assert_array_equal(y[delay:], x[: len(x) - delay])  # bit-exact, delayed
        assert not y[:delay].any() and not calls
        assert extractor.status_info()["target_active"] is False
        monkeypatch.undo()


def test_residual_mixing_follows_the_attenuation(tiny_neural) -> None:
    """``y = est + g * (dry - est)`` with ``g = 10^(-dB / 20)``."""
    x = noise(6000, seed=4)
    outputs = {}
    for db in (0.0, 15.0, 30.0, 200.0):
        extractor = make(tiny_neural, MODEL_SR, target("dishes", db=db))
        outputs[db] = run(extractor, x, MODEL_SR)
    delay = extractor.latency_samples
    dry = np.concatenate((np.zeros(delay, dtype=np.float32), x[: len(x) - delay]))
    estimate = outputs[200.0]  # g = 1e-10: the model's estimate alone
    np.testing.assert_allclose(outputs[0.0], dry, atol=1e-6)  # 0 dB = the (delayed) input
    for db in (15.0, 30.0):
        g = 10 ** (-db / 20)
        np.testing.assert_allclose(outputs[db], estimate + g * (dry - estimate), atol=1e-5)
    assert float(np.max(np.abs(estimate - dry))) > 0.05
    # the removed part gets quieter by exactly the slider value
    removed = lambda y: float(np.sqrt(np.mean((y - estimate)[delay:] ** 2)))  # noqa: E731
    ratio_db = 20 * np.log10(removed(outputs[15.0]) / removed(outputs[0.0]))
    assert ratio_db == pytest.approx(-15.0, abs=0.01)


def test_changing_the_attenuation_needs_no_model_restart(tiny_neural) -> None:
    x = noise(4000, seed=5)
    live = make(tiny_neural, MODEL_SR, target("speech", db=30.0))
    first = run(live, x[:2000], MODEL_SR)
    live.set_target(target("speech", db=10.0))  # same classes, another slider value
    second = run(live, x[2000:], MODEL_SR)
    reference = make(tiny_neural, MODEL_SR, target("speech", db=10.0))
    expected = run(reference, x, MODEL_SR)
    np.testing.assert_allclose(second, expected[2000:], atol=1e-5)  # same model state
    assert not np.allclose(first, expected[:2000], atol=1e-4)


def count_resets(extractor: NeuralTSEExtractor, monkeypatch) -> list[int]:
    calls: list[int] = []
    original = extractor.core.init_state
    monkeypatch.setattr(extractor.core, "init_state",
                        lambda: calls.append(1) or original())
    return calls


@pytest.mark.parametrize("mode", ["keep", "reset"])
def test_model_state_on_a_target_change(tiny_neural, monkeypatch, mode: str) -> None:
    x = noise(6000, seed=6)
    extractor = make(tiny_neural, MODEL_SR, target("speech"), on_target_change=mode)
    resets = count_resets(extractor, monkeypatch)
    run(extractor, x[:3000], MODEL_SR)
    assert len(resets) == 1  # the first step starts from an empty state
    extractor.set_target(target("music"))
    after = run(extractor, x[3000:], MODEL_SR)
    assert len(resets) == (2 if mode == "reset" else 1)

    # what a model that never saw the first half would give, vs one that kept its history
    fresh = make(tiny_neural, MODEL_SR, target("music"))
    kept = make(tiny_neural, MODEL_SR, target("music"))
    run(kept, x[:3000], MODEL_SR)
    kept_out = run(kept, x[3000:], MODEL_SR)
    settled = slice(200, None)  # the encoder's memory of the tiny model is 24 samples
    if mode == "keep":
        np.testing.assert_allclose(after[settled], kept_out[settled], atol=1e-4)
    else:
        assert not np.allclose(after[:60], kept_out[:60], atol=1e-5)
    del fresh


def test_reset_starts_from_scratch(tiny_neural) -> None:
    x = noise(3000, seed=7)
    extractor = make(tiny_neural, 48_000, target("traffic"))
    first = run(extractor, x, 48_000)
    run(extractor, noise(1000, seed=8), 48_000)
    extractor.reset()
    np.testing.assert_array_equal(run(extractor, x, 48_000), first)
    assert extractor.status_info()["target_active"] is True


def test_selecting_and_deselecting_fades(tiny_neural, monkeypatch) -> None:
    """Pass-through <-> extraction is faded over ``ramp_ms`` (no click), and the model starts
    from a clean state when a class is selected again."""
    x = noise(8000, seed=9)
    extractor = make(tiny_neural, MODEL_SR, None, ramp_ms=20.0)  # 320 samples at 16 kHz
    resets = count_resets(extractor, monkeypatch)
    delay = extractor.latency_samples
    dry = np.concatenate((np.zeros(delay, dtype=np.float32), x[: len(x) - delay]))

    y0 = run(extractor, x[:2000], MODEL_SR, (400,))
    np.testing.assert_array_equal(y0, dry[:2000])
    extractor.set_target(target("speech", db=60.0))
    y1 = run(extractor, x[2000:4000], MODEL_SR, (400,))
    difference = np.abs(y1 - dry[2000:4000])
    assert difference[:5].max() < 0.02 < difference[400:].max()  # starts at the input, fades
    assert len(resets) == 1

    extractor.set_target(target())  # deselect: fade back, the model still runs meanwhile
    y2 = run(extractor, x[4000:6000], MODEL_SR, (400,))
    difference = np.abs(y2 - dry[4000:6000])
    assert difference[:200].max() > 0.02 and not difference[400:].any()
    assert extractor.status_info()["target_active"] is False

    extractor.set_target(target("speech", db=60.0))  # again: a fresh model state
    run(extractor, x[6000:], MODEL_SR, (400,))
    assert len(resets) == 2


def test_unavailable_model_is_reported(tiny_neural, config, tmp_path) -> None:
    with pytest.raises(NeuralUnavailable, match="no neural extraction checkpoint"):
        build_neural_extractor(config)  # the fixture has no checkpoint
    with pytest.raises(NeuralUnavailable, match="soundspy train extraction"):
        load_neural_core(tmp_path / "nothing.pt", "cpu")
    (tmp_path / "broken.pt").write_bytes(b"junk")
    with pytest.raises(NeuralUnavailable):
        load_neural_core(tmp_path / "broken.pt", "cpu")
    checkpoint = tiny_neural.extraction.neural_tse.checkpoint
    assert load_neural_core(checkpoint, "cpu") is load_neural_core(checkpoint, "cpu")  # cached
    core = load_neural_core(checkpoint, "cpu")
    assert core.labels == list(config.labels) and core.sample_rate == MODEL_SR
    assert (core.hop, core.context, core.lookahead) == (HOP, CONTEXT, LOOKAHEAD)
    vector = core.label_vector({"music", "traffic"})
    assert vector.tolist() == [1.0 if lab in ("music", "traffic") else 0.0
                               for lab in config.labels]


# --- compiled (TorchScript-traced) steps -------------------------------------------------------


def test_compiled_steps_give_the_same_output(tiny_neural) -> None:
    """``compile: true`` traces the step per number of steps per call; the output must not
    change, and a step count that was not prepared runs untraced."""
    x = noise(8000, seed=11)
    spec = target("speech", "dishes", db=30.0)
    plain = make(tiny_neural, MODEL_SR, spec)  # the fixture has compile off
    reference = run(plain, x, MODEL_SR, (320,))
    assert plain.status_info()["compiled"] is False and not plain.core._compiled

    compiled = make(tiny_neural, MODEL_SR, spec, compile=True)
    compiled.start()  # traces calls of 1, 2 and 3 steps (block_chunks .. + 2)
    core = compiled.core
    assert sorted(k for k, v in core._compiled.items() if v is not None) == [1, 2, 3]
    assert compiled.status_info()["compiled"] is True
    np.testing.assert_allclose(run(compiled, x, MODEL_SR, (20,)), reference, atol=2e-5)  # 1-2 steps
    compiled.reset()
    # 320-sample chunks = 26-27 steps per call: not prepared, so the plain step is used
    np.testing.assert_allclose(run(compiled, x, MODEL_SR, (320,)), reference, atol=2e-5)
    assert 26 not in core._compiled

    assert core.compile_steps([2, 2, 0, -1, 5]) == [2, 5]  # idempotent; nonsense is ignored
    core._compiled.clear()  # the core is shared (cached): leave it as other tests expect it


def test_a_wrong_trace_is_not_used(tiny_neural, monkeypatch) -> None:
    import soundspy.extraction.neural_tse as module

    extractor = make(tiny_neural, MODEL_SR, target("speech"), compile=True)
    core = extractor.core
    monkeypatch.setattr(module, "COMPILE_TOLERANCE", -1.0)  # every trace "differs"
    assert core.compile_steps([1, 2]) == []
    assert core._compiled == {1: None, 2: None}
    x = noise(2000, seed=12)
    reference = run(make(tiny_neural, MODEL_SR, target("speech")), x, MODEL_SR, (24,))
    np.testing.assert_allclose(run(extractor, x, MODEL_SR, (24,)), reference, atol=2e-5)
    assert extractor.status_info()["compiled"] is False
    core._compiled.clear()
