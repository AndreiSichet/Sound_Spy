"""The output-driven soft gate of the neural extractor (step 07b): timing, depth, identity when
off, and streaming == whole file with the gate on."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from soundspy.config import NeuralGateConfig
from soundspy.core.audio import AudioBuffer
from soundspy.core.target import TargetSpec
from soundspy.extraction.gate import OutputGate
from soundspy.extraction.neural_tse import (
    build_neural_extractor,
    extract_whole,
    whole_estimates,
)

SR = 44_100
BLOCK = 416  # one model step of the real model: 9.4 ms
CFG = NeuralGateConfig(enabled=True, attack_ms=5, release_ms=150, open_db=-12, close_db=-22,
                       depth_db=25, hold_ms=200)


def noise(n: int, seed: int = 0, level: float = 0.1) -> np.ndarray:
    return (level * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def gains_db(gate: OutputGate, estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """The gate's gain per block, in dB (estimate in, gated estimate out)."""
    out = gate.process(estimate, reference)
    blocks = len(estimate) // BLOCK
    e = estimate[:blocks * BLOCK].reshape(blocks, BLOCK)
    o = out[:blocks * BLOCK].reshape(blocks, BLOCK)
    return 20 * np.log10(np.abs(o[:, -1]) / np.abs(e[:, -1]))  # last sample = the step's gain


def test_wanted_gain_is_a_line_in_db_between_the_thresholds() -> None:
    gate = OutputGate(CFG, SR, BLOCK)
    assert gate.wanted_gain_db(0.0) == 0.0 and gate.wanted_gain_db(-12.0) == 0.0
    assert gate.wanted_gain_db(-22.0) == -25.0 and gate.wanted_gain_db(-40.0) == -25.0
    assert gate.wanted_gain_db(-17.0) == pytest.approx(-12.5)


def test_closes_after_hold_and_release_to_the_configured_depth() -> None:
    gate = OutputGate(CFG, SR, BLOCK)
    x = noise(SR * 2)
    y = np.full_like(x, 1.0) * 0.0 + 0.003 * np.sign(x)  # the model keeps ~ -30 dB: no target
    g = gains_db(gate, y, x)
    step_ms = 1000 * BLOCK / SR
    hold_steps = int(np.ceil(CFG.hold_ms / step_ms))
    assert np.all(g[:hold_steps] > -0.01)  # starts open and holds
    assert g[hold_steps + 1] < -0.5  # then starts closing ...
    assert g[hold_steps + int(150 / step_ms)] > -25 * (1 - np.exp(-1)) - 2  # ... with ~150 ms
    assert g[-1] == pytest.approx(-25.0, abs=0.1)  # down to the depth
    assert gate.closed_share > 0.5


def test_opens_within_the_attack_time_when_the_output_rises() -> None:
    gate = OutputGate(CFG, SR, BLOCK)
    x = noise(SR)
    quiet = 0.003 * np.sign(x)
    gains_db(gate, quiet, x)  # one second without target: closed
    assert gate.gain_db == pytest.approx(-25.0, abs=0.5)
    loud = 0.7 * x  # the target starts: the estimate holds most of the input
    g = gains_db(gate, loud[:BLOCK * 4], x[:BLOCK * 4])
    # within the attack time (5 ms) plus the one step in which it is measured
    assert g[1] > -1.0 and g[2] > -0.1


def test_pauses_shorter_than_the_hold_do_not_close_it() -> None:
    gate = OutputGate(CFG, SR, BLOCK)
    x = noise(SR * 2)
    y = 0.7 * x
    pause = slice(SR // 2, SR // 2 + int(0.12 * SR))  # a 120 ms pause between two words
    y[pause] = 0.0003 * np.sign(x[pause])
    g = gains_db(gate, y, x)
    assert g.min() > -1.0


def test_depth_and_thresholds_follow_the_config() -> None:
    deep = dataclasses.replace(CFG, depth_db=40.0, hold_ms=0.0)
    gate = OutputGate(deep, SR, BLOCK)
    x = noise(SR * 3)
    g = gains_db(gate, 0.003 * np.sign(x), x)
    assert g[-1] == pytest.approx(-40.0, abs=0.1)
    gate = OutputGate(dataclasses.replace(CFG, hold_ms=0.0), SR, BLOCK)
    g = gains_db(gate, 0.17 * x, x)  # -15.4 dB: between the thresholds
    expected = gate.wanted_gain_db(20 * np.log10(0.17))
    assert -25.0 < expected < 0.0 and g[-1] == pytest.approx(expected, abs=0.3)


def test_any_split_into_calls_gives_the_same_output() -> None:
    x = noise(SR)
    y = 0.003 * np.sign(x)
    y[SR // 2:] = 0.6 * x[SR // 2:]
    whole = OutputGate(CFG, SR, BLOCK).process(y, x)
    gate = OutputGate(CFG, SR, BLOCK)
    parts = [gate.process(y[a:a + n], x[a:a + n])
             for a, n in ((0, BLOCK * 3), (BLOCK * 3, BLOCK * 50), (BLOCK * 53, len(x)))]
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=1e-7)
    with pytest.raises(ValueError, match="differ"):
        gate.process(y[:10], x[:11])


# --- inside the extractor (tiny random model, CPU) ----------------------------------------------


def _with_gate(config, **changes):
    neural = config.extraction.neural_tse
    gate = dataclasses.replace(neural.gate, enabled=True, hold_ms=20.0, **changes)
    return dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, neural_tse=dataclasses.replace(neural, gate=gate)))


def _run(extractor, x: np.ndarray, rate: int, size: int) -> np.ndarray:
    out = []
    for a in range(0, len(x), size):
        out.append(extractor.process(AudioBuffer(x[a:a + size].reshape(-1, 1), rate))
                   .samples.reshape(-1))
    return np.concatenate(out)


@pytest.mark.parametrize("rate", [16_000, 48_000])
def test_streaming_equals_the_whole_file_with_the_gate_on(tiny_neural, rate: int) -> None:
    # thresholds high enough that the random tiny model's output really is gated
    config = _with_gate(tiny_neural, open_db=10.0, close_db=-5.0)
    spec = TargetSpec(labels=frozenset({"speech"}), attenuation_db=30.0)
    x = noise(rate // 2, level=0.2)
    extractor = build_neural_extractor(config)
    assert extractor.status_info()["gate"] is True
    extractor.set_target(spec)
    extractor.configure(rate, 1)
    gate_cfg = config.extraction.neural_tse.gate
    whole = extract_whole(extractor.core, x, rate, spec.labels, 30.0, gate_cfg)
    ungated = extract_whole(extractor.core, x, rate, spec.labels, 30.0)
    assert float(np.max(np.abs(whole - ungated))) > 0.01  # the gate did something
    delay = extractor.latency_samples
    for size in (rate // 50, 333):
        extractor.reset()
        y = _run(extractor, x, rate, size)
        np.testing.assert_allclose(y[delay:], whole[:len(x) - delay], atol=2e-5)
    assert 0.0 <= extractor.status_info()["gate_closed"] <= 1.0
    # whole_estimates: one model run, the ungated estimate first
    plain, gated = whole_estimates(extractor.core, x, rate, spec.labels, (gate_cfg,))
    assert plain.shape == gated.shape == x.shape
    assert float(np.sum(gated ** 2)) < float(np.sum(plain ** 2))


def test_gate_off_is_the_extractor_without_a_gate(tiny_neural) -> None:
    rate = 16_000
    x = noise(rate // 2, level=0.2)
    spec = TargetSpec(labels=frozenset({"music"}), attenuation_db=20.0)
    outputs = []
    for extractor in (build_neural_extractor(tiny_neural),  # the fixture: gate off
                      build_neural_extractor(_with_gate(tiny_neural), gate=False)):
        assert extractor.status_info()["gate"] is False
        extractor.set_target(spec)
        extractor.configure(rate, 1)
        outputs.append(_run(extractor, x, rate, 320))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    on = build_neural_extractor(tiny_neural, gate=True)  # the session's switch
    assert on.status_info()["gate"] is True
