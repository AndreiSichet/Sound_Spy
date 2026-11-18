"""``HybridExtractor`` (step 07c): routing, fallbacks, one latency, pass-through, streaming ==
the whole-file evaluation functions, route switches, the chain and the API (tiny models)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from soundspy.core.audio import AudioBuffer
from soundspy.core.target import TargetSpec
from soundspy.engine.factory import build_chain
from soundspy.extraction.activity import TimelineActivity
from soundspy.extraction.hybrid import (
    OFF,
    SPEECH,
    WAVEFORMER,
    HybridExtractor,
    HybridUnavailable,
    build_hybrid_extractor,
)
from soundspy.extraction.offline import resolve_method, run_offline

SR = 48000


@pytest.fixture
def hybrid_config(config, neural_kit, speech_kit, tmp_path):
    """Tiny Waveformer (16 kHz) + tiny speech enhancer (48 kHz), method hybrid."""
    return speech_kit.make(neural_kit.make(config, tmp_path, method="hybrid"), tmp_path)


def _signal(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * 220 * t) * (t % 0.5 < 0.25)
            + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


def _timeline(labels=("speech", "music"), seconds: float = 3.0) -> TimelineActivity:
    """Speech 'sure absent' in the first second, unsure in the second, on in the third."""
    times = np.arange(0.0, seconds, 0.25)
    probs = np.zeros((times.size, len(labels)))
    probs[(times >= 1.0) & (times < 2.0), 0] = 0.3
    probs[times >= 2.0, 0] = 0.9
    return TimelineActivity(list(labels), times, probs >= 0.5, probabilities=probs,
                            thresholds=dict.fromkeys(labels, 0.5))


def test_routing_rules(hybrid_config) -> None:
    ext = build_hybrid_extractor(hybrid_config)
    assert ext.route_for(()) == OFF
    assert ext.route_for({"speech"}) == SPEECH
    assert ext.route_for({"speech", "music"}) == WAVEFORMER  # never two extractors summed
    assert ext.route_for({"alarm"}) == WAVEFORMER
    assert not ext.notes


def test_without_the_speech_enhancer_speech_goes_to_the_waveformer(tiny_neural) -> None:
    config = dataclasses.replace(tiny_neural, extraction=dataclasses.replace(
        tiny_neural.extraction, method="hybrid"))
    ext = build_hybrid_extractor(config)  # the test config has no speech model
    assert ext.route_for({"speech"}) == WAVEFORMER
    assert any("speech enhancer" in note for note in ext.notes)
    chain = build_chain("semantic", config, config.labels, with_detector=False)
    assert isinstance(chain.stage("extractor"), HybridExtractor)
    assert "hybrid_route_unavailable" in [code for code, _ in chain.warnings]


def test_speech_enhancer_alone_passes_other_selections_through(config, speech_kit,
                                                               tmp_path) -> None:
    only_speech = speech_kit.make(config, tmp_path)  # no Waveformer checkpoint
    ext = build_hybrid_extractor(only_speech)
    assert ext.route_for({"speech"}) == SPEECH and ext.route_for({"music"}) == OFF


def test_neither_route_falls_back_to_the_baseline(config) -> None:
    hybrid = dataclasses.replace(config, extraction=dataclasses.replace(config.extraction,
                                                                        method="hybrid"))
    with pytest.raises(HybridUnavailable):
        build_hybrid_extractor(hybrid)
    method, warning = resolve_method(hybrid)
    assert method == "mask_baseline" and "Hybrid extraction is not available" in warning


def test_one_latency_and_pass_through(hybrid_config) -> None:
    ext = build_hybrid_extractor(hybrid_config)
    ext.configure(SR, 1)
    assert ext.latency_samples == 1919  # DeepFilterNet3's 40.0 ms; the Waveformer is padded
    x = _signal(1.0)
    out = run_offline(ext, AudioBuffer(x.reshape(-1, 1), SR),
                      TargetSpec(labels=frozenset(), attenuation_db=40))
    assert np.abs(out.samples.ravel() - x).max() < 1e-6  # nothing selected: the input


@pytest.mark.parametrize("labels", [("speech",), ("music",), ("speech", "music")])
def test_streaming_equals_the_whole_file_functions(hybrid_config, labels) -> None:
    from soundspy.training.evaluate_hybrid import (
        HybridCores,
        chosen_route,
        gate_output,
        route_estimates,
    )

    source = _timeline()
    ext = build_hybrid_extractor(hybrid_config, source)
    x = _signal(3.0)
    target = TargetSpec(labels=frozenset(labels), attenuation_db=30)
    stream = run_offline(ext, AudioBuffer(x.reshape(-1, 1), SR), target,
                         chunk_frames=960).samples.ravel()
    cores = HybridCores(ext._routes[WAVEFORMER].stream.core, ext._speech.core,
                        hybrid_config.extraction.neural_tse.live)
    est = route_estimates(cores, x, SR, labels, hybrid_config.extraction.hybrid.speech.labels)
    route = chosen_route(hybrid_config, labels)
    whole = gate_output(est[route], x, labels, 30, hybrid_config.extraction.hybrid.gate, source,
                        SR, cores.latencies(SR)["enhancer_route"], warmup=route != "wf")
    assert np.abs(stream - whole).max() < 1e-4
    assert ext.status_info()["route"] == (SPEECH if labels == ("speech",) else WAVEFORMER)


def test_route_switch_crossfades_without_jumps(hybrid_config) -> None:
    ext = build_hybrid_extractor(hybrid_config)
    ext.configure(SR, 1)
    ext.set_target(TargetSpec(labels=frozenset({"speech"}), attenuation_db=20))
    ext.reset()
    x = _signal(3.0)
    parts = []
    for i in range(0, x.size, 960):
        if i == SR:
            ext.set_target(TargetSpec(labels=frozenset({"speech", "music"}), attenuation_db=20))
        if i == 2 * SR:
            ext.set_target(TargetSpec(labels=frozenset(), attenuation_db=20))
        parts.append(ext.process(AudioBuffer(x[i:i + 960].reshape(-1, 1), SR)).samples)
    y = np.concatenate(parts).ravel()
    assert np.isfinite(y).all() and np.abs(y).max() < 2 * np.abs(x).max()
    assert ext.route == OFF and ext.status_info()["gate_state"] in ("open", "closed")


def test_chain_builds_the_hybrid_with_the_detectors_activity(hybrid_config) -> None:
    chain = build_chain("semantic", hybrid_config, hybrid_config.labels, with_detector=False)
    ext = chain.stage("extractor")
    assert isinstance(ext, HybridExtractor)
    assert "gating_unavailable" in [code for code, _ in chain.warnings]  # no detector here
    info = ext.status_info()
    assert info["method"] == "hybrid" and set(info["routes"]) == {SPEECH, WAVEFORMER}
    assert info["speech_enhancer"]["latency_ms"] == 39.98
    off = build_chain("semantic", hybrid_config, hybrid_config.labels, with_detector=False,
                      gate=False).stage("extractor")
    assert off.status_info()["gate"] is False


FILE_SOURCE = {"mode": "SEMANTIC", "source": {"type": "file", "path": "audio/tone.wav"}}


def test_api_session_with_the_hybrid(app_config, neural_kit, speech_kit, tmp_path: Path,
                                     no_leaked_threads) -> None:
    from soundspy.api.app import create_app

    tone = _signal(1.0)
    (tmp_path / "audio").mkdir()
    sf.write(str(tmp_path / "audio" / "tone.wav"), np.stack([tone, tone], 1), SR,
             subtype="FLOAT")
    paths = {**app_config.paths, "raw": tmp_path / "data" / "raw"}
    root = dataclasses.replace(app_config, project_root=tmp_path, paths=paths)
    config = speech_kit.make(neural_kit.make(root, tmp_path, method="hybrid"), tmp_path)
    with TestClient(create_app(config)) as client:
        body = client.get("/api/config").json()
        assert body["default_method"] == "hybrid" and body["default_gate"] is True
        status = client.get("/api/session").json()
        assert status["extraction_method"] == "hybrid" and status["extraction_gate"] is True
        client.put("/api/session/target", json={"labels": ["speech"]})
        started = client.post("/api/session/start", json=FILE_SOURCE).json()
        assert started["extraction"]["method"] == "hybrid"
        assert started["extraction"]["route"] == SPEECH
        assert started["extraction"]["latency_ms"] == 39.98
        client.post("/api/session/stop")
        # another method starts with its own gate default (neural: the 07b gate, off)
        assert client.put("/api/session/gate", json={"enabled": False}).json()[
            "extraction_gate"] is False
        neural = client.put("/api/session/method", json={"method": "neural_tse"}).json()
        assert neural["extraction_gate"] is False
        hybrid = client.put("/api/session/method", json={"method": "hybrid"}).json()
        assert hybrid["extraction_gate"] is True
