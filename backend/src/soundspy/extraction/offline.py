"""Running an extractor over a whole signal, and building one from the configuration.

* :func:`resolve_method`: which method really runs: the requested one, or the baseline with a
  reason when ``neural_tse`` has no checkpoint (step 07) or ``hybrid`` has neither of its routes
  (step 07c).
* :func:`build_extractor`: the extractor of a method (``hybrid``, ``neural_tse``,
  ``mask_baseline`` with a given activity source, or ``passthrough``); used by the live chain
  factory, offline processing and the evaluation.
* :func:`run_offline`: feeds a buffer through an extractor block by block (the same ``process``
  the live session calls), then flushes its delay, so the result is **time-aligned** with the
  input and has the same length.
* :func:`extract_file` / :func:`extract_buffer`: offline extraction of a recording. With the
  baseline the detector runs over the whole file first with centred windows
  (``centred_timeline``), then the extractor uses that timeline: no detector lag, unlike live.
  The neural extractor needs no detector: the file simply goes through it block by block.
  The hybrid (step 07c) gets the centred timeline for its gate when a detector is there.
* :func:`start_run` / :func:`execute_run`: the same as a **processing run** on disk
  (``io/runs.py``: WAV + sidecar in ``outputs/processed``), used by the CLI and the API worker.
* :func:`irm_oracle`: the ideal ratio mask from the true target and rest, in the same STFT. No
  single-channel masking method can do better; it is the upper bound in the evaluation.
"""

from __future__ import annotations

import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import AppConfig, MaskBaselineConfig
from soundspy.core.audio import AudioBuffer
from soundspy.core.target import TargetSpec
from soundspy.dsp.channels import ChannelReducer
from soundspy.dsp.stft import StftSpec, istft, stft
from soundspy.engine.processor import Passthrough, Processor
from soundspy.extraction.activity import ActivitySource, TimelineActivity, centred_timeline
from soundspy.extraction.mask_baseline import MaskBaselineExtractor
from soundspy.io import runs as run_store
from soundspy.io.storage import absolute_path, stored_path

METHODS = ("hybrid", "neural_tse", "mask_baseline", "passthrough")
# what offline processing offers (passthrough would be a copy)
RUN_METHODS = ("hybrid", "neural_tse", "mask_baseline")
# methods that use the detector's timeline offline (the baseline to gate, the hybrid's gate)
TIMELINE_METHODS = ("mask_baseline", "hybrid")


class ExtractionUnavailable(RuntimeError):
    """Offline extraction cannot run (e.g. no detection model). Message for the user."""


def default_gate(config: AppConfig, method: str) -> bool:
    """Whether a method's gate is on when the session did not choose (step 07c): the hybrid's
    gate for ``hybrid``, the step-07b output gate for ``neural_tse``."""
    if method == "hybrid":
        return config.extraction.hybrid.gate.enabled
    if method == "neural_tse":
        return config.extraction.neural_tse.gate.enabled
    return False


def hybrid_available(config: AppConfig) -> str | None:
    """``None`` if at least one route of the hybrid extractor can be built (step 07c)."""
    from soundspy.config import speech_model_dir

    speech = (speech_model_dir(config) / "config.ini").is_file()
    if speech or neural_available(config) is None:
        return None
    return ("neither the Waveformer checkpoint nor the speech enhancement model exists "
            "(`soundspy data download deepfilternet`)")


def neural_available(config: AppConfig) -> str | None:
    """``None`` if the neural extractor can be built, else the reason why not."""
    checkpoint = config.extraction.neural_tse.checkpoint
    if checkpoint is None:
        return "no checkpoint is configured (extraction.neural_tse.checkpoint)"
    if not Path(checkpoint).is_file():
        return (f"its model file does not exist yet ({checkpoint}); train it with "
                "`soundspy train extraction`")
    return None


def resolve_method(config: AppConfig, requested: str | None = None) -> tuple[str, str | None]:
    """``(method that runs, warning or None)``.

    ``requested`` (default: ``extraction.method``) is used as it is, except that ``neural_tse``
    without a usable checkpoint falls back to ``mask_baseline``; the warning says so.
    """
    method = requested or config.extraction.method
    if method not in METHODS:
        raise ValueError(f"unknown extraction method {method!r}; methods: {list(METHODS)}")
    if method == "hybrid":
        reason = hybrid_available(config)
        if reason is not None:
            return "mask_baseline", ("Hybrid extraction is not available: " + reason
                                     + ". Using the baseline (detector-gated) method instead.")
        return method, None
    if method == "neural_tse":
        reason = neural_available(config)
        if reason is not None:
            return "mask_baseline", ("Neural extraction is not available: " + reason
                                     + ". Using the baseline (detector-gated) method instead.")
    return method, None


def build_extractor(config: AppConfig, activity: ActivitySource | None, *,
                    method: str | None = None, device: str | None = None,
                    gate: bool | None = None) -> Processor:
    """The extractor of ``method`` exactly (no fallback: see :func:`resolve_method`).
    ``gate``: the neural output gate on / off (``None``: as configured)."""
    method = method or config.extraction.method
    if method == "hybrid":
        from soundspy.extraction.hybrid import HybridUnavailable, build_hybrid_extractor

        try:
            return build_hybrid_extractor(config, activity, device=device, gate=gate)
        except HybridUnavailable as exc:
            raise ExtractionUnavailable(str(exc)) from exc
    if method == "neural_tse":
        from soundspy.extraction.neural_tse import NeuralUnavailable, build_neural_extractor

        try:
            return build_neural_extractor(config, device=device, gate=gate)
        except NeuralUnavailable as exc:
            raise ExtractionUnavailable(str(exc)) from exc
    if method == "mask_baseline":
        return MaskBaselineExtractor(config.extraction.mask_baseline, activity)
    if method == "passthrough":
        return Passthrough()
    raise ValueError(f"unknown extraction method {method!r}; methods: {list(METHODS)}")


def run_offline(extractor: Processor, audio: AudioBuffer, target: TargetSpec, *,
                chunk_frames: int = 4096) -> AudioBuffer:
    """``audio`` through ``extractor``, delay-compensated (same length, aligned with the input)."""
    sr, channels = audio.sample_rate, audio.num_channels
    out_channels = extractor.configure(sr, channels)
    extractor.set_target(target)
    extractor.reset()
    extractor.start()  # e.g. a GPU model's warm-up, on this thread
    delay = extractor.latency_samples
    padded = np.concatenate((audio.samples, np.zeros((delay, channels), dtype=np.float32)))
    parts = [extractor.process(AudioBuffer(padded[i:i + chunk_frames], sr)).samples
             for i in range(0, padded.shape[0], chunk_frames)]
    out = np.concatenate(parts) if parts else np.zeros((0, out_channels), np.float32)
    return AudioBuffer(np.ascontiguousarray(out[delay:delay + audio.num_frames]), sr,
                       dict(audio.metadata))


def mono(audio: AudioBuffer) -> AudioBuffer:
    """One channel via the chain's own reducer (mean), as in the live ``SEMANTIC`` chain."""
    if audio.num_channels == 1:
        return audio
    reducer = ChannelReducer("mean")
    reducer.configure(audio.sample_rate, audio.num_channels)
    return reducer.process(audio)


@dataclass
class OfflineResult:
    audio: AudioBuffer  # one channel, aligned with the input
    activity: TimelineActivity | None
    seconds: float  # processing time
    info: dict[str, Any]


def extract_buffer(config: AppConfig, audio: AudioBuffer, labels: Collection[str], *,
                   attenuation_db: float | None = None, method: str | None = None,
                   device: str | None = None, gate: bool | None = None) -> OfflineResult:
    """Offline extraction of a buffer (any rate/channels).

    ``mask_baseline``: centred detection first, then the extractor. ``neural_tse``: the buffer
    goes through the model block by block. ``attenuation_db`` defaults to
    ``extraction.default_attenuation_db``.
    """
    started = time.perf_counter()
    method = method or resolve_method(config)[0]
    if attenuation_db is None:
        attenuation_db = config.extraction.default_attenuation_db
    target = TargetSpec(labels=frozenset(labels), attenuation_db=attenuation_db)
    target.validate(config.labels)
    activity = None
    notes: list[str] = []
    if method == "mask_baseline" and target.labels:
        activity = _centred(config, audio, device)
    elif method == "hybrid" and target.labels:
        try:  # the hybrid's gate uses the detector; without it the gate stays open
            activity = _centred(config, audio, device)
        except ExtractionUnavailable as exc:
            notes.append(f"gate without detector (stays open): {exc}")
    extractor = build_extractor(config, activity, method=method, device=device, gate=gate)
    out = run_offline(extractor, mono(audio), target,
                      chunk_frames=config.extraction.offline_chunk_frames)
    info = {
        "method": method,
        "activity_mode": "timeline" if activity is not None else "none",
        "labels": sorted(target.labels),
        "attenuation_db": target.attenuation_db,
        "latency_samples": extractor.latency_samples,
    }
    if activity is not None:
        info["target_segments"] = {lab: [[round(a, 3), round(b, 3)] for a, b in
                                         activity.segments(lab)] for lab in sorted(target.labels)}
    status = getattr(extractor, "status_info", None)
    if method == "neural_tse" and callable(status):
        report = status()
        info.update(model=report.get("model"), device=report.get("device"),
                    gate=report.get("gate"), gate_closed=report.get("gate_closed"))
    if method == "hybrid" and callable(status):
        report = status()
        info.update(route=report.get("route"), model=report.get("model"),
                    device=report.get("device"), gate=report.get("gate"),
                    gate_closed=report.get("gate_closed"),
                    notes=notes + list(report.get("notes") or []))
    return OfflineResult(out, activity, time.perf_counter() - started, info)


def _centred(config: AppConfig, audio: AudioBuffer, device: str | None) -> TimelineActivity:
    from soundspy.detection.inference import DetectorUnavailable, load_core

    det = config.detection
    if det is None or det.checkpoint is None:
        raise ExtractionUnavailable(
            "no detection model is configured (detection.checkpoint); the baseline needs it to "
            "know when the selected sounds are present")
    try:
        core = load_core(Path(det.checkpoint), device or det.live.device)
    except DetectorUnavailable as exc:
        raise ExtractionUnavailable(str(exc)) from exc
    return centred_timeline(core, audio, hop_s=det.live.hop_s, ema_alpha=det.live.ema_alpha,
                            hysteresis=det.live.hysteresis)


def extract_file(config: AppConfig, path: Path, labels: Collection[str], *,
                 attenuation_db: float | None = None, method: str | None = None,
                 device: str | None = None, gate: bool | None = None) -> OfflineResult:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return extract_buffer(config, AudioBuffer(data, int(sr)), labels,
                          attenuation_db=attenuation_db, method=method, device=device,
                          gate=gate)


def irm_oracle(mixture: np.ndarray, target: np.ndarray, rest: np.ndarray,
               cfg: MaskBaselineConfig) -> np.ndarray:
    """Ideal ratio mask ``sqrt(|T|^2 / (|T|^2 + |R|^2))`` applied to the mixture (1-D arrays)."""
    spec = StftSpec(cfg.n_fft, cfg.hop)
    mix, tgt, rst = (stft(np.asarray(x, dtype=np.float32).ravel(), spec)
                     for x in (mixture, target, rest))
    t2 = tgt.real ** 2 + tgt.imag ** 2
    r2 = rst.real ** 2 + rst.imag ** 2
    mask = np.sqrt(t2 / np.maximum(t2 + r2, 1e-20)).astype(np.float32)
    return istft(mix * mask, spec, np.asarray(mixture).size)


# --------------------------------------------------------------------------------------------
# Processing runs (WAV + sidecar in outputs/processed)


def runs_dir(config: AppConfig) -> Path:
    return Path(config.paths["outputs"]) / "processed"


def method_parameters(config: AppConfig, method: str, gate: bool | None = None
                      ) -> dict[str, Any]:
    """What a run sidecar records about the method (the neural model's file and gate)."""
    import dataclasses

    if method == "mask_baseline":
        return dataclasses.asdict(config.extraction.mask_baseline)
    if method == "hybrid":
        hy = config.extraction.hybrid
        waveformer = method_parameters(config, "neural_tse", False)
        gate_cfg = hy.gate if gate is None else dataclasses.replace(hy.gate, enabled=gate)
        speech = dataclasses.asdict(hy.speech)
        speech["model_dir"] = None if hy.speech.model_dir is None else str(hy.speech.model_dir)
        return {"gate": dataclasses.asdict(gate_cfg), "speech": speech,
                "ramp_ms": hy.ramp_ms, "waveformer": waveformer}
    if method == "neural_tse":
        cfg = config.extraction.neural_tse
        checkpoint = Path(cfg.checkpoint) if cfg.checkpoint else None
        gate_cfg = cfg.gate if gate is None else dataclasses.replace(cfg.gate, enabled=gate)
        return {
            "model_name": checkpoint.parent.parent.name if checkpoint else None,
            "gate": dataclasses.asdict(gate_cfg),
            "checkpoint": stored_path(checkpoint, config.project_root) if checkpoint else None,
            "checkpoint_modified": (time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(checkpoint.stat().st_mtime))
                if checkpoint and checkpoint.is_file() else None),
            "model": dataclasses.asdict(cfg.model),
            "block_chunks": cfg.live.block_chunks,
        }
    return {}


def run_method(config: AppConfig, requested: str | None = None) -> str:
    """The method an offline run uses: ``requested``, or the default that can actually run."""
    if requested is not None:
        if requested not in RUN_METHODS:
            raise ValueError(f"unknown method {requested!r}; methods: {list(RUN_METHODS)}")
        return requested
    method = resolve_method(config)[0]
    return method if method in RUN_METHODS else "mask_baseline"


def start_run(config: AppConfig, recording: Path, labels: Collection[str], *,
              attenuation_db: float | None = None, method: str | None = None,
              gate: bool | None = None) -> dict[str, Any]:
    """Create a ``pending`` run for ``recording`` (validates labels and method)."""
    method = run_method(config, method)
    if attenuation_db is None:
        attenuation_db = config.extraction.default_attenuation_db
    target = TargetSpec(labels=frozenset(labels), attenuation_db=attenuation_db)
    if not target.labels:
        raise ValueError("select at least one sound class to extract")
    target.validate(config.labels)
    return run_store.create_run(
        runs_dir(config), recording=stored_path(recording, config.project_root),
        labels=sorted(target.labels), attenuation_db=target.attenuation_db, method=method,
        parameters=method_parameters(config, method, gate),
        activity_mode="timeline" if method in TIMELINE_METHODS else "none")


def execute_run(config: AppConfig, run_id: str, *, device: str | None = None) -> dict[str, Any]:
    """Process a pending run; returns its final sidecar (``done`` or ``failed``, never raises)."""
    directory = runs_dir(config)
    run = run_store.update_run(directory, run_id, status="running")
    try:
        path = absolute_path(run["recording"], config.project_root)
        if not path.is_file():
            raise ExtractionUnavailable(f"the recording file is gone: {run['recording']}")
        gate = (run.get("parameters") or {}).get("gate")
        result = extract_file(config, path, run["labels"], attenuation_db=run["attenuation_db"],
                              method=run["method"], device=device,
                              gate=gate.get("enabled") if isinstance(gate, dict) else None)
        return run_store.finish_run(directory, run_id, result.audio,
                                    processing_seconds=result.seconds, info=result.info)
    except Exception as exc:  # noqa: BLE001 - a run records its failure instead of raising
        message = str(exc) if isinstance(exc, ExtractionUnavailable) else \
            f"{type(exc).__name__}: {exc}"
        return run_store.fail_run(directory, run_id, message)
