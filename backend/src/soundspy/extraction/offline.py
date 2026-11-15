"""Running an extractor over a whole signal, and building one from the configuration.

* :func:`build_extractor`: the extractor named by ``extraction.method`` with a given activity
  source (used by the live chain factory, offline processing and the evaluation).
* :func:`run_offline`: feeds a buffer through an extractor block by block (the same ``process``
  the live session calls), then flushes its delay, so the result is **time-aligned** with the
  input and has the same length.
* :func:`extract_file` / :func:`extract_buffer`: offline extraction of a recording. The detector
  runs over the whole file first with centred windows (``centred_timeline``), then the extractor
  uses that timeline: no detector lag, unlike live.
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

METHODS = ("mask_baseline", "passthrough")
RUN_METHODS = ("mask_baseline",)  # what offline processing offers (passthrough would be a copy)


class ExtractionUnavailable(RuntimeError):
    """Offline extraction cannot run (e.g. no detection model). Message for the user."""


def build_extractor(config: AppConfig, activity: ActivitySource | None, *,
                    method: str | None = None) -> Processor:
    method = method or config.extraction.method
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
                   attenuation_db: float = 30.0, method: str | None = None,
                   device: str | None = None) -> OfflineResult:
    """Offline extraction of a buffer (any rate/channels): centred detection, then the extractor."""
    started = time.perf_counter()
    method = method or config.extraction.method
    target = TargetSpec(labels=frozenset(labels), attenuation_db=attenuation_db)
    target.validate(config.labels)
    activity = None
    if method == "mask_baseline" and target.labels:
        activity = _centred(config, audio, device)
    extractor = build_extractor(config, activity, method=method)
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
                 attenuation_db: float = 30.0, method: str | None = None,
                 device: str | None = None) -> OfflineResult:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return extract_buffer(config, AudioBuffer(data, int(sr)), labels,
                          attenuation_db=attenuation_db, method=method, device=device)


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


def method_parameters(config: AppConfig, method: str) -> dict[str, Any]:
    import dataclasses

    if method == "mask_baseline":
        return dataclasses.asdict(config.extraction.mask_baseline)
    return {}


def start_run(config: AppConfig, recording: Path, labels: Collection[str], *,
              attenuation_db: float = 30.0, method: str | None = None) -> dict[str, Any]:
    """Create a ``pending`` run for ``recording`` (validates labels and method)."""
    method = method or config.extraction.method
    if method not in RUN_METHODS:
        raise ValueError(f"unknown method {method!r}; methods: {list(RUN_METHODS)}")
    target = TargetSpec(labels=frozenset(labels), attenuation_db=attenuation_db)
    if not target.labels:
        raise ValueError("select at least one sound class to extract")
    target.validate(config.labels)
    return run_store.create_run(
        runs_dir(config), recording=stored_path(recording, config.project_root),
        labels=sorted(target.labels), attenuation_db=target.attenuation_db, method=method,
        parameters=method_parameters(config, method), activity_mode="timeline")


def execute_run(config: AppConfig, run_id: str, *, device: str | None = None) -> dict[str, Any]:
    """Process a pending run; returns its final sidecar (``done`` or ``failed``, never raises)."""
    directory = runs_dir(config)
    run = run_store.update_run(directory, run_id, status="running")
    try:
        path = absolute_path(run["recording"], config.project_root)
        if not path.is_file():
            raise ExtractionUnavailable(f"the recording file is gone: {run['recording']}")
        result = extract_file(config, path, run["labels"], attenuation_db=run["attenuation_db"],
                              method=run["method"], device=device)
        return run_store.finish_run(directory, run_id, result.audio,
                                    processing_seconds=result.seconds, info=result.info)
    except Exception as exc:  # noqa: BLE001 - a run records its failure instead of raising
        message = str(exc) if isinstance(exc, ExtractionUnavailable) else \
            f"{type(exc).__name__}: {exc}"
        return run_store.fail_run(directory, run_id, message)
