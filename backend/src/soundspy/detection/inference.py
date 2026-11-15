"""Running the detection model: live (``Detector``) and offline (``detect_timeline``).

**Shared core.** ``DetectionCore`` holds the trained model, its tuned thresholds and the device.
``predict(windows)`` takes 1 s windows at the model rate and returns one probability per class.
``Smoother`` turns a stream of probabilities into stable on/off decisions: an exponential moving
average (``p = a * new + (1 - a) * previous``) and hysteresis (a class switches **on** at its
threshold and **off** only below ``threshold - hysteresis``), so labels do not flicker.

**Live** (``Detector``, a ``SideProcessor`` in the ``SEMANTIC`` chain):

* every input chunk is mixed to mono (``ChannelReducer``) and written into a ring buffer holding
  the last ``window_s`` at the **native** rate;
* every ``hop_s`` of audio (counted in samples, so it does not depend on the chunk size) the
  current window is handed to an **inference thread**, which resamples it to the model rate and
  runs the model;
* if that thread is still busy with the previous window, the new one is **skipped** and counted:
  the audio path never waits for the model;
* ``observe()`` returns the results finished since the previous call as ``DetectionFrame``
  events (``t`` = end of the analysed window).

**Offline** (``detect_timeline``): the same windows, hop and smoothing over a whole recording,
without threads; returns per-class probabilities over time and the active segments.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.core.audio import AudioBuffer
from soundspy.core.events import Detection, DetectionFrame, Event
from soundspy.dsp.channels import ChannelReducer
from soundspy.dsp.resample import resample_array
from soundspy.engine.processor import SideProcessor

log = logging.getLogger(__name__)

#: Longest a session start waits for the inference thread's warm-up (a hung GPU must not hang it).
WARM_UP_TIMEOUT_S = 60.0


class DetectorUnavailable(RuntimeError):
    """The detector cannot run (no checkpoint, no torch, loading failed). Message for the user."""


# --------------------------------------------------------------------------------------------
# Core


class DetectionCore:
    """Trained model + thresholds on a device; thread-safe ``predict`` (one call at a time)."""

    def __init__(self, model, thresholds: dict[str, float], device: str = "cpu",
                 name: str = "detector") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.half = self.device.type == "cuda"
        self.model = model.to(self.device).eval()
        self.labels: list[str] = list(model.spec.labels)
        missing = [lab for lab in self.labels if lab not in thresholds]
        if missing:
            raise DetectorUnavailable(f"thresholds missing for {missing}")
        self.thresholds = {lab: float(thresholds[lab]) for lab in self.labels}
        self.sample_rate = int(model.spec.sample_rate)
        self.window_s = float(model.spec.window_s)
        self.window_samples = int(round(self.window_s * self.sample_rate))
        self.name = name
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, device: str = "auto") -> DetectionCore:
        """Load ``best.pt`` and the experiment's ``thresholds.json`` (two folders up)."""
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise DetectorUnavailable(
                f"no detection model at {checkpoint}: train one with `soundspy train detection`"
                " (or set detection.checkpoint in configs/detection/default.yaml)")
        thresholds_path = checkpoint.parent.parent / "thresholds.json"
        if not thresholds_path.exists():
            raise DetectorUnavailable(f"{thresholds_path} is missing (training did not finish)")
        try:
            import json

            import torch

            from soundspy.detection.model import load_checkpoint

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                raise DetectorUnavailable("detection.live.device is 'cuda' but CUDA is not "
                                          "available")
            model, _payload = load_checkpoint(checkpoint)
            thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["thresholds"]
        except DetectorUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any loading problem becomes a user message
            raise DetectorUnavailable(f"loading {checkpoint} failed: "
                                      f"{type(exc).__name__}: {exc}") from exc
        return cls(model, thresholds, device, name=checkpoint.parent.parent.name)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """``(n, samples)`` at the model rate -> ``(n, classes)`` probabilities (float32)."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).to(self.device)
        with self._lock, torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.half):
            p = torch.sigmoid(self.model(x).float())
        return p.cpu().numpy()

    def warm_up(self) -> None:
        self.predict(np.zeros((1, self.window_samples), dtype=np.float32))


_CORE_CACHE: dict[tuple[str, float, str], DetectionCore] = {}
_CORE_CACHE_LOCK = threading.Lock()


def load_core(checkpoint: Path, device: str = "auto") -> DetectionCore:
    """``DetectionCore.from_checkpoint`` cached per (file, modification time, device)."""
    checkpoint = Path(checkpoint)
    key = (str(checkpoint.resolve()), checkpoint.stat().st_mtime if checkpoint.exists() else 0.0,
           device)
    with _CORE_CACHE_LOCK:
        core = _CORE_CACHE.get(key)
        if core is None:
            core = DetectionCore.from_checkpoint(checkpoint, device)
            _CORE_CACHE.clear()  # one model in memory at a time
            _CORE_CACHE[key] = core
    return core


# --------------------------------------------------------------------------------------------
# Smoothing


class Smoother:
    """EMA of probabilities + hysteresis on the per-class thresholds."""

    def __init__(self, labels: list[str], thresholds: dict[str, float], ema_alpha: float,
                 hysteresis: float) -> None:
        self.labels = list(labels)
        self.on = np.array([thresholds[lab] for lab in self.labels], dtype=np.float64)
        self.off = np.maximum(self.on - hysteresis, 0.0)
        self.alpha = float(ema_alpha)
        self.reset()

    def reset(self) -> None:
        self.p: np.ndarray | None = None
        self.active = np.zeros(len(self.labels), dtype=bool)

    def update(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = np.asarray(probs, dtype=np.float64)
        self.p = probs if self.p is None else self.alpha * probs + (1 - self.alpha) * self.p
        self.active = np.where(self.active, self.p >= self.off, self.p >= self.on)
        return self.p.copy(), self.active.copy()


# --------------------------------------------------------------------------------------------
# Offline


@dataclass
class Timeline:
    duration_s: float
    hop_s: float
    window_s: float
    labels: list[str]
    times: list[float]  # end of each window
    probabilities: dict[str, list[float]]  # smoothed, per window
    segments: list[Detection] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"duration_s": self.duration_s, "hop_s": self.hop_s, "window_s": self.window_s,
                "labels": self.labels, "times": self.times, "probabilities": self.probabilities,
                "segments": [{"label": s.label, "probability": s.probability,
                              "t_start": s.t_start, "t_end": s.t_end} for s in self.segments]}


def window_ends(num_samples: int, sample_rate: int, window_s: float, hop_s: float) -> list[int]:
    """End sample of every analysis window: the first one is full, then every hop."""
    win = int(round(window_s * sample_rate))
    hop = max(1, int(round(hop_s * sample_rate)))
    if num_samples < win:
        return [num_samples] if num_samples > 0 else []
    return list(range(win, num_samples + 1, hop))


def detect_timeline(core: DetectionCore, audio: AudioBuffer, *, hop_s: float, ema_alpha: float,
                    hysteresis: float, batch: int = 32) -> Timeline:
    """Offline detection over a whole buffer (any rate/channels)."""
    mono = ChannelReducer("mean").process(audio) if audio.num_channels > 1 else audio
    x = resample_array(mono.samples, audio.sample_rate, core.sample_rate).ravel()
    win = core.window_samples
    ends = window_ends(x.shape[0], core.sample_rate, core.window_s, hop_s)
    windows = np.zeros((len(ends), win), dtype=np.float32)
    for i, end in enumerate(ends):
        seg = x[max(0, end - win):end]
        windows[i, win - seg.shape[0]:] = seg
    raw = np.concatenate([core.predict(windows[i:i + batch])
                          for i in range(0, len(windows), batch)]) if len(windows) else \
        np.zeros((0, len(core.labels)), np.float32)
    smoother = Smoother(core.labels, core.thresholds, ema_alpha, hysteresis)
    probs, active = [], []
    for row in raw:
        p, a = smoother.update(row)
        probs.append(p)
        active.append(a)
    times = [round(e / core.sample_rate, 3) for e in ends]
    segments = _segments(core.labels, times, probs, active, core.window_s)
    duration = audio.num_frames / audio.sample_rate
    return Timeline(round(duration, 3), hop_s, core.window_s, core.labels, times,
                    {lab: [round(float(p[j]), 4) for p in probs]
                     for j, lab in enumerate(core.labels)}, segments)


def _segments(labels, times, probs, active, window_s) -> list[Detection]:
    """Active runs per class -> ``Detection(t_start = first window start, t_end = last end)``.

    A run starts one window before its first active frame, so after a short off-gap it can begin
    before the previous run ended; such runs are merged (no overlapping segments of one class).
    """
    out: list[Detection] = []
    for j, lab in enumerate(labels):
        runs: list[tuple[float, float, list[float]]] = []
        start, acc = None, []
        for i, t in enumerate(times):
            if active[i][j]:
                if start is None:
                    start = max(0.0, t - window_s)
                acc.append(float(probs[i][j]))
            if start is not None and (not active[i][j] or i == len(times) - 1):
                end = t if active[i][j] else times[i - 1]
                if runs and start <= runs[-1][1]:
                    prev_start, _, prev_acc = runs.pop()
                    start, acc = prev_start, prev_acc + acc
                runs.append((start, end, acc))
                start, acc = None, []
        out += [Detection(t=a, label=lab, probability=round(float(np.mean(ps)), 4),
                          t_start=round(a, 3), t_end=round(b, 3)) for a, b, ps in runs]
    out.sort(key=lambda d: (d.t_start, d.label))
    return out


# --------------------------------------------------------------------------------------------
# Live


@dataclass
class _Job:
    t: float
    window: np.ndarray  # native rate, mono


class Detector(SideProcessor):
    """Live detection: ring buffer + hop scheduling + one inference thread (skip when busy)."""

    def __init__(self, core: DetectionCore, *, hop_s: float, ema_alpha: float,
                 hysteresis: float, warm_up: bool = True, cpu_threads: int | None = None
                 ) -> None:
        self.core = core
        self.hop_s = float(hop_s)
        self.smoother = Smoother(core.labels, core.thresholds, ema_alpha, hysteresis)
        self.warm_up_on_start = warm_up
        self.cpu_threads = cpu_threads
        self.skipped = 0
        self.windows_done = 0
        self._latencies: deque[float] = deque(maxlen=50)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._job: _Job | None = None
        self._busy = False
        self._results: list[tuple[float, np.ndarray]] = []
        self._thread: threading.Thread | None = None
        self._warmed = threading.Event()
        self._stopping = False
        self._sr = 0
        self._ring = np.zeros(0, dtype=np.float32)
        self._filled = 0
        self._pos = 0
        self._samples_seen = 0
        self._next_hop = 0
        self._reducer = ChannelReducer("mean")

    # -- lifecycle ---------------------------------------------------------------------------

    def configure(self, sample_rate: int, num_channels: int) -> None:
        super().configure(sample_rate, num_channels)
        self._sr = int(sample_rate)
        self._win = int(round(self.core.window_s * sample_rate))
        self._hop = max(1, int(round(self.hop_s * sample_rate)))
        if num_channels > 1:
            self._reducer.configure(sample_rate, num_channels)
        self.reset()
        self._start_thread()
        # The warm-up runs on the inference thread itself: CUDA (cuDNN/cuBLAS) sets up per-thread
        # state on a thread's first call, ~3 s on the laptop. Warming up on the session thread left
        # that cost to the first live window, and the next 12 hops were skipped.
        if self.warm_up_on_start and not self._warmed.wait(WARM_UP_TIMEOUT_S):
            log.warning("detector warm-up did not finish within %.0f s", WARM_UP_TIMEOUT_S)

    def reset(self) -> None:
        with self._lock:
            self._ring = np.zeros(max(1, getattr(self, "_win", 1)), dtype=np.float32)
            self._filled = self._pos = self._samples_seen = 0
            self._next_hop = getattr(self, "_win", 0)
            self._job, self._results = None, []
        self.smoother.reset()

    def close(self) -> None:
        thread = self._thread
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._thread = None

    def _start_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping = False
        self._warmed = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="soundspy-detector",
                                        daemon=True)
        self._thread.start()

    # -- audio side --------------------------------------------------------------------------

    def observe(self, chunk: AudioBuffer) -> list[Event]:
        mono = self._reducer.process(chunk) if chunk.num_channels > 1 else chunk
        x = mono.samples.ravel()
        i = 0
        while i < x.shape[0]:
            # write up to the next hop boundary, so windows end exactly on hop multiples
            take = min(x.shape[0] - i, self._next_hop - self._samples_seen)
            if take > 0:
                self._write(x[i:i + take])
                i += take
                self._samples_seen += take
            if self._samples_seen >= self._next_hop:
                self._submit(self._samples_seen / self._sr)
                self._next_hop += self._hop
        return self._drain()

    def _write(self, x: np.ndarray) -> None:
        n, size = x.shape[0], self._ring.shape[0]
        if n >= size:
            self._ring[:] = x[-size:]
            self._pos = 0
        else:
            end = self._pos + n
            if end <= size:
                self._ring[self._pos:end] = x
            else:
                first = size - self._pos
                self._ring[self._pos:] = x[:first]
                self._ring[:n - first] = x[first:]
            self._pos = end % size
        self._filled = min(size, self._filled + n)

    def _window(self) -> np.ndarray:
        return np.concatenate([self._ring[self._pos:], self._ring[:self._pos]])

    def _submit(self, t: float) -> None:
        with self._cond:
            if self._busy or self._job is not None:
                self.skipped += 1
                return
            self._job = _Job(t, self._window())
            self._cond.notify()

    def _drain(self) -> list[Event]:
        with self._lock:
            results, self._results = self._results, []
        events: list[Event] = []
        for t, probs in results:
            p, active = self.smoother.update(probs)
            labels = self.core.labels
            events.append(DetectionFrame(
                t=round(t, 3),
                probabilities={lab: round(float(min(1.0, max(0.0, p[j]))), 4)
                               for j, lab in enumerate(labels)},
                active=[lab for j, lab in enumerate(labels) if active[j]]))
        return events

    # -- inference thread --------------------------------------------------------------------

    def _worker(self) -> None:
        if self.cpu_threads and str(self.core.device) == "cpu":
            # Per-thread under OpenMP: only this thread is limited, offline `detect` is not.
            # With all cores, PyTorch's idle threads busy-wait between windows (~10 cores).
            import torch

            torch.set_num_threads(self.cpu_threads)
        if self.warm_up_on_start:
            try:
                self.core.warm_up()
            except Exception:  # noqa: BLE001 - a failed warm-up only costs the first window
                log.exception("detector warm-up failed")
        self._warmed.set()
        while True:
            with self._cond:
                while self._job is None and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                job, self._job, self._busy = self._job, None, True
            try:
                started = time.perf_counter()
                window = resample_array(job.window.reshape(-1, 1), self._sr,
                                        self.core.sample_rate).ravel()
                n = self.core.window_samples
                if window.shape[0] != n:
                    window = np.pad(window, (max(0, n - window.shape[0]), 0))[-n:]
                probs = self.core.predict(window[None, :])[0]
                elapsed = time.perf_counter() - started
                with self._lock:
                    self._latencies.append(elapsed)
                    self._results.append((job.t, probs))
                    self.windows_done += 1
            except Exception:  # noqa: BLE001 - keep the session alive; counted as skipped
                log.exception("detector inference failed")
                with self._lock:
                    self.skipped += 1
            finally:
                with self._lock:
                    self._busy = False

    # -- status ------------------------------------------------------------------------------

    def status_info(self) -> dict[str, Any]:
        with self._lock:
            lat = list(self._latencies)
        return {"model": self.core.name, "device": str(self.core.device), "hop_s": self.hop_s,
                "latency_ms": round(float(np.mean(lat)) * 1000, 2) if lat else None,
                "skipped": int(self.skipped)}


def build_detector(config, *, checkpoint: Path | None = None, device: str | None = None
                   ) -> Detector:
    """A ``Detector`` from the detection config. Raises ``DetectorUnavailable`` with a reason."""
    det = config.detection
    path = checkpoint if checkpoint is not None else det.checkpoint
    if path is None:
        raise DetectorUnavailable("no detection checkpoint is configured "
                                  "(detection.checkpoint in configs/detection/default.yaml)")
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise DetectorUnavailable("PyTorch is not installed (pip install -e \".[ml]\")") from exc
    core = load_core(Path(path), device or det.live.device)
    return Detector(core, hop_s=det.live.hop_s, ema_alpha=det.live.ema_alpha,
                    hysteresis=det.live.hysteresis, cpu_threads=det.live.cpu_threads)


__all__ = ["DetectionCore", "Detector", "DetectorUnavailable", "Smoother", "Timeline",
           "build_detector", "checkpoint_for", "detect_file", "detect_timeline", "load_core",
           "window_ends"]


def checkpoint_for(config, experiment: str | None = None) -> Path | None:
    """``experiments/<experiment>/checkpoints/best.pt``, or ``detection.checkpoint``."""
    if experiment:
        return Path(config.paths["experiments"]) / experiment / "checkpoints" / "best.pt"
    return config.detection.checkpoint


def detect_file(config, path: Path, *, experiment: str | None = None, device: str | None = None
                ) -> Timeline:
    """Offline detection of a WAV file with the configured (or named) experiment's model."""
    import soundfile as sf

    checkpoint = checkpoint_for(config, experiment)
    if checkpoint is None:
        raise DetectorUnavailable("no detection checkpoint is configured")
    core = load_core(checkpoint, device or config.detection.live.device)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    live = config.detection.live
    return detect_timeline(core, AudioBuffer(data, int(sr)), hop_s=live.hop_s,
                           ema_alpha=live.ema_alpha, hysteresis=live.hysteresis)
