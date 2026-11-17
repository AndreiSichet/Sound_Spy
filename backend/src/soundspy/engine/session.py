"""``Session``: one running processing session.

A session ties together an ``AudioSource``, a ``ProcessorChain``, an optional output sink
(``SpeakerSink`` for live monitoring) and an optional recording, and runs them on one **worker
thread**::

    source.chunks(chunk_frames)  ──►  chain.process(chunk)  ──►  output.write(out)  (if monitoring)
            │                               │                        (or chain.dry when bypassed)
            └─► recorder.write(chunk)       └─► events ──► on_event(...)  (levels, waveform, ...)
               (the RAW input chunk)

- **Changes arrive between chunks.** ``set_target``, ``set_monitor`` and ``set_bypass`` only
  store the request; the worker applies it before processing the next chunk, so a chunk is never
  processed with a half-changed setting. Recording start/stop take a lock the worker also holds
  while writing.
- **Bypass (A/B listening).** With bypass on, the speaker gets the chain's *dry* signal (the
  input after channel reduction, delayed exactly as much as the processing delays the output)
  instead of the output. Processing keeps running and both signals are time-aligned, so
  switching is instant and seamless.
- **Timing.** Every chunk's processing time is measured. The real-time factor (RTF) is processing
  time ÷ chunk duration: 1.0 means the computer only just keeps up. ``status`` reports mean and
  p95 over the last ``engine.rtf_window_s``; a ``WarningEvent`` is emitted when p95 RTF exceeds
  ``engine.rtf_warning`` (at most once per window).
- **Errors** in the worker (device lost, microphone stall, a failing stage) set
  ``state="error"`` with a message, emit a status event, release everything and end the thread.
  They never propagate to the caller's process.
- ``stop()`` is idempotent: it closes the source (which unblocks the worker), joins the thread,
  closes the output and finishes an active recording.
- **Garbage collection.** A full collection scans every long-lived object in the process (the
  model, PyTorch, the web framework: ~140,000 objects, 45-75 ms on the laptop) and Python stops
  *all* threads meanwhile, the audio thread included. That is longer than the output queue's
  cushion, so it caused an occasional underrun. While a session runs, everything allocated
  before it is therefore **frozen** (``gc.freeze()``: kept out of the collector's scans), which
  makes a full collection take about a millisecond; ``gc.unfreeze()`` when the session ends.

``on_event`` is called **on the worker thread**; the API layer hands events to its event loop.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import AppConfig
from soundspy.core.events import EngineState, EngineStatus, Event, WarningEvent
from soundspy.core.target import TargetSpec
from soundspy.engine.chain import ProcessorChain
from soundspy.io.base import AudioSink, AudioSource
from soundspy.io.storage import RecordingWriter

logger = logging.getLogger(__name__)

JOIN_TIMEOUT_S = 5.0  # generous upper bound for the worker to notice a stop


class SessionError(RuntimeError):
    """An invalid request for the session's current state (e.g. recording twice)."""


class Session:
    def __init__(
        self,
        source: AudioSource,
        chain: ProcessorChain,
        *,
        output: AudioSink | None,
        on_event: Callable[[Event], None],
        config: AppConfig,
        mode: str = "semantic",
    ) -> None:
        self._source = source
        self._chain = chain
        self._output = output
        self._on_event = on_event
        self._engine = config.engine
        self._mode = mode

        self._state: EngineState = "idle"
        self._message = ""
        self._lock = threading.Lock()  # state + pending changes
        self._rec_lock = threading.Lock()  # recorder (worker writes while holding it)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._released = False
        self._frozen = False

        self._pending_target: TargetSpec | None = None
        self._pending_monitor: tuple[bool, float, bool] | None = None
        self._pending_bypass: bool | None = None
        self.bypass = False
        self.target: TargetSpec | None = None
        self.monitoring = False
        self.gain_db = float(getattr(output, "gain_db", 0.0) or 0.0)
        self.muted = False

        self._recorder: RecordingWriter | None = None
        self._rec_overflows_at_start = 0
        self.last_recording: Path | None = None

        self._chunk_frames = 0
        self._chunk_s = 0.0
        self._out_channels = 0
        self.chunks_processed = 0
        self._times: deque[float] = deque()
        self._last_warning = -1e9
        self._last_status = 0.0

    # --- lifecycle -----------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._state == "running"

    @property
    def state(self) -> EngineState:
        return self._state

    def start(self) -> None:
        if self._thread is not None:
            raise SessionError("a session can only be started once")
        source = self._source
        source.open()
        try:
            sr, ch = source.sample_rate, source.num_channels
            self._out_channels = self._chain.configure(sr, ch)
            self._chunk_frames = self._engine.chunk_frames(sr)
            self._chunk_s = self._chunk_frames / sr
            window = max(1, round(self._engine.rtf_window_s / self._chunk_s))
            self._times = deque(maxlen=window)
            output_problem = None
            if self._output is not None:
                try:
                    self._output.open(sr, self._out_channels)
                except Exception as exc:  # noqa: BLE001 - monitoring is optional
                    logger.warning("output unavailable: %s", exc)
                    output_problem = f"Listening is unavailable for this session: {exc}"
                    self._message = output_problem
                    self._output = None
        except Exception:
            source.close()
            raise
        gc.collect()  # now, before audio flows ...
        gc.freeze()  # ... and nothing allocated so far is scanned again during the session
        self._frozen = True
        self._started_at = time.monotonic()
        self._state = "running"
        self._thread = threading.Thread(target=self._run, name="soundspy-session", daemon=True)
        self._thread.start()
        self._emit(self.status)
        if output_problem is not None:  # tell the user why Listen will not work
            self._emit(WarningEvent(t=0.0, code="output_unavailable", message=output_problem))
        for code, message in self._chain.warnings:  # e.g. no detection model
            self._emit(WarningEvent(t=0.0, code=code, message=message))

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        self._source.close()  # unblocks a worker waiting for audio
        if thread is not threading.current_thread():
            thread.join(timeout=JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.error("session worker did not stop within %.0f s", JOIN_TIMEOUT_S)
        self._release()
        with self._lock:
            if self._state == "running":
                self._state = "idle"
                self._message = "stopped"
        self._emit(self.status)

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _release(self) -> None:
        """Close the output and finish an active recording (idempotent)."""
        with self._lock:
            if self._released:
                return
            self._released = True
        if self._output is not None:
            try:
                self._output.close()
            except Exception:  # noqa: BLE001
                logger.exception("closing the output failed")
        with self._rec_lock:
            if self._recorder is not None:
                self.last_recording = self._finish_recording()
        try:
            self._chain.close()  # stops the detector's inference thread
        except Exception:  # noqa: BLE001
            logger.exception("closing the processing chain failed")
        if self._frozen:
            self._frozen = False
            gc.unfreeze()  # back to normal collection between sessions

    # --- control (any thread) ------------------------------------------------------------------

    def set_target(self, target: TargetSpec) -> None:
        with self._lock:
            self._pending_target = target
        if not self.running:
            self._apply_pending()

    def set_monitor(self, enabled: bool, gain_db: float, muted: bool = False) -> None:
        if enabled and self._output is None:
            raise SessionError("no output device is available for monitoring")
        with self._lock:
            self._pending_monitor = (enabled, float(gain_db), muted)
        if not self.running:
            self._apply_pending()

    def set_bypass(self, enabled: bool) -> None:
        """Play the unprocessed (delay-matched) signal instead of the chain output."""
        with self._lock:
            self._pending_bypass = bool(enabled)
        if not self.running:
            self._apply_pending()

    def start_recording(
        self, directory: str | Path, label: str | None = None, notes: str = ""
    ) -> None:
        if not self.running:
            raise SessionError("no running session to record")
        source = self._source
        with self._rec_lock:
            if self._recorder is not None:
                raise SessionError("already recording")
            meta = source.describe()
            self._recorder = RecordingWriter(
                directory,
                source.sample_rate,
                source.num_channels,
                label=label,
                notes=notes,
                metadata={k: v for k, v in meta.items() if v is not None},
            )
            self._rec_overflows_at_start = source.stats.overflows
        self._emit(self.status)

    def stop_recording(self) -> Path:
        with self._rec_lock:
            if self._recorder is None:
                raise SessionError("not recording")
            path = self._finish_recording()
        self.last_recording = path
        self._emit(self.status)
        return path

    def _finish_recording(self) -> Path:
        """Close the recorder (caller holds ``_rec_lock``)."""
        recorder, self._recorder = self._recorder, None
        assert recorder is not None
        recorder.overflows = self._source.stats.overflows - self._rec_overflows_at_start
        return recorder.close()

    def _apply_pending(self) -> None:
        with self._lock:
            target, self._pending_target = self._pending_target, None
            monitor, self._pending_monitor = self._pending_monitor, None
            bypass, self._pending_bypass = self._pending_bypass, None
        if bypass is not None:
            self.bypass = bypass
        if target is not None:
            self.target = target
            self._chain.set_target(target)
        if monitor is not None:
            enabled, gain_db, muted = monitor
            output = self._output
            if output is not None:
                if enabled != self.monitoring:
                    getattr(output, "flush", lambda: None)()  # fresh prebuffer either way
                set_gain = getattr(output, "set_gain", None)
                if set_gain is not None:
                    set_gain(gain_db, muted)
            self.monitoring = enabled and output is not None
            self.gain_db, self.muted = gain_db, muted

    # --- worker --------------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            # stage hooks that need this thread (a GPU model's warm-up); the source only starts
            # queueing audio when the loop below asks for the first chunk
            self._chain.start()
            for chunk in self._source.chunks(self._chunk_frames):
                if self._stop.is_set():
                    break
                self._apply_pending()
                started = time.perf_counter()
                out, events = self._chain.process(chunk)
                self._times.append(time.perf_counter() - started)
                self.chunks_processed += 1
                if self.monitoring and self._output is not None:
                    dry = self._chain.dry if self.bypass else None
                    self._output.write(dry if dry is not None else out)
                with self._rec_lock:
                    if self._recorder is not None:
                        self._recorder.write(chunk)  # the raw input, all channels
                for event in events:
                    self._emit(event)
                self._periodic()
            else:
                if not self._stop.is_set():
                    with self._lock:
                        self._state, self._message = "idle", "source finished"
        except Exception as exc:  # noqa: BLE001 - reported as state="error", never re-raised
            if not self._stop.is_set():
                logger.exception("session worker failed")
                with self._lock:
                    self._state, self._message = "error", f"{type(exc).__name__}: {exc}"
        if not self._stop.is_set():
            self._source.close()
            self._release()
            self._emit(self.status)

    def _periodic(self) -> None:
        now = time.monotonic()
        if now - self._last_status >= self._engine.status_interval_s:
            self._last_status = now
            self._emit(self.status)
        rtf_p95 = self._rtf_p95()
        if (
            rtf_p95 is not None
            and len(self._times) == self._times.maxlen
            and rtf_p95 > self._engine.rtf_warning
            and now - self._last_warning >= self._engine.rtf_window_s
        ):
            self._last_warning = now
            self._emit(
                WarningEvent(
                    t=self._elapsed(),
                    code="rtf_high",
                    message=(
                        f"processing is slow: p95 real-time factor {rtf_p95:.2f} "
                        f"(over {self._engine.rtf_warning:.2f}) - audio may start to drop"
                    ),
                )
            )

    def _emit(self, event: Event) -> None:
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001 - a broken listener must not stop the audio
            logger.exception("on_event callback failed")

    # --- status --------------------------------------------------------------------------------

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._started_at, 3) if self._started_at else 0.0

    def _rtf_p95(self) -> float | None:
        if not self._times or not self._chunk_s:
            return None
        return float(np.percentile(np.asarray(self._times), 95)) / self._chunk_s

    @property
    def latency_ms(self) -> float | None:
        if not self._chunk_s:
            return None
        latency = self._source.input_latency_s + self._chunk_s
        latency += self._chain.latency_samples / self._source.sample_rate  # e.g. the STFT
        if self.monitoring and self._output is not None:
            latency += float(getattr(self._output, "latency_s", 0.0))
        return round(latency * 1000, 1)

    @property
    def status(self) -> EngineStatus:
        source, output = self._source, self._output
        info: dict[str, Any] = source.describe()
        times = np.asarray(self._times) if self._times else None
        with self._rec_lock:
            recorder = self._recorder
            rec_seconds = recorder.duration_s if recorder is not None else 0.0
        rtf = self._rtf_p95()
        # A monitor change waits for the next chunk boundary; report what was asked for, so a
        # reply to the request already shows the new setting.
        with self._lock:
            pending = self._pending_monitor
            pending_bypass = self._pending_bypass
        if pending is not None:
            monitoring = pending[0] and output is not None
            gain_db, muted = pending[1], pending[2]
        else:
            monitoring, gain_db, muted = self.monitoring, self.gain_db, self.muted
        return EngineStatus(
            t=self._elapsed(),
            state=self._state,
            mode=self._mode,
            device=info.get("device"),
            sample_rate=source.sample_rate,
            channels=source.num_channels,
            message=self._message,
            source=info.get("source"),
            output_device=getattr(getattr(output, "device", None), "name", None),
            monitoring=monitoring,
            muted=muted,
            gain_db=gain_db,
            recording=recorder is not None,
            recording_seconds=round(rec_seconds, 2),
            latency_ms=self.latency_ms,
            chunk_ms=round(self._chunk_s * 1000, 2) if self._chunk_s else None,
            chunks_processed=self.chunks_processed,
            proc_ms_mean=round(float(times.mean()) * 1000, 3) if times is not None else None,
            proc_ms_p95=round(float(np.percentile(times, 95)) * 1000, 3)
            if times is not None
            else None,
            rtf_p95=round(rtf, 4) if rtf is not None else None,
            input_overflows=source.stats.overflows,
            output_underruns=int(getattr(output, "underruns", 0)),
            dropped_chunks=source.stats.dropped_chunks,
            output_dropped=int(getattr(output, "dropped_chunks", 0)),
            detector=self._detector_info(),
            bypass=self.bypass if pending_bypass is None else pending_bypass,
            extraction=self._extraction_info(),
        )

    def _extraction_info(self) -> dict[str, Any] | None:
        """``{method, latency_ms, gating, target_active}`` of the chain's extractor."""
        if not self._chain.has("extractor"):
            return None
        stage = self._chain.stage("extractor")
        info = getattr(stage, "status_info", None)
        if callable(info):
            return info()
        return {"method": "passthrough", "latency_ms": 0.0, "gating": False,
                "target_active": False}

    def _detector_info(self) -> dict[str, Any] | None:
        if not self._chain.has("detector"):
            return None
        stage = self._chain.stage("detector")
        info = getattr(stage, "status_info", None)
        return info() if callable(info) else None
