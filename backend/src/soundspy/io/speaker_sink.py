"""``SpeakerSink``: plays processed audio on an output device (live monitoring).

The input (microphone) and output (speakers/headphones) are two separate streams with two
separate clocks, so they never line up perfectly. Between them sits a small queue of chunks:

- ``write(chunk)`` (session worker thread) applies **gain**, **upmixes** 1 channel to every output
  channel, runs the **safety limiter** and appends the chunk to the queue. Above
  ``max_queue_chunks`` the **oldest** chunk is dropped and counted: latency stays bounded even if
  the input clock runs slightly fast.
- The PortAudio **callback** (audio thread) pulls frames from the queue. Playback starts only once
  ``prebuffer_chunks`` are queued (a jitter buffer). If the queue runs dry while playing, the
  missing frames are silence and an **underrun** is counted; playback then waits for the
  prebuffer again rather than crackling chunk by chunk.
- **Draining (step 07b).** After a stall the chunks that piled up arrive in a burst and the
  queue fills. Input and output then run at the same speed again, so nothing would ever empty
  it: the latency stayed at its maximum until the session was restarted. Now, when the queue has
  been above ``prebuffer_chunks + 1`` for ``drain_after_s`` without a break, the oldest chunks
  are dropped down to the prebuffer level and the event is counted (``drained``). The jump is
  hidden by a ``drain_fade_ms`` crossfade from the audio that would have played next into the
  newer audio that plays instead, so there is no click.

**Safety limiter.** The laptop microphone can deliver samples up to +7.44 dBFS (Windows adds gain
after the converter), and the user can add up to +12 dB. Before anything reaches the speakers, a
peak limiter with instant attack and a ``limiter_release_ms`` release keeps every sample below
``limiter_ceiling_dbfs`` (a final hard clip catches rounding). It is always on.

Input with more channels than the output is an error: the chain must reduce channels first
(``ChannelReducer``); only upmixing 1 -> N happens here.

**Device lost (step 08).** Once the device has called back, a gap of ``stall_timeout_s`` without
callbacks means it is gone (headphones or a USB device unplugged mid-session): the next
``write`` raises ``OutputLostError`` with a message for the user, and the session ends in its
error state instead of queueing into nothing. **Stream options (step 08):** ``output.latency``
(PortAudio's suggested latency) and ``output.exclusive`` (WASAPI exclusive mode).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from soundspy.config import OutputConfig
from soundspy.core.audio import AudioBuffer
from soundspy.io.base import AudioSink, valid_frames
from soundspy.io.devices import OutputDevice, _sd


class OutputLostError(RuntimeError):
    """The output device stopped calling back while the session ran (unplugged?)."""


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


class PeakLimiter:
    """Instant-attack, exponential-release peak limiter (shared gain across channels)."""

    def __init__(self, sample_rate: int, ceiling_dbfs: float, release_ms: float) -> None:
        self.ceiling = db_to_gain(ceiling_dbfs)
        self.release = math.exp(-1.0 / (sample_rate * release_ms / 1000.0))
        self.env = 1.0  # current gain reduction (1 = none)

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(x), axis=1)
        if self.env >= 1.0 and float(peak.max(initial=0.0)) <= self.ceiling:
            return x  # fast path: nothing to limit and no release in progress
        need = np.minimum(1.0, self.ceiling / np.maximum(peak, 1e-12))
        gains = np.empty_like(need)
        env, release = self.env, self.release
        for i, target in enumerate(need):
            env = target if target < env else 1.0 - (1.0 - env) * release
            if env > target:
                env = target
            gains[i] = env
        self.env = env
        out = x * gains.reshape(-1, 1).astype(np.float32)  # same gain for every channel
        return np.clip(out, -self.ceiling, self.ceiling)


class SpeakerSink(AudioSink):
    def __init__(self, device: OutputDevice, output: OutputConfig, *, blocksize: int = 0,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.device = device
        self.config = output
        self.blocksize = blocksize
        self.gain_db = output.default_gain_db
        self.muted = False
        self.underruns = 0
        self.dropped_chunks = 0
        self.drained = 0  # times the queue was emptied back to the prebuffer level
        self.drained_chunks = 0
        self._clock = clock
        self._above_since: float | None = None  # since when the queue is above its cushion
        self.frames_played = 0
        self.sample_rate: int | None = None
        self.in_channels: int | None = None
        self.out_channels: int | None = None
        self._lock = threading.Lock()
        self._queue: deque[np.ndarray] = deque()
        self._current: np.ndarray | None = None
        self._offset = 0
        self._playing = False
        self._stream: Any = None
        self._limiter: PeakLimiter | None = None
        self._last_callback: float | None = None  # clock() of the device's last callback

    # --- lifecycle -----------------------------------------------------------------------------

    def open(self, sample_rate: int, num_channels: int) -> None:
        if self._stream is not None:
            raise RuntimeError("SpeakerSink is already open")
        out_channels = min(self.device.max_channels, 2)
        if num_channels > out_channels:
            raise ValueError(
                f"{num_channels}-channel audio cannot be played on {self.device.name!r} "
                f"({out_channels} channels): reduce channels in the chain first"
            )
        sd = _sd()
        extra = None
        if self.device.hostapi == "Windows WASAPI" and hasattr(sd, "WasapiSettings"):
            extra = (sd.WasapiSettings(exclusive=True) if self.config.exclusive
                     else sd.WasapiSettings(auto_convert=True))
        kwargs: dict[str, Any] = {}
        if self.config.latency is not None:
            kwargs["latency"] = self.config.latency
        self.sample_rate, self.in_channels, self.out_channels = (
            sample_rate,
            num_channels,
            out_channels,
        )
        self._limiter = PeakLimiter(
            sample_rate, self.config.limiter_ceiling_dbfs, self.config.limiter_release_ms
        )
        stream = sd.OutputStream(
            device=self.device.index,
            channels=out_channels,
            samplerate=sample_rate,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
            extra_settings=extra,
            **kwargs,
        )
        self._stream = stream
        self._last_callback = None
        stream.start()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        self.flush()

    def flush(self) -> None:
        """Forget queued audio and wait for the prebuffer again (no underrun is counted)."""
        with self._lock:
            self._queue.clear()
            self._current, self._offset, self._playing = None, 0, False
            self._above_since = None

    # --- producer side (session worker) --------------------------------------------------------

    def set_gain(self, gain_db: float, muted: bool = False) -> None:
        self.gain_db, self.muted = gain_db, muted

    def write(self, chunk: AudioBuffer) -> None:
        if self._stream is None or self._limiter is None:
            raise RuntimeError("SpeakerSink.open() must be called before write()")
        if chunk.num_channels != self.in_channels:
            raise ValueError(
                f"chunk has {chunk.num_channels} channel(s), the sink was opened for "
                f"{self.in_channels}"
            )
        last = self._last_callback
        if last is not None and self._clock() - last > self.config.stall_timeout_s:
            raise OutputLostError(
                f"the output device {self.device.name!r} stopped playing (unplugged, or switched "
                "off in Windows?) - plug it back in or choose another output, then press "
                "Restart")
        n = valid_frames(chunk)
        if n == 0:
            return
        data = chunk.samples[:n]
        if self.muted:
            data = np.zeros_like(data)
        else:
            data = data * np.float32(db_to_gain(self.gain_db))
        if data.shape[1] == 1 and self.out_channels and self.out_channels > 1:
            data = np.repeat(data, self.out_channels, axis=1)  # upmix 1 -> N
        data = self._limiter.process(data.astype(np.float32, copy=False))
        with self._lock:
            self._queue.append(np.ascontiguousarray(data, dtype=np.float32))
            while len(self._queue) > self.config.max_queue_chunks:
                self._queue.popleft()
                self.dropped_chunks += 1
            self._drain_if_stuck()
            if not self._playing and len(self._queue) >= self.config.prebuffer_chunks:
                self._playing = True

    def _drain_if_stuck(self) -> None:
        """Called with the lock held, after a chunk was queued (see the module docstring)."""
        cfg = self.config
        if len(self._queue) <= cfg.prebuffer_chunks + 1:
            self._above_since = None
            return
        now = self._clock()
        if self._above_since is None:
            self._above_since = now
            return
        if now - self._above_since < cfg.drain_after_s:
            return
        old = self._queue[0]  # what would have played next
        dropped = 0
        while len(self._queue) > max(1, cfg.prebuffer_chunks):
            self._queue.popleft()
            dropped += 1
        new = self._queue[0]
        fade = min(old.shape[0], new.shape[0],
                   round(cfg.drain_fade_ms * (self.sample_rate or 0) / 1000.0))
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32).reshape(-1, 1)
            head = new.copy()
            head[:fade] = (1.0 - ramp) * old[:fade] + ramp * new[:fade]
            self._queue[0] = head
        self.drained += 1
        self.drained_chunks += dropped
        self._above_since = None

    # --- consumer side (PortAudio callback) ----------------------------------------------------

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        self._last_callback = self._clock()
        outdata.fill(0.0)
        with self._lock:
            if not self._playing:
                return
            filled = 0
            while filled < frames:
                if self._current is None or self._offset >= self._current.shape[0]:
                    if not self._queue:
                        self.underruns += 1
                        self._playing = False  # re-prebuffer instead of crackling
                        self._current, self._offset = None, 0
                        break
                    self._current, self._offset = self._queue.popleft(), 0
                take = min(frames - filled, self._current.shape[0] - self._offset)
                outdata[filled : filled + take] = self._current[self._offset : self._offset + take]
                self._offset += take
                filled += take
            self.frames_played += filled

    # --- latency -------------------------------------------------------------------------------

    @property
    def queued_frames(self) -> int:
        with self._lock:
            queued = sum(c.shape[0] for c in self._queue)
            if self._current is not None:
                queued += self._current.shape[0] - self._offset
        return queued

    @property
    def latency_s(self) -> float:
        """Output stream latency plus the audio waiting in the queue."""
        stream = self._stream
        try:
            stream_latency = float(getattr(stream, "latency", 0.0)) if stream is not None else 0.0
        except (TypeError, ValueError):
            stream_latency = 0.0
        sr = self.sample_rate or 1
        return stream_latency + self.queued_frames / sr
