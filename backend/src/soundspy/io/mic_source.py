"""``MicSource``: live microphone input through ``sounddevice``.

One ``sd.InputStream`` runs a callback on PortAudio's own thread. The callback does as little as
possible: it keeps the profile's ``raw_channels`` (device channels, a hardware selection rather
than a downmix), then either

- copies into the preallocated buffer of a running ``read()``, or
- appends to a pending block and cuts fixed-size chunks into a bounded queue for ``chunks()``.
  When the queue is full the **oldest** chunk is dropped and counted: for live audio, fresh sound
  is worth more than stale sound.

If no audio arrives for ``stall_timeout_s`` (default 2 s) the waiting call raises
``MicStallError`` instead of hanging. ``close()`` and leaving ``chunks()`` always stop the stream.

**Start-up silence.** A freshly opened WASAPI stream on the development laptop delivers exactly
500 ms of digital zeros before real audio. A real microphone never produces exact zeros (there is
always some hiss), so after ``open()`` leading all-zero frames are discarded, up to
``warmup_max_s`` (default 1 s) so that a genuinely muted microphone still delivers its silence
and gets reported as muted. The count is in ``stats.warmup_frames_discarded``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Any

import numpy as np

from soundspy.core.audio import AudioBuffer
from soundspy.io.base import AudioSource, frames_for
from soundspy.io.devices import ResolvedInput, _sd


class MicStallError(RuntimeError):
    """The microphone stream is open but delivers no audio."""


class MicSource(AudioSource):
    def __init__(
        self,
        resolved: ResolvedInput,
        *,
        max_queued_chunks: int = 32,
        stall_timeout_s: float = 2.0,
        blocksize: int = 0,
        warmup_max_s: float = 1.0,
    ) -> None:
        super().__init__()
        if max_queued_chunks < 1:
            raise ValueError("max_queued_chunks must be >= 1")
        self.resolved = resolved
        self.sample_rate = resolved.sample_rate
        self.num_channels = resolved.num_channels
        self.max_queued_chunks = max_queued_chunks
        self.stall_timeout_s = stall_timeout_s
        self.blocksize = blocksize
        self.warmup_max_s = warmup_max_s
        self._warmup_left = 0

        self._keep = np.asarray(resolved.channel_indices, dtype=np.intp)
        self._cond = threading.Condition()
        self._stream: Any = None
        self._closed = True
        self._last_audio = 0.0
        # read() state
        self._read_buf: np.ndarray | None = None
        self._read_filled = 0
        # chunks() state
        self._chunk_frames = 0
        self._pending = np.empty((0, self.num_channels), dtype=np.float32)
        self._queue: deque[np.ndarray] = deque()

    # --- lifecycle -----------------------------------------------------------------------------

    def open(self) -> None:
        if self._stream is not None:
            return
        sd = _sd()
        extra = None
        if self.resolved.wasapi_auto_convert and hasattr(sd, "WasapiSettings"):
            extra = sd.WasapiSettings(auto_convert=True)
        stream = sd.InputStream(
            device=self.resolved.device.index,
            channels=self.resolved.open_channels,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
            extra_settings=extra,
        )
        with self._cond:
            self._closed = False
            self._last_audio = time.monotonic()
            self._warmup_left = round(self.warmup_max_s * self.sample_rate)
            self._stream = stream
        stream.start()

    def close(self) -> None:
        with self._cond:
            stream, self._stream = self._stream, None
            self._closed = True
            self._cond.notify_all()
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    # --- PortAudio callback (runs on the audio thread) -------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        data = indata[:, self._keep]  # profile's raw_channels; fancy indexing copies
        with self._cond:
            if status is not None and getattr(status, "input_overflow", False):
                self.stats.overflows += 1
            if frames:
                self._last_audio = time.monotonic()
            if self._warmup_left:
                nonzero = np.flatnonzero(data.any(axis=1))
                leading = int(nonzero[0]) if nonzero.size else data.shape[0]
                drop = min(leading, self._warmup_left)
                self.stats.warmup_frames_discarded += drop
                self._warmup_left = 0 if nonzero.size else self._warmup_left - drop
                data = data[drop:]
                frames = data.shape[0]
                if not frames:
                    return
            if self._read_buf is not None:
                n = min(frames, self._read_buf.shape[0] - self._read_filled)
                self._read_buf[self._read_filled : self._read_filled + n] = data[:n]
                self._read_filled += n
                self._cond.notify_all()
            elif self._chunk_frames:
                self._pending = np.concatenate((self._pending, data))
                while self._pending.shape[0] >= self._chunk_frames:
                    chunk = self._pending[: self._chunk_frames]
                    self._pending = self._pending[self._chunk_frames :]
                    if len(self._queue) >= self.max_queued_chunks:
                        self._queue.popleft()
                        self.stats.dropped_chunks += 1
                    self._queue.append(np.ascontiguousarray(chunk))
                self._cond.notify_all()

    # --- consumers -----------------------------------------------------------------------------

    def _metadata(self) -> dict[str, Any]:
        device = self.resolved.device
        return {
            "source": "mic",
            "device": device.name,
            "device_index": device.index,
            "hostapi": device.hostapi,
            "profile": self.resolved.profile_name,
            "channel_indices": list(self.resolved.channel_indices),
        }

    def _check_alive(self) -> None:
        """Called with the condition held while waiting for audio."""
        if self._closed:
            raise RuntimeError("the microphone stream was closed")
        silent_for = time.monotonic() - self._last_audio
        if silent_for > self.stall_timeout_s:
            raise MicStallError(
                f"no audio from {self.resolved.device.name!r} ({self.resolved.device.hostapi}) "
                f"for {silent_for:.1f} s - is another app using the microphone, or is microphone "
                "access disabled in Windows privacy settings?"
            )

    def read(self, seconds: float) -> AudioBuffer:
        n = frames_for(seconds, self.sample_rate)
        if self._stream is None:
            raise RuntimeError("MicSource.read() needs an open stream (use `with MicSource(...)`)")
        buf = np.empty((n, self.num_channels), dtype=np.float32)
        with self._cond:
            overflows_before = self.stats.overflows
            self._last_audio = time.monotonic()
            self._read_buf, self._read_filled = buf, 0
            try:
                while self._read_filled < n:
                    self._check_alive()
                    self._cond.wait(timeout=0.1)
            finally:
                self._read_buf = None
            overflows = self.stats.overflows - overflows_before
            self.stats.frames_delivered += n
        meta = {
            **self._metadata(),
            "overflows": overflows,
            "warmup_frames_discarded": self.stats.warmup_frames_discarded,
        }
        return AudioBuffer(buf, self.sample_rate, meta)

    def chunks(self, chunk_frames: int) -> Iterator[AudioBuffer]:
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1")
        if self._stream is None:
            self.open()
        with self._cond:
            self._queue.clear()
            self._pending = np.empty((0, self.num_channels), dtype=np.float32)
            self._chunk_frames = chunk_frames
            self._last_audio = time.monotonic()
        meta = self._metadata()
        index = 0
        try:
            while True:
                with self._cond:
                    while not self._queue:
                        if self._closed:
                            return
                        self._check_alive()
                        self._cond.wait(timeout=0.1)
                    data = self._queue.popleft()
                    self.stats.frames_delivered += data.shape[0]
                yield AudioBuffer(data, self.sample_rate, {**meta, "chunk_index": index})
                index += 1
        finally:
            with self._cond:
                self._chunk_frames = 0
            self.close()
