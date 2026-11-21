"""Acoustic loopback latency measurement (step 08, ``soundspy measure-latency``).

A click train is played through the **output device** while the **input device** records, in
one full-duplex PortAudio stream: playback and capture share one sample clock, so a click found
``d`` samples after it was written took ``d / sample_rate`` seconds from the program to the
speaker, through the air and back from the microphone (the **hardware round trip**: output
latency + input latency + the acoustic path, about 1 ms at laptop distances).

**Through the app** (``--through-app``): Sound Spy's own session runs at the same time
(microphone -> chain -> speaker). Every click then arrives twice at the microphone: directly
(``d1``) and once more after the app has played it back (``d2``). ``d2 - d1`` is the app's
microphone -> speaker delay (its input latency, chunking, processing, output queue and output
latency), plus one more acoustic path (~1 ms, negligible and not subtracted).

Finding a click: a **matched filter**, the cross-correlation of the recording with the click's
own waveform (a short Hann-windowed noise burst: sharp, broadband, unlike speech or music),
scaled by the click's energy so that it reads as the click's amplitude in the recording. Its
median over the silent lead-in before the first click is the room's noise floor (the whole
recording's median when the lead-in is too short).
A click counts when the strongest peak in the window after its playback position stands
``min_snr`` times (18 dB) above that floor; the arrival is the **earliest** lag in the window
that reaches ``onset`` (half, -6 dB) of that peak within ``lookback_ms`` (3 ms) before it, so a
strong reflection just after the direct sound does not move it (and noise earlier in the window
cannot).

**The app's copy** of a click is what the microphone heard, played once more through the
speaker: it has passed the speaker and the room twice and no longer looks like the click. So it
is searched with the **recorded direct arrival** as the template (``segment_ms`` from 1 ms
before it): the copy differs from that only by one more pass, like the direct arrival from the
played click. The lag of the best match is the app's delay itself; the search starts
``min_gap_s`` or half the hardware round trip after the direct arrival (the app's own streams
have at least that delay), past most of the direct click's reverberation.

*(Step 08 follow-up: the first version normalised the correlation by the recording's
local energy as well; that measures how alike the shapes are, not how loud the click is, and the
laptop speaker and the room change the shape of a 3 ms burst so much that real clicks scored
7-8, just under the threshold, at any volume.)*

**Coupling**: how much of the click comes back, in the **worst third-octave band** between 200 Hz
and 10 kHz (the recorded direct arrivals, 30 ms each, against the played click): the loop gain
that decides whether the app could howl is the largest over frequency. The recorded band powers
are averaged over every click found, the room noise in each band (measured in the silent lead-in
before the first click: median over 30 ms pieces, so one pop does not count) is taken off, and
only bands where the click has energy (within 10 dB of its strongest band) and the recording
stands 3 dB above the noise count. ``app_gain_db`` picks the app's playback gain from it so that
the loop speaker -> microphone -> app -> speaker stays ``LOOP_DB`` below unity (-20 ... +20 dB,
this measurement only). ``LOOP_DB`` is **-25 dB**, not the -10 dB of the first version: on the
laptop the loop through the app turned out about 13 dB stronger than this estimate (the
speaker's own processing: one loud click does not show it) and howled at +10 dB.

**Feedback guard**: while recording, ``play_and_record(abort_rise_db=...)`` stops the run when
the last second of the recording is that much louder than the silent lead-in (a loop building
up); ``feedback_db`` measures the rise afterwards (the median 20 ms level of the last 3 s against
the lead-in's), and the command repeats the run 10 dB quieter.

This module, with the rest of ``io/``, is the only place that touches ``sounddevice``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from soundspy.io.devices import _sd

CLICK_MS = 3.0
LOOP_DB = -25.0  # the loop gain speaker -> microphone -> app -> speaker during the app modes
FEEDBACK_DB = 10.0  # a recording this much louder at the end than in its lead-in: feedback
ABORT_RISE_DB = 25.0  # stop a run at once when its last second is this much louder
MAX_APP_GAIN_DB = 20.0  # this measurement only (the page's volume ends at +12 dB)
MIN_APP_GAIN_DB = -20.0


@dataclass(frozen=True)
class ClickTrain:
    signal: np.ndarray  # mono float32, the whole playback
    positions: tuple[int, ...]  # sample index of each click's start
    template: np.ndarray  # one click
    sample_rate: int


def click_train(sample_rate: int, *, clicks: int = 10, period_s: float = 1.0,
                lead_s: float = 0.5, tail_s: float = 1.0, level_dbfs: float = -6.0,
                click_ms: float = CLICK_MS, seed: int = 8) -> ClickTrain:
    """``clicks`` noise bursts ``period_s`` apart after ``lead_s`` of silence (the streams
    settle), then ``tail_s`` of silence (the last click's arrivals)."""
    n = max(8, int(round(click_ms * sample_rate / 1000.0)))
    burst = np.random.default_rng(seed).standard_normal(n) * np.hanning(n)
    burst = (burst / np.max(np.abs(burst)) * 10.0 ** (level_dbfs / 20.0)).astype(np.float32)
    lead, period = int(round(lead_s * sample_rate)), int(round(period_s * sample_rate))
    total = lead + clicks * period + int(round(tail_s * sample_rate))
    signal = np.zeros(total, dtype=np.float32)
    positions = tuple(lead + i * period for i in range(clicks))
    for p in positions:
        signal[p:p + n] = burst
    return ClickTrain(signal, positions, burst, int(sample_rate))


def _correlation(recorded: np.ndarray, template: np.ndarray) -> np.ndarray:
    """|matched filter| of ``recorded`` with ``template`` at every lag, scaled by the
    template's energy: the least-squares amplitude of the click at that lag (1.0 = the click as
    played)."""
    x = np.asarray(recorded, dtype=np.float64).ravel()
    t = np.asarray(template, dtype=np.float64).ravel()
    n = x.size + t.size - 1
    size = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(t, size)), size)[:x.size]
    return np.abs(corr) / max(float(t @ t), 1e-20)


def _db(ratio: float) -> float:
    return round(20.0 * float(np.log10(max(ratio, 1e-12))), 1)


def _matched(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """|matched filter| of ``x`` with ``template`` (valid lags only), scaled by its energy."""
    size = 1 << (x.size + template.size - 2).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(template, size)), size)
    return np.abs(corr[:max(0, x.size - template.size + 1)]) / max(float(template @ template),
                                                                    1e-20)


def _bands(n: int, sr: int) -> list[np.ndarray]:
    """Third-octave band masks (200 Hz - 10 kHz) over the bins of an ``n``-point rfft."""
    f = np.fft.rfftfreq(n, 1.0 / sr)
    out, centre = [], 200.0
    while centre <= 10000.0:
        sel = (f >= centre / 2 ** (1 / 6)) & (f < centre * 2 ** (1 / 6))
        if sel.any():
            out.append(sel)
        centre *= 2 ** (1 / 3)
    return out


def _band_power(x: np.ndarray, n: int, bands: list[np.ndarray]) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(x, n)) ** 2
    return np.array([float(spectrum[sel].sum()) for sel in bands])


def _noise_bands(noise: np.ndarray, length: int, n: int, bands: list[np.ndarray]) -> np.ndarray:
    """The room noise's power per band for a ``length``-sample piece: the median over the
    pieces of ``noise`` (zeros when there is too little of it)."""
    pieces = [noise[i:i + length] for i in range(0, noise.size - length + 1, length)]
    if not pieces:
        return np.zeros(len(bands))
    return np.median(np.stack([_band_power(piece, n, bands) for piece in pieces]), axis=0)


def _band_coupling_db(segments: list[np.ndarray], template: np.ndarray, noise: np.ndarray,
                      sr: int) -> float | None:
    """The largest third-octave band ratio (dB) of the ``segments`` (equal length; their band
    powers averaged, less the room noise) over ``template``, over the bands where the template
    has energy (within 10 dB of its strongest band) and the recording stands 3 dB above the
    noise; ``None`` without segments or usable bands."""
    if not segments:
        return None
    length = segments[0].size
    n = 1 << int(np.ceil(np.log2(max(length, template.size) * 2)))
    bands = _bands(n, sr)
    rec = np.mean(np.stack([_band_power(s, n, bands) for s in segments]), axis=0)
    played = _band_power(template, n, bands)
    room = _noise_bands(noise, length, n, bands)
    usable = (played > 0.1 * played.max()) & (rec >= 2.0 * room) & (rec > 0)
    if not usable.any():
        return None
    ratio = (rec[usable] - room[usable]) / played[usable]
    return round(10.0 * float(np.log10(max(float(ratio.max()), 1e-30))), 1)


def _onset(window: np.ndarray, onset: float, lookback: int) -> int:
    """The earliest lag, at most ``lookback`` samples before the strongest peak of ``window``,
    that reaches ``onset`` x that peak (the direct sound, when a reflection is stronger)."""
    top = int(np.argmax(window))
    lo = max(0, top - lookback)
    return lo + int(np.argmax(window[lo:top + 1] >= onset * float(window[top])))


@dataclass
class Arrivals:
    """Per click: the direct arrival and (through the app) the app's arrival, in samples after
    the click was played; ``None`` where none was found."""

    direct: list[int | None] = field(default_factory=list)
    app: list[int | None] = field(default_factory=list)
    snr_db: list[float] = field(default_factory=list)  # the direct peak above the noise floor
    app_snr_db: list[float | None] = field(default_factory=list)
    coupling_db: float | None = None  # the worst band, recorded / played (module docstring)


def find_arrivals(recorded: np.ndarray, train: ClickTrain, *, max_delay_s: float = 0.5,
                  through_app: bool = False, min_gap_s: float = 0.05, app_max_s: float = 0.8,
                  min_snr: float = 8.0, onset: float = 0.5, lookback_ms: float = 3.0,
                  segment_ms: float = 10.0) -> Arrivals:
    """The arrivals of every click (see the module docstring)."""
    sr = train.sample_rate
    x = np.asarray(recorded, dtype=np.float64).ravel()
    corr = _correlation(x, train.template)
    pre = int(round(0.001 * sr))
    # the room noise: the lead-in before the first click, minus its first 0.1 s (stream start)
    first_click = min(train.positions) if train.positions else x.size
    noise = x[min(int(0.1 * sr), first_click // 2):first_click]
    lead = corr[min(int(0.1 * sr), first_click // 2):first_click]
    floor = float(np.median(lead if lead.size >= sr // 10 else corr)) + 1e-12
    seg_n = int(round(segment_ms * sr / 1000.0))
    span = int(round(0.030 * sr))  # the direct arrival and its early reflections
    segments: list[np.ndarray] = []
    out = Arrivals()
    max_d, gap, app_max = (int(round(s * sr)) for s in (max_delay_s, min_gap_s, app_max_s))
    back = int(round(lookback_ms * sr / 1000.0))
    positions = list(train.positions)

    def skip(snr_db: float = 0.0) -> None:
        out.direct.append(None)
        out.app.append(None)
        out.snr_db.append(snr_db)
        out.app_snr_db.append(None)

    for k, p in enumerate(positions):
        # never search into the next click's arrivals
        nxt = positions[k + 1] - p if k + 1 < len(positions) else corr.size - p
        window = corr[p:p + min(max_d, nxt)]
        if window.size == 0:
            skip()
            continue
        peak = float(window.max())
        if peak / floor < min_snr:
            skip(_db(peak / floor))
            continue
        d = _onset(window, onset, back)
        out.direct.append(d)
        out.snr_db.append(_db(peak / floor))
        start = max(0, p + d - pre)
        if start + pre + span <= x.size:
            segments.append(x[start:start + pre + span])
        if not through_app:
            out.app.append(None)
            out.app_snr_db.append(None)
            continue
        # the app's copy: matched against this click's own recorded direct arrival
        segment = x[start:start + seg_n]
        first = max(gap, d // 2)
        lo, hi = start + first, min(x.size, p + d + app_max, p + nxt + d - gap)
        if segment.size < seg_n or hi - lo <= seg_n:
            out.app.append(None)
            out.app_snr_db.append(None)
            continue
        later = _matched(x[lo:hi], segment)
        app_peak = float(later.max())
        # the room's noise through the same filter (the window itself holds the click's echoes)
        quiet = _matched(noise, segment) if noise.size > 4 * segment.size else later
        app_floor = float(np.median(quiet)) + 1e-12
        out.app_snr_db.append(_db(app_peak / app_floor))
        found = app_peak / app_floor >= min_snr
        # lag k in ``later`` = the copy starts at lo + k; the segment started at ``start``
        out.app.append(d + (lo - start) + _onset(later, onset, back) if found else None)
    out.coupling_db = _band_coupling_db(segments, train.template, noise, sr)
    return out


def app_gain_db(coupling_db: float | None) -> float:
    """The app's playback gain for the app modes: the loop gain ``coupling + gain`` held at
    :data:`LOOP_DB`, within the page's range; 0 dB when the coupling is unknown."""
    if coupling_db is None:
        return 0.0
    return float(np.clip(LOOP_DB - coupling_db, MIN_APP_GAIN_DB, MAX_APP_GAIN_DB))


def _levels_db(x: np.ndarray, sr: int, block_ms: float = 20.0) -> np.ndarray:
    n = max(1, int(round(block_ms * sr / 1000.0)))
    blocks = x[:x.size // n * n].reshape(-1, n)
    return 10.0 * np.log10(np.maximum(np.mean(blocks.astype(np.float64) ** 2, axis=1), 1e-24))


def feedback_db(recorded: np.ndarray, train: ClickTrain, last_s: float = 3.0) -> float | None:
    """How much louder the recording's end is than its silent lead-in: the median 20 ms level of
    the last ``last_s`` seconds recorded (an aborted run ends where it stopped) against the
    lead-in's. About 0 dB when every click's echoes die away; far above :data:`FEEDBACK_DB`
    when the loop builds up. ``None`` when there is no lead-in or nothing after it."""
    sr = train.sample_rate
    x = np.asarray(recorded, dtype=np.float64).ravel()
    first = min(train.positions) if train.positions else 0
    lead = x[min(int(0.1 * sr), first // 2):first]
    nonzero = np.flatnonzero(x)
    end = int(nonzero[-1]) + 1 if nonzero.size else 0
    late = x[max(first, end - int(last_s * sr)):end]
    if lead.size < sr // 10 or late.size < sr // 10:
        return None
    return round(float(np.median(_levels_db(late, sr)) - np.median(_levels_db(lead, sr))), 1)


def median_or_none(values: Sequence[float | None]) -> float | None:
    found = [v for v in values if v is not None]
    return round(float(np.median(found)), 1) if found else None


def summarize_ms(samples: Sequence[int | None], sample_rate: int) -> dict[str, Any]:
    """Median / min / max in milliseconds over the clicks that were found."""
    found = [1000.0 * s / sample_rate for s in samples if s is not None]
    if not found:
        return {"found": 0, "median_ms": None, "min_ms": None, "max_ms": None}
    return {"found": len(found), "median_ms": round(float(np.median(found)), 1),
            "min_ms": round(min(found), 1), "max_ms": round(max(found), 1)}


def app_delays(arrivals: Arrivals) -> list[int | None]:
    """``d2 - d1`` per click (the app's microphone -> speaker delay)."""
    return [None if d is None or a is None else a - d
            for d, a in zip(arrivals.direct, arrivals.app, strict=True)]


def play_and_record(train: ClickTrain, *, input_index: int, output_index: int,
                    input_channels: int, keep_channel: int = 0, output_channels: int = 2,
                    wasapi: bool = True, latency: str | float | None = None,
                    blocksize: int = 0, timeout_s: float | None = None,
                    on_block: Callable[[], None] | None = None,
                    abort_rise_db: float | None = None) -> np.ndarray:
    """Play ``train`` and record the input in one duplex stream; returns the recording (mono,
    ``keep_channel`` of the input), sample-aligned with the playback. ``abort_rise_db``: stop
    early (the rest stays zero) once the last second recorded is that much louder than the
    lead-in (feedback building up)."""
    sd = _sd()
    extra = None
    if wasapi and hasattr(sd, "WasapiSettings"):
        extra = (sd.WasapiSettings(auto_convert=True), sd.WasapiSettings(auto_convert=True))
    signal = train.signal
    recorded = np.zeros(signal.size, dtype=np.float32)
    pos = [0]
    done = threading.Event()
    first = min(train.positions) if train.positions else 0
    lead_from = min(int(0.1 * train.sample_rate), first // 2)
    lead_power: list[float] = []
    second = train.sample_rate

    def callback(indata, outdata, frames, time_info, status) -> None:
        i = pos[0]
        n = max(0, min(frames, signal.size - i))
        outdata.fill(0.0)
        if n:
            outdata[:n] = signal[i:i + n, None]
            recorded[i:i + n] = indata[:n, keep_channel]
        pos[0] = i + frames
        if on_block is not None:
            on_block()
        if abort_rise_db is not None and pos[0] >= first:
            if not lead_power:
                lead_power.append(float(np.mean(recorded[lead_from:first] ** 2)) + 1e-24)
            elif pos[0] >= first + second:
                now = float(np.mean(recorded[pos[0] - second:pos[0]] ** 2)) + 1e-24
                if 10.0 * np.log10(now / lead_power[0]) > abort_rise_db:
                    done.set()
                    raise sd.CallbackStop
        if pos[0] >= signal.size:
            done.set()
            raise sd.CallbackStop

    kwargs: dict[str, Any] = dict(device=(input_index, output_index),
                                  channels=(input_channels, output_channels),
                                  samplerate=train.sample_rate, dtype="float32",
                                  blocksize=blocksize, callback=callback, extra_settings=extra)
    if latency is not None:
        kwargs["latency"] = latency
    stream = sd.Stream(**kwargs)
    with stream:
        wait = timeout_s if timeout_s is not None else signal.size / train.sample_rate + 5.0
        if not done.wait(wait):
            raise RuntimeError("the loopback stream did not finish in time")
    return recorded
