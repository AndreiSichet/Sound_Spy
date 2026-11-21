"""Acoustic loopback latency (step 08): click detection on synthetic recordings, the duplex
stream through a fake device with a known delay, and ``measure-latency`` end to end."""

from __future__ import annotations

import numpy as np

from soundspy.io import loopback as lb

SR = 48000


def _recording(train: lb.ClickTrain, delay: int, app: int | None = None, noise: float = 1e-3):
    rng = np.random.default_rng(0)
    out = noise * rng.standard_normal(train.signal.size).astype(np.float32)
    for p in train.positions:
        n = train.template.size
        out[p + delay:p + delay + n] += 0.4 * train.template
        if app is not None:  # the app plays the microphone back: a weaker second arrival
            out[p + delay + app:p + delay + app + n] += 0.1 * train.template
    return out


def _speaker_and_room(sig: np.ndarray, coupling_db: float) -> np.ndarray:
    """A laptop speaker (band-pass 300 Hz - 6 kHz), a desk reflection 1 ms later and a short room
    tail, ``coupling_db`` below the played level: the shape of a 3 ms click changes a lot."""
    from scipy import signal

    rng = np.random.default_rng(5)
    b, a = signal.butter(2, [300, 6000], btype="bandpass", fs=SR)
    tail = np.arange(int(0.3 * SR)) / SR
    room = rng.standard_normal(tail.size) * np.exp(-6.9 * tail / 0.4) * 0.04
    room[0] += 1.0
    room[48] += 0.5
    y = signal.fftconvolve(signal.lfilter(b, a, sig), room)[:sig.size]
    return y * 10 ** (coupling_db / 20)


def _room_noise(size: int, dbfs: float = -60.0) -> np.ndarray:
    from scipy import signal

    noise = signal.lfilter([1.0], [1.0, -0.95], np.random.default_rng(6).standard_normal(size))
    return noise * 10 ** (dbfs / 20) / noise.std()


def _played(train: lb.ClickTrain, delay: int) -> np.ndarray:
    out = np.zeros(train.signal.size)
    out[delay:] = train.signal[:-delay]
    return out


def test_click_train_layout() -> None:
    train = lb.click_train(SR, clicks=5, period_s=0.25, lead_s=0.5, tail_s=1.0)
    assert train.positions == tuple(int(0.5 * SR) + i * int(0.25 * SR) for i in range(5))
    assert train.signal.size == int(0.5 * SR) + 5 * int(0.25 * SR) + SR
    assert np.abs(train.signal).max() <= 10 ** (-6 / 20) + 1e-6


def test_direct_arrivals_give_the_round_trip() -> None:
    train = lb.click_train(SR)
    arrivals = lb.find_arrivals(_recording(train, 2000), train)
    summary = lb.summarize_ms(arrivals.direct, SR)
    assert summary["found"] == 10 and summary["median_ms"] == round(2000 / SR * 1000, 1)


def test_the_app_arrival_gives_the_app_delay() -> None:
    train = lb.click_train(SR)
    app = int(0.098 * SR)  # the app's microphone -> speaker delay
    arrivals = lb.find_arrivals(_recording(train, 1500, app), train, through_app=True)
    summary = lb.summarize_ms(lb.app_delays(arrivals), SR)
    assert summary["found"] == 10 and abs(summary["median_ms"] - 98.0) < 0.1


def test_a_click_reshaped_by_the_speaker_and_the_room_is_found() -> None:
    """The owner's first real run (21 Nov): the first version scored such clicks 7-8 against a
    threshold of 8 at any volume (it measured the shape, not the level)."""
    train = lb.click_train(SR)
    delay = int(0.147 * SR)
    recording = _speaker_and_room(_played(train, delay), -25.0) + _room_noise(train.signal.size)
    arrivals = lb.find_arrivals(recording, train)
    summary = lb.summarize_ms(arrivals.direct, SR)
    assert summary["found"] == 10 and abs(summary["median_ms"] - 147.0) < 0.5
    assert min(arrivals.snr_db) > 18.0
    # the worst band is at least the direct path's gain (never too low: the howl margin)
    assert arrivals.coupling_db is not None and -25.5 < arrivals.coupling_db < -15.0


def test_the_app_copy_is_found_after_two_passes() -> None:
    train = lb.click_train(SR)
    delay, app = int(0.147 * SR), int(0.250 * SR)
    direct = _speaker_and_room(_played(train, delay), -20.0)
    heard = direct + _room_noise(train.signal.size)
    gain = 5.0  # loop -15 dB broadband
    replay = np.zeros_like(heard)
    replay[app:] = heard[:-app] * 10 ** (gain / 20)  # the app plays the microphone back
    arrivals = lb.find_arrivals(heard + _speaker_and_room(replay, -20.0), train, through_app=True)
    summary = lb.summarize_ms(lb.app_delays(arrivals), SR)
    assert summary["found"] >= 9 and abs(summary["median_ms"] - 250.0) < 1.0


def test_the_apps_gain_keeps_the_loop_below_unity() -> None:
    assert lb.app_gain_db(None) == 0.0
    assert lb.app_gain_db(-20.0) == -5.0  # loop -25 dB (the owner's laptop: -19.8 dB coupling)
    assert lb.app_gain_db(-35.0) == 10.0
    assert lb.app_gain_db(-60.0) == lb.MAX_APP_GAIN_DB
    assert lb.app_gain_db(0.0) == lb.MIN_APP_GAIN_DB


def test_silence_finds_nothing() -> None:
    train = lb.click_train(SR, clicks=3)
    quiet = 1e-3 * np.random.default_rng(1).standard_normal(train.signal.size)
    arrivals = lb.find_arrivals(quiet, train)
    assert lb.summarize_ms(arrivals.direct, SR)["found"] == 0  # headphones on: no false click


def test_duplex_stream_through_a_fake_loopback(fake_sounddevice) -> None:
    fake_sounddevice.loopback_delay_frames = 1234
    train = lb.click_train(SR, clicks=4, period_s=0.3, tail_s=0.5)
    recorded = lb.play_and_record(train, input_index=5, output_index=7, input_channels=2)
    arrivals = lb.find_arrivals(recorded, train)
    assert arrivals.direct == [1234] * 4


def test_measure_latency_hardware_mode(config, fake_sounddevice, tmp_path) -> None:
    import dataclasses

    from soundspy.cli import measure_latency_cmd as cmd

    fake_sounddevice.loopback_delay_frames = 960
    config = dataclasses.replace(config, paths={**config.paths, "experiments": tmp_path})
    result = cmd.measure(config, ["hardware"], clicks=3, period_s=0.3, progress=lambda m: None)
    hw = result["modes"]["hardware"]
    assert hw["found"] == 3 and hw["median_ms"] == 20.0
    # the fake loopback is flat at 0.5: -6 dB, so the app would play at -19 dB (loop -25 dB)
    assert abs(result["coupling_db"] + 6.0) < 0.5 and abs(result["app_gain_db"] + 19.0) < 0.5
    assert result["recordings"] == {"hardware": None}
    path = cmd.save(config, result)
    assert path.is_file() and (tmp_path / "realtime" / "latency_latest.json").is_file()
    assert "hardware round trip" in cmd.format_result(result)


def test_measure_latency_saves_the_recordings(config, fake_sounddevice, tmp_path) -> None:
    import dataclasses

    import soundfile as sf

    from soundspy.cli import measure_latency_cmd as cmd

    fake_sounddevice.loopback_delay_frames = 960
    config = dataclasses.replace(config, paths={**config.paths, "experiments": tmp_path})
    result = cmd.measure(config, ["hardware"], clicks=2, period_s=0.3, progress=lambda m: None,
                         audio_dir=tmp_path / "audio")
    data, sr = sf.read(result["recordings"]["hardware"])
    assert sr == 48000 and data.ndim == 1 and np.abs(data).max() > 0.1
    assert cmd.diagnosis(result) == []  # every click found: nothing to explain


def test_a_short_measurement_explains_itself() -> None:
    from soundspy.cli import measure_latency_cmd as cmd

    result = {"clicks": 10, "modes": {
        "hardware": {"title": "hardware round trip", "found": 10, "best_snr_db": 30.0},
        "speech": {"title": "Hybrid, speech route", "found": 0, "snr_db": 30.2,
                   "app_snr_db": 14.9}}}
    (line,) = cmd.diagnosis(result)
    assert "0/10" in line and "+14.9 dB" in line and "18 dB needed" in line


def test_feedback_is_measured_against_the_lead_in() -> None:
    """The owner's second run (21 Nov): the loop through the app built up by about 46 dB."""
    train = lb.click_train(SR)
    stable = _speaker_and_room(_played(train, 7000), -20.0) + _room_noise(train.signal.size)
    assert abs(lb.feedback_db(stable, train)) < 3.0
    t = np.arange(train.signal.size) / SR
    growing = stable + _room_noise(train.signal.size) * 10 ** (np.maximum(t - 1.0, 0) * 15 / 20)
    assert lb.feedback_db(growing, train) > lb.FEEDBACK_DB


def test_the_noise_floor_is_the_lead_in_not_the_whole_recording() -> None:
    """Loud sound later in the recording (feedback) must not hide the clicks before it."""
    train = lb.click_train(SR)
    x = _speaker_and_room(_played(train, 7000), -20.0) + _room_noise(train.signal.size)
    later = np.arange(x.size - train.positions[3]) / SR
    x[train.positions[3]:] += 0.05 * np.sin(2 * np.pi * 1000 * later)
    arrivals = lb.find_arrivals(x, train)
    assert all(d is not None for d in arrivals.direct[:3])


def test_a_run_that_builds_up_is_stopped(fake_sounddevice) -> None:
    fake_sounddevice.loopback_delay_frames = 960
    fake_sounddevice.loopback_growth_db_per_s = 40.0
    train = lb.click_train(SR, clicks=6, period_s=1.0)
    recorded = lb.play_and_record(train, input_index=5, output_index=7, input_channels=2,
                                  abort_rise_db=lb.ABORT_RISE_DB)
    stopped_at = int(np.flatnonzero(recorded)[-1])
    assert stopped_at < train.signal.size - SR  # stopped well before the end
    assert lb.feedback_db(recorded, train) > lb.FEEDBACK_DB

