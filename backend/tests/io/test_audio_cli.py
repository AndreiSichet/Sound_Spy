"""The audio commands (`devices`, `check-mic`, `record`) against the fake device."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soundspy.cli.main import main


def test_devices_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["devices", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["active_profile"] == "laptop"
    assert report["resolved_input"]["index"] == 5
    assert report["resolved_input"]["hostapi"] == "Windows WASAPI"
    assert len(report["inputs"]) == 5 and len(report["outputs"]) == 4


def test_devices_table_marks_default_and_resolved(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["devices"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    mme_default = next(line for line in lines if line.strip().startswith("*") and " 1 " in line)
    assert "MME" in mme_default
    resolved = next(line for line in lines if ">>" in line and "Microphone Array" in line)
    assert "Windows WASAPI" in resolved
    assert "profile 'laptop' -> device 5" in out
    assert "Headphones (USB Audio)" in out


def test_devices_with_unknown_device_reports_why(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["devices", "--device", "Zoom H6"]) == 0
    assert "does not resolve" in capsys.readouterr().out


def test_check_mic_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-mic", "--seconds", "0.2", "--countdown", "0"]) == 0
    out = capsys.readouterr().out
    assert "Windows WASAPI" in out
    assert "RMS dBFS" in out and "noise floor" in out
    # fake channel 1 = channel 0 delayed by 3 samples
    assert "lag +3 samples" in out and "distinct microphones" in out
    assert "clap near the LEFT side" in out


def test_check_mic_warns_about_digital_silence(
    fake_sounddevice, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_sounddevice.leading_zero_frames = 10**9  # a muted / gated input
    assert main(["check-mic", "--seconds", "0.2", "--countdown", "0"]) == 0
    out = capsys.readouterr().out
    assert "exact digital silence" in out and "muted or very quiet" in out
    assert "start-up silence discarded: 48000 frames" in out
    assert "noise suppression is probably active" in out


def test_check_mic_normal_noise_has_no_suppression_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["check-mic", "--seconds", "0.2", "--countdown", "0"]) == 0
    out = capsys.readouterr().out
    assert "noise suppression" not in out and "exact digital silence" not in out


def test_check_mic_meter(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-mic", "--meter", "--meter-seconds", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "ch0 [" in out and "dropped chunks" in out


def test_check_mic_with_missing_device_fails_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-mic", "--device", "Zoom H6", "--countdown", "0"]) == 1
    assert "error:" in capsys.readouterr().err


def test_record_writes_wav_and_sidecar(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["record", "--seconds", "0.2", "--label", "test", "--notes", "cli", "--out",
         str(tmp_path), "--countdown", "0"]
    )
    assert code == 0
    wavs = list(tmp_path.glob("test_*.wav"))
    assert len(wavs) == 1
    sidecar = json.loads(wavs[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["label"] == "test" and sidecar["notes"] == "cli"
    assert sidecar["profile"] == "laptop" and sidecar["hostapi"] == "Windows WASAPI"
    assert sidecar["sample_rate"] == 48000 and sidecar["channels"] == 2
    assert sidecar["duration_s"] == pytest.approx(0.2)
    assert f"Saved: {wavs[0]}" in capsys.readouterr().out
