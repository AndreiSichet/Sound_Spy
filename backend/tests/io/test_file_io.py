"""FileSource and FileSink."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundspy.config import DeviceProfile
from soundspy.core.audio import AudioBuffer
from soundspy.io.file_sink import FileSink
from soundspy.io.file_source import FileSource


def _wav(path: Path, frames: int = 2500, channels: int = 2, sr: int = 16000) -> np.ndarray:
    # channel c holds frame_number + c * 0.25 (scaled) so frames and channels are recognisable
    n = np.arange(frames, dtype=np.float32)[:, None]
    data = ((n + np.arange(channels)[None, :] * 0.25) / 10000).astype(np.float32)
    sf.write(str(path), data, sr, subtype="FLOAT")
    return data


def test_chunks_are_exact_and_last_is_zero_padded(tmp_path: Path) -> None:
    data = _wav(tmp_path / "a.wav")
    chunks = list(FileSource(tmp_path / "a.wav").chunks(1000))
    assert [c.samples.shape for c in chunks] == [(1000, 2)] * 3
    assert "valid_frames" not in chunks[0].metadata
    assert chunks[-1].metadata["valid_frames"] == 500
    np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks])[:2500], data)
    assert not chunks[-1].samples[500:].any()


def test_mono_file_is_2d(tmp_path: Path) -> None:
    _wav(tmp_path / "m.wav", channels=1)
    buf = FileSource(tmp_path / "m.wav").read(0.01)
    assert buf.samples.shape == (160, 1)


def test_exact_multiple_has_no_padding_chunk(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav", frames=2000)
    chunks = list(FileSource(tmp_path / "a.wav").chunks(1000))
    assert len(chunks) == 2 and all("valid_frames" not in c.metadata for c in chunks)


def test_loop_wraps_around(tmp_path: Path) -> None:
    data = _wav(tmp_path / "a.wav", frames=2500)
    src = FileSource(tmp_path / "a.wav", loop=True)
    chunks = []
    for chunk in src.chunks(1000):
        chunks.append(chunk)
        if len(chunks) == 4:
            break
    joined = np.concatenate([c.samples for c in chunks])
    np.testing.assert_array_equal(joined[2500:4000], data[:1500])
    assert all("valid_frames" not in c.metadata for c in chunks)


def test_profile_raw_channels_are_applied(tmp_path: Path) -> None:
    data = _wav(tmp_path / "six.wav", channels=6)
    profile = DeviceProfile(name="respeaker", raw_channels=(2, 3, 4, 5))
    src = FileSource(tmp_path / "six.wav", profile=profile)
    assert src.num_channels == 4
    buf = src.read(0.1)
    np.testing.assert_array_equal(buf.samples, data[:1600, 2:6])
    assert buf.metadata["channel_indices"] == [2, 3, 4, 5]
    assert buf.metadata["profile"] == "respeaker"


def test_profile_needing_missing_channels_is_refused(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav")
    with pytest.raises(ValueError, match="has only 2 channel"):
        FileSource(tmp_path / "a.wav", profile=DeviceProfile(name="r", raw_channels=(2, 3)))


def test_read_past_the_end_is_padded(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav", frames=1000)
    buf = FileSource(tmp_path / "a.wav").read(0.1)  # 1600 frames requested
    assert buf.num_frames == 1600 and buf.metadata["valid_frames"] == 1000


def test_realtime_paces_chunks(tmp_path: Path) -> None:
    _wav(tmp_path / "a.wav", frames=800)  # 5 chunks of 10 ms
    started = time.monotonic()
    chunks = list(FileSource(tmp_path / "a.wav", realtime=True).chunks(160))
    elapsed = time.monotonic() - started
    assert len(chunks) == 5 and elapsed >= 0.035


def test_file_sink_round_trip_honours_valid_frames(tmp_path: Path) -> None:
    data = _wav(tmp_path / "in.wav")
    out = tmp_path / "out" / "copy.wav"
    with FileSink(out) as sink:
        sink.open(16000, 2)
        for chunk in FileSource(tmp_path / "in.wav").chunks(1000):
            sink.write(chunk)
    written, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == 16000 and sink.frames_written == 2500
    np.testing.assert_array_equal(written, data)
    assert sf.info(str(out)).subtype == "FLOAT"


def test_file_sink_refuses_to_overwrite_and_wrong_format(tmp_path: Path) -> None:
    _wav(tmp_path / "x.wav")
    with pytest.raises(FileExistsError):
        FileSink(tmp_path / "x.wav").open(16000, 2)
    sink = FileSink(tmp_path / "y.wav")
    sink.open(16000, 2)
    with pytest.raises(ValueError, match="does not match"):
        sink.write(AudioBuffer(np.zeros((10, 1), dtype=np.float32), 16000))
    sink.close()
    with pytest.raises(RuntimeError):
        FileSink(tmp_path / "z.wav").write(AudioBuffer(np.zeros((10, 1), dtype=np.float32), 16000))
