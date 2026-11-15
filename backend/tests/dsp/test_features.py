"""EfficientAT front-end parity with the reference implementation (stored fixture)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from soundspy.dsp.features import EfficientATMel

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "efficientat_mel_reference.npy"


def parity_signal(sr: int = 32000, seconds: float = 1.0) -> np.ndarray:
    """The fixed signal the fixture was generated from (EfficientAT's AugmentMelSTFT, eval)."""
    t = np.arange(int(sr * seconds)) / sr
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * (200 + 6000 * t) * t)
    x += 0.05 * np.sin(2 * np.pi * 9000 * t) * (t > 0.5)
    return x.astype(np.float32)


def test_matches_reference_implementation() -> None:
    reference = np.load(FIXTURE)
    mel = EfficientATMel().eval()
    with torch.no_grad():
        ours = mel(torch.from_numpy(parity_signal()[None, :])).numpy()
    assert ours.shape == (1, 1, 128, 100)
    assert mel.frames(32000) == 100
    np.testing.assert_allclose(ours[0, 0], reference, atol=1e-4)


def test_batch_and_training_masks() -> None:
    mel = EfficientATMel(freqm=48, timem=20, mel_range_aug=True).train()
    x = torch.from_numpy(np.stack([parity_signal(), parity_signal()]))
    out = mel(x)
    assert out.shape == (2, 1, 128, 100)
    mel.eval()
    a, b = mel(x)
    torch.testing.assert_close(a, b)  # eval: deterministic, no masking
