# The EfficientAT front-end below is adapted from EfficientAT, models/preprocess.py
# (AugmentMelSTFT) — https://github.com/fschmid56/EfficientAT
# Copyright (c) 2022 Florian Schmid. MIT License (full text:
# soundspy/detection/backbones/efficientat/LICENSE). Itself adapted from
# https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
# Changes for Sound Spy: return_complex=True + view_as_real instead of the deprecated
# return_complex=False (same values); the eval-mode mel filterbank is computed once and cached;
# masking/fmin/fmax augmentation are switchable; no printing; float32 forced on the matmul via
# torch.autocast(enabled=False) for the active device type.
"""Model-specific feature extractors (log-mel front-ends).

A pretrained model only works with **exactly** the features it was trained on: same sample
rate, FFT size, window, hop, mel bands and normalisation. Each backbone therefore gets its own
front-end here, and a test compares it with the reference implementation's output on a fixed
signal (``tests/fixtures/efficientat_mel_reference.npy``).

``EfficientATMel`` (for ``mn10_as``): 32 kHz mono -> pre-emphasis (-0.97) -> STFT (n_fft 1024,
symmetric Hann window of 800, hop 320 = 10 ms, centred) -> power -> 128 Kaldi mel bands
(0-15 kHz) -> ``log(x + 1e-5)`` -> ``(x + 4.5) / 5``. Output ``(batch, 1, 128, frames)``; 1 s gives
100 frames. In training mode it can mask random frequency bands / time frames (SpecAugment) and
jitter the mel range, as EfficientAT did when training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
from torch import nn


@dataclass(frozen=True)
class MelSpec:
    sample_rate: int = 32_000
    n_mels: int = 128
    win_length: int = 800
    hop_length: int = 320
    n_fft: int = 1024
    fmin: float = 0.0
    fmax: float | None = None  # None -> sr / 2 - fmax_aug_range / 2 (= 15 kHz), as EfficientAT
    fmin_aug_range: int = 10
    fmax_aug_range: int = 2000


EFFICIENTAT_MEL = MelSpec()


class EfficientATMel(nn.Module):
    """Waveform ``(batch, samples)`` @ 32 kHz -> normalised log-mel ``(batch, 1, mels, frames)``."""

    def __init__(self, spec: MelSpec = EFFICIENTAT_MEL, *, freqm: int = 0, timem: int = 0,
                 mel_range_aug: bool = False) -> None:
        super().__init__()
        self.spec = spec
        self.fmax = spec.fmax if spec.fmax is not None else (
            spec.sample_rate // 2 - spec.fmax_aug_range // 2)
        self.mel_range_aug = mel_range_aug
        self.register_buffer("window", torch.hann_window(spec.win_length, periodic=False),
                             persistent=False)
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]),
                             persistent=False)
        self.register_buffer("mel_basis", self._mel_basis(spec.fmin, self.fmax),
                             persistent=False)
        self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True) if freqm \
            else nn.Identity()
        self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True) if timem \
            else nn.Identity()

    def _mel_basis(self, fmin: float, fmax: float) -> torch.Tensor:
        s = self.spec
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            s.n_mels, s.n_fft, s.sample_rate, fmin, fmax,
            vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        return torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0)

    def frames(self, num_samples: int) -> int:
        return (num_samples - 1) // self.spec.hop_length + 1  # pre-emphasis drops one sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spec
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, s.n_fft, hop_length=s.hop_length, win_length=s.win_length,
                       center=True, normalized=False, window=self.window, return_complex=True)
        x = torch.view_as_real(x)
        x = (x ** 2).sum(dim=-1)  # power mag
        mel_basis = self.mel_basis
        if self.training and self.mel_range_aug:
            fmin = s.fmin + torch.randint(s.fmin_aug_range, (1,)).item()
            fmax = self.fmax + s.fmax_aug_range // 2 - torch.randint(s.fmax_aug_range, (1,)).item()
            mel_basis = self._mel_basis(fmin, fmax).to(x.device)
        with torch.autocast(device_type=x.device.type, enabled=False):
            melspec = torch.matmul(mel_basis.float(), x.float())
        melspec = (melspec + 0.00001).log()
        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)
        melspec = (melspec + 4.5) / 5.  # fast normalization
        return melspec.unsqueeze(1)
