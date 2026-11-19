# Speech-selected synthetic benchmarks (test, 30 dB)

Recipes: {'speech': 53, 'pair_a': 100, 'pair_absent': 100, 'librispeech': 100}. SI-SDRi (dB) where the target is present, suppression (dB) for music only.

| System | Speech alone | speech + music | music only (dB) | LibriSpeech |
|---|---|---|---|---|
| DeepFilterNet3, no gate | +6.97 | +6.27 | -24.3 | +12.18 |
| 07c hybrid gate (centred detector) | +7.13 | +7.00 | -45.2 | +12.49 |
| 07c hybrid gate (live detector) | +4.70 | +4.03 | -46.1 | +10.65 |
| 07d speech-route gate (centred detector) | +5.55 | +4.86 | -42.8 | +12.04 |
| 07d speech-route gate (live detector) | +4.84 | +4.61 | -42.8 | +11.88 |
