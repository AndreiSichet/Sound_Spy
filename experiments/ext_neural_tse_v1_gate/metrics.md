# Extraction evaluation: neural_tse (test)

500 recipes (483 with a target, 17 with the target class absent), rendered at 48000 Hz, attenuation 30 dB, detector det_mn10_v1 on cuda, extractor tse_waveformer_v1 on cuda. 3000 s of audio in 1114 s (render 11 s, detector 33 s, baseline x3 61 s, neural x2 877 s).

## Overall

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | -0.11 | +0.00 | +0.00 | +0.00 | 0 | 0 % | +0.00 |
| mask_baseline + timeline | +2.08 | +2.19 | +0.00 | +4.17 | 53 | 28 % | -26.60 |
| mask_baseline + live | -4.96 | -4.85 | -0.96 | +0.80 | 175 | 32 % | -28.69 |
| mask_baseline + oracle | +7.97 | +8.09 | +7.25 | +8.77 | 5 | 0 % | -30.00 |
| IRM oracle | +24.41 | +24.52 | +18.24 | +24.66 | 0 | 0 % | -100.00 |
| neural_tse without the gate (whole file) | +3.17 | +3.28 | +3.32 | +5.51 | 51 | 0 % | -12.48 |
| neural_tse offline (whole file) | +2.86 | +2.97 | +3.13 | +5.34 | 55 | 1 % | -13.98 |
| neural_tse streaming (as live) | +2.86 | +2.97 | +3.13 | +5.34 | 55 | 1 % | -13.98 |
| neural_tse streaming, 15 dB | +3.13 | +3.24 | +3.10 | +5.02 | 19 | 3 % | -8.95 |
| neural_tse streaming, 40 dB | +2.49 | +2.60 | +3.04 | +5.33 | 63 | 0 % | -15.73 |

## SI-SDRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| speech | 93 | +3.74 | -6.66 | +7.80 | +22.89 | +3.60 | +3.38 |
| music | 95 | +0.64 | -5.30 | +7.50 | +23.49 | +2.65 | +2.17 |
| footsteps | 94 | +1.07 | -4.08 | +6.63 | +23.26 | +4.41 | +4.30 |
| dishes | 88 | +3.46 | -5.26 | +8.21 | +25.50 | +3.84 | +3.73 |
| chair_scrape | 96 | -0.32 | -3.02 | +7.79 | +23.54 | +0.16 | -0.33 |
| alarm | 82 | +3.80 | -4.63 | +8.63 | +23.92 | +3.75 | +3.70 |
| traffic | 84 | -0.39 | -4.86 | +6.07 | +20.57 | +2.46 | +2.12 |

## SI-SDRi (dB) by target level relative to the rest

| Target vs rest | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| <= -5 dB | 118 | +3.09 | -4.13 | +8.86 | +24.33 | +3.89 | +3.65 |
| -5 .. 5 dB | 253 | +2.32 | -4.26 | +8.40 | +24.47 | +3.20 | +2.86 |
| >= 5 dB | 112 | +0.94 | -6.91 | +6.56 | +24.84 | +2.79 | +2.51 |

## SI-SDRi (dB) by number of foreground sources

| Foreground sources | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| 1 | 165 | +3.76 | -4.36 | +9.78 | +29.27 | +4.19 | +3.67 |
| 2 | 167 | +1.87 | -4.38 | +7.37 | +23.31 | +2.66 | +2.44 |
| 3 | 151 | +0.83 | -5.90 | +7.02 | +20.67 | +2.95 | +2.79 |

## SI-SDRi (dB) by how long the target sounds

| Target active | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| < 1 s | 196 | +5.44 | -5.55 | +11.53 | +32.55 | +5.07 | +4.75 |
| 1 .. 3 s | 203 | +1.06 | -4.26 | +7.50 | +20.70 | +2.37 | +2.14 |
| >= 3 s | 84 | -2.66 | -4.63 | +1.48 | +15.02 | +1.27 | +0.84 |

## SI-SDRi (dB) for one or several target classes

| Selection | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| 1 target class | 334 | +3.16 | -4.89 | +9.17 | +26.79 | +3.89 | +3.50 |
| 2+ target classes | 149 | +0.03 | -4.74 | +5.66 | +19.44 | +1.91 | +1.80 |

## SNRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| speech | 93 | +5.31 | +0.34 | +8.59 | +23.09 | +5.61 | +5.57 |
| music | 95 | +3.15 | +1.07 | +8.33 | +23.60 | +5.17 | +5.00 |
| footsteps | 94 | +3.79 | +1.53 | +7.59 | +23.44 | +5.91 | +5.87 |
| dishes | 88 | +4.95 | +0.28 | +8.71 | +25.62 | +5.71 | +5.50 |
| chair_scrape | 96 | +1.84 | +0.48 | +8.40 | +23.64 | +3.64 | +3.41 |
| alarm | 82 | +6.05 | +2.05 | +9.44 | +24.07 | +6.02 | +5.98 |
| traffic | 84 | +2.87 | +0.88 | +7.17 | +20.77 | +4.24 | +4.14 |

## Suppression (dB) when the target class is absent, by selected class

| Selected class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| speech | 2 | -30.00 | -30.00 | -30.00 | -100.00 | -9.20 | -9.49 |
| music | 3 | -20.70 | -23.89 | -30.00 | -100.00 | -8.77 | -10.41 |
| footsteps | 3 | -25.51 | -30.00 | -30.00 | -100.00 | -9.26 | -9.70 |
| chair_scrape | 4 | -29.88 | -29.21 | -30.00 | -100.00 | -12.44 | -15.07 |
| alarm | 1 | -30.00 | -30.00 | -30.00 | -100.00 | -22.26 | -27.66 |
| traffic | 4 | -26.01 | -29.80 | -30.00 | -100.00 | -16.91 | -17.59 |

## SI-SDRi (dB) by the loudest sound that was not selected

| Loudest other sound | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| speech | 14 | +3.53 | -4.15 | +9.05 | +23.61 | +4.50 | +4.00 |
| music | 20 | -0.58 | -2.85 | +5.14 | +19.83 | +1.46 | +1.01 |
| footsteps | 25 | +2.76 | -3.86 | +8.18 | +20.41 | +5.11 | +5.00 |
| dishes | 14 | -0.33 | -6.91 | +8.87 | +26.89 | +2.00 | +1.82 |
| chair_scrape | 16 | -0.12 | -6.08 | +5.29 | +23.70 | +1.01 | +1.30 |
| alarm | 26 | +1.49 | -7.84 | +10.65 | +31.53 | +1.40 | +1.30 |
| traffic | 19 | +2.21 | -4.25 | +8.61 | +23.16 | +3.20 | +3.09 |
| bg | 349 | +2.51 | -4.73 | +8.09 | +24.62 | +3.50 | +3.14 |

## Benchmark `absent` (200 recipes)

Every class in turn selected while it is not in the mixture (1-3 other classes, 1-3 background sounds). The correct output is silence.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | - | - | - | - | - | - | +0.00 |
| mask_baseline + timeline | - | - | - | - | - | - | -26.12 |
| mask_baseline + live | - | - | - | - | - | - | -26.53 |
| mask_baseline + oracle | - | - | - | - | - | - | -30.00 |
| IRM oracle | - | - | - | - | - | - | -100.00 |
| neural_tse without the gate (whole file) | - | - | - | - | - | - | -12.60 |
| neural_tse offline (whole file) | - | - | - | - | - | - | -14.14 |
| neural_tse streaming (as live) | - | - | - | - | - | - | -14.14 |
| neural_tse streaming, 15 dB | - | - | - | - | - | - | -9.68 |
| neural_tse streaming, 40 dB | - | - | - | - | - | - | -15.31 |

Suppression (dB) by selected class:

| Selected class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| speech | 29 | -28.17 | -28.54 | -30.00 | -100.00 | -15.08 | -16.96 |
| music | 29 | -26.29 | -26.05 | -30.00 | -100.00 | -11.76 | -12.85 |
| footsteps | 29 | -25.81 | -26.95 | -30.00 | -100.00 | -9.99 | -10.80 |
| dishes | 29 | -27.10 | -27.06 | -30.00 | -100.00 | -15.95 | -17.91 |
| chair_scrape | 28 | -26.53 | -26.80 | -30.00 | -100.00 | -11.88 | -13.57 |
| alarm | 28 | -23.56 | -24.26 | -30.00 | -100.00 | -13.37 | -15.51 |
| traffic | 28 | -25.25 | -25.96 | -30.00 | -100.00 | -10.05 | -11.29 |

Suppression (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| speech | 20 | -24.45 | -24.54 | -30.00 | -100.00 | -14.47 | -16.35 |
| music | 19 | -27.50 | -29.13 | -30.00 | -100.00 | -11.27 | -12.28 |
| footsteps | 19 | -30.00 | -30.00 | -30.00 | -100.00 | -14.29 | -16.98 |
| dishes | 13 | -27.86 | -27.87 | -30.00 | -100.00 | -12.69 | -14.76 |
| chair_scrape | 22 | -26.52 | -25.48 | -30.00 | -100.00 | -13.22 | -14.83 |
| alarm | 17 | -22.85 | -25.91 | -30.00 | -100.00 | -10.30 | -11.21 |
| traffic | 18 | -25.13 | -25.35 | -30.00 | -100.00 | -10.64 | -11.59 |
| bg | 72 | -25.77 | -26.00 | -30.00 | -100.00 | -12.80 | -14.27 |

## Benchmark `pair_absent` (100 recipes)

Only the second class of the pair (sustained clips, no background), the first one selected: the owner's 'music playing, Speech selected'.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | - | - | - | - | - | - | +0.00 |
| mask_baseline + timeline | - | - | - | - | - | - | -29.57 |
| mask_baseline + live | - | - | - | - | - | - | -29.25 |
| mask_baseline + oracle | - | - | - | - | - | - | -30.00 |
| IRM oracle | - | - | - | - | - | - | -100.00 |
| neural_tse without the gate (whole file) | - | - | - | - | - | - | -15.26 |
| neural_tse offline (whole file) | - | - | - | - | - | - | -16.10 |
| neural_tse streaming (as live) | - | - | - | - | - | - | -16.10 |
| neural_tse streaming, 15 dB | - | - | - | - | - | - | -10.18 |
| neural_tse streaming, 40 dB | - | - | - | - | - | - | -18.40 |

Suppression (dB) by selected class:

| Selected class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| speech | 100 | -29.57 | -29.25 | -30.00 | -100.00 | -15.26 | -16.10 |

Suppression (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| music | 100 | -29.57 | -29.25 | -30.00 | -100.00 | -15.26 | -16.10 |

## Benchmark `pair_a` (100 recipes)

Both classes of the pair, the first one selected.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | +0.63 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +7.71 | +7.09 | +5.37 | +7.85 | 4 | 20 % | - |
| mask_baseline + live | -4.01 | -4.64 | -1.01 | +0.05 | 39 | 33 % | - |
| mask_baseline + oracle | +11.59 | +10.96 | +10.03 | +11.21 | 0 | 0 % | - |
| IRM oracle | +31.89 | +31.26 | +19.52 | +31.36 | 0 | 0 % | - |
| neural_tse without the gate (whole file) | +5.96 | +5.33 | +4.80 | +6.66 | 4 | 0 % | - |
| neural_tse offline (whole file) | +5.62 | +5.00 | +4.66 | +6.35 | 5 | 1 % | - |
| neural_tse streaming (as live) | +5.62 | +5.00 | +4.66 | +6.35 | 5 | 1 % | - |
| neural_tse streaming, 15 dB | +5.13 | +4.50 | +4.28 | +5.79 | 2 | 3 % | - |
| neural_tse streaming, 40 dB | +5.56 | +4.94 | +4.64 | +6.39 | 5 | 0 % | - |

## Benchmark `pair_b` (100 recipes)

Both classes of the pair, the second one selected.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | +0.64 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +0.68 | +0.04 | -0.99 | +2.58 | 22 | 9 % | - |
| mask_baseline + live | -6.24 | -6.88 | -4.48 | -0.96 | 49 | 13 % | - |
| mask_baseline + oracle | +3.76 | +3.11 | -0.07 | +4.25 | 5 | 0 % | - |
| IRM oracle | +31.14 | +30.49 | +19.81 | +30.55 | 0 | 0 % | - |
| neural_tse without the gate (whole file) | +2.75 | +2.11 | +1.42 | +4.46 | 15 | 0 % | - |
| neural_tse offline (whole file) | +2.48 | +1.84 | +1.35 | +4.39 | 16 | 0 % | - |
| neural_tse streaming (as live) | +2.48 | +1.84 | +1.35 | +4.39 | 16 | 0 % | - |
| neural_tse streaming, 15 dB | +2.80 | +2.16 | +1.58 | +3.98 | 6 | 0 % | - |
| neural_tse streaming, 40 dB | +2.25 | +1.61 | +1.29 | +4.43 | 18 | 0 % | - |

## Benchmark `librispeech` (100 recipes)

The target is LibriSpeech test-clean read speech (16 kHz recordings), with one other class and the usual background.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | -0.88 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +6.64 | +7.52 | +6.06 | +8.34 | 0 | 0 % | - |
| mask_baseline + live | -6.27 | -5.39 | -4.00 | +1.47 | 42 | 8 % | - |
| mask_baseline + oracle | +8.38 | +9.26 | +8.49 | +9.97 | 0 | 0 % | - |
| IRM oracle | +23.39 | +24.27 | +17.67 | +24.39 | 0 | 0 % | - |
| neural_tse without the gate (whole file) | +7.18 | +8.06 | +7.33 | +9.00 | 0 | 0 % | - |
| neural_tse offline (whole file) | +7.03 | +7.91 | +7.18 | +8.92 | 0 | 0 % | - |
| neural_tse streaming (as live) | +7.03 | +7.91 | +7.18 | +8.92 | 0 | 0 % | - |
| neural_tse streaming, 15 dB | +5.84 | +6.72 | +6.47 | +7.74 | 0 | 0 % | - |
| neural_tse streaming, 40 dB | +7.06 | +7.94 | +7.14 | +8.97 | 0 | 0 % | - |

SI-SDRi (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | baseline oracle | IRM | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|
| music | 10 | +8.75 | -4.00 | +9.66 | +28.12 | +5.67 | +5.82 |
| footsteps | 11 | +6.98 | -4.79 | +9.00 | +27.54 | +8.15 | +8.14 |
| dishes | 10 | +8.10 | -9.59 | +9.93 | +26.53 | +8.16 | +8.31 |
| chair_scrape | 10 | +7.19 | -10.58 | +7.41 | +32.37 | +4.76 | +4.71 |
| alarm | 11 | +7.89 | -0.26 | +10.74 | +29.16 | +8.28 | +8.21 |
| traffic | 5 | +6.47 | -4.45 | +7.94 | +17.74 | +9.99 | +9.96 |
| bg | 43 | +7.34 | -5.11 | +9.27 | +19.64 | +9.05 | +8.67 |
