# Extraction evaluation: neural_tse (test)

500 recipes (483 with a target, 17 with the target class absent), rendered at 48000 Hz, attenuation 30 dB, detector det_mn10_v1 on cuda, extractor tse_waveformer_v2b on cuda. 3000 s of audio in 1110 s (render 21 s, detector 33 s, baseline x3 56 s, neural x2 867 s).

## Overall

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | -0.11 | +0.00 | +0.00 | +0.00 | 0 | 0 % | +0.00 |
| mask_baseline + timeline | +2.08 | +2.19 | +0.00 | +4.17 | 53 | 28 % | -26.60 |
| mask_baseline + live | -4.96 | -4.85 | -0.96 | +0.80 | 175 | 32 % | -28.69 |
| mask_baseline + oracle | +7.97 | +8.09 | +7.25 | +8.77 | 5 | 0 % | -30.00 |
| IRM oracle | +24.41 | +24.52 | +18.24 | +24.66 | 0 | 0 % | -100.00 |
| previous neural model (whole file) | +3.17 | +3.28 | +3.32 | +5.51 | 51 | 0 % | -12.48 |
| neural_tse without the gate (whole file) | +2.77 | +2.88 | +2.85 | +5.59 | 59 | 0 % | -15.58 |
| neural_tse offline (whole file) | +2.40 | +2.51 | +2.58 | +5.38 | 66 | 2 % | -17.13 |
| neural_tse streaming (as live) | +2.40 | +2.51 | +2.58 | +5.38 | 66 | 2 % | -17.13 |
| neural_tse streaming, 15 dB | +2.99 | +3.10 | +2.78 | +5.19 | 22 | 7 % | -10.62 |
| neural_tse streaming, 40 dB | +1.84 | +1.95 | +2.43 | +5.34 | 74 | 0 % | -19.45 |

## Overall, the same recipes in simulated rooms (reverberant copy)

Every event convolved with its own room impulse response (RT60 0.2-0.8 s, seed 4242); the target is the reverberant target.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | -0.12 | +0.00 | +0.00 | +0.00 | 0 | 0 % | +0.00 |
| mask_baseline + timeline | +0.58 | +0.70 | +0.00 | +3.57 | 70 | 33 % | -24.64 |
| mask_baseline + live | -5.32 | -5.20 | -0.17 | +0.82 | 187 | 36 % | -25.28 |
| mask_baseline + oracle | +7.60 | +7.72 | +6.94 | +8.52 | 7 | 0 % | -30.00 |
| IRM oracle | +22.50 | +22.62 | +17.67 | +22.77 | 0 | 0 % | -100.00 |
| previous neural model (whole file) | +2.80 | +2.92 | +3.11 | +5.29 | 49 | 0 % | -12.16 |
| neural_tse without the gate (whole file) | +2.31 | +2.43 | +2.52 | +5.28 | 62 | 0 % | -15.61 |
| neural_tse offline (whole file) | +1.95 | +2.07 | +2.13 | +5.16 | 71 | 1 % | -16.91 |
| neural_tse streaming (as live) | +1.95 | +2.07 | +2.13 | +5.16 | 71 | 1 % | -16.91 |
| neural_tse streaming, 15 dB | +2.61 | +2.73 | +2.40 | +5.03 | 29 | 9 % | -10.50 |
| neural_tse streaming, 40 dB | +1.40 | +1.52 | +2.04 | +5.12 | 82 | 0 % | -19.44 |

## SI-SDRi (dB) by target class, reverberant copy

| Target class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| speech | 93 | +1.26 | -5.17 | +7.85 | +21.30 | +3.45 | +2.73 | +2.57 |
| music | 95 | +0.34 | -6.43 | +7.23 | +21.43 | +2.06 | +1.28 | +0.91 |
| footsteps | 94 | -0.39 | -3.65 | +6.40 | +22.36 | +3.98 | +3.44 | +3.14 |
| dishes | 88 | +1.06 | -7.26 | +7.76 | +23.15 | +3.49 | +3.01 | +2.84 |
| chair_scrape | 96 | -0.93 | -4.23 | +7.74 | +22.05 | -0.05 | -0.14 | -0.59 |
| alarm | 82 | +1.82 | -4.74 | +8.05 | +22.41 | +3.43 | +3.54 | +3.37 |
| traffic | 84 | -1.66 | -5.44 | +5.61 | +18.82 | +2.08 | +0.73 | +0.09 |

## SI-SDRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| speech | 93 | +3.74 | -6.66 | +7.80 | +22.89 | +3.60 | +2.93 | +2.68 |
| music | 95 | +0.64 | -5.30 | +7.50 | +23.49 | +2.65 | +1.64 | +1.25 |
| footsteps | 94 | +1.07 | -4.08 | +6.63 | +23.26 | +4.41 | +3.90 | +3.75 |
| dishes | 88 | +3.46 | -5.26 | +8.21 | +25.50 | +3.84 | +3.21 | +2.97 |
| chair_scrape | 96 | -0.32 | -3.02 | +7.79 | +23.54 | +0.16 | +0.22 | -0.28 |
| alarm | 82 | +3.80 | -4.63 | +8.63 | +23.92 | +3.75 | +4.01 | +3.79 |
| traffic | 84 | -0.39 | -4.86 | +6.07 | +20.57 | +2.46 | +1.30 | +0.73 |

## SI-SDRi (dB) by target level relative to the rest

| Target vs rest | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| <= -5 dB | 118 | +3.09 | -4.13 | +8.86 | +24.33 | +3.89 | +2.90 | +2.63 |
| -5 .. 5 dB | 253 | +2.32 | -4.26 | +8.40 | +24.47 | +3.20 | +2.90 | +2.49 |
| >= 5 dB | 112 | +0.94 | -6.91 | +6.56 | +24.84 | +2.79 | +2.81 | +2.42 |

## SI-SDRi (dB) by number of foreground sources

| Foreground sources | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| 1 | 165 | +3.76 | -4.36 | +9.78 | +29.27 | +4.19 | +4.07 | +3.45 |
| 2 | 167 | +1.87 | -4.38 | +7.37 | +23.31 | +2.66 | +2.12 | +1.80 |
| 3 | 151 | +0.83 | -5.90 | +7.02 | +20.67 | +2.95 | +2.43 | +2.26 |

## SI-SDRi (dB) by how long the target sounds

| Target active | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| < 1 s | 196 | +5.44 | -5.55 | +11.53 | +32.55 | +5.07 | +5.71 | +5.27 |
| 1 .. 3 s | 203 | +1.06 | -4.26 | +7.50 | +20.70 | +2.37 | +1.58 | +1.26 |
| >= 3 s | 84 | -2.66 | -4.63 | +1.48 | +15.02 | +1.27 | -0.57 | -0.93 |

## SI-SDRi (dB) for one or several target classes

| Selection | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| 1 target class | 334 | +3.16 | -4.89 | +9.17 | +26.79 | +3.89 | +3.73 | +3.28 |
| 2+ target classes | 149 | +0.03 | -4.74 | +5.66 | +19.44 | +1.91 | +0.97 | +0.78 |

## SNRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| speech | 93 | +5.31 | +0.34 | +8.59 | +23.09 | +5.61 | +5.64 | +5.55 |
| music | 95 | +3.15 | +1.07 | +8.33 | +23.60 | +5.17 | +5.09 | +4.93 |
| footsteps | 94 | +3.79 | +1.53 | +7.59 | +23.44 | +5.91 | +6.01 | +5.87 |
| dishes | 88 | +4.95 | +0.28 | +8.71 | +25.62 | +5.71 | +5.71 | +5.48 |
| chair_scrape | 96 | +1.84 | +0.48 | +8.40 | +23.64 | +3.64 | +3.93 | +3.67 |
| alarm | 82 | +6.05 | +2.05 | +9.44 | +24.07 | +6.02 | +6.38 | +6.25 |
| traffic | 84 | +2.87 | +0.88 | +7.17 | +20.77 | +4.24 | +3.85 | +3.70 |

## Suppression (dB) when the target class is absent, by selected class

| Selected class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| speech | 2 | -30.00 | -30.00 | -30.00 | -100.00 | -9.20 | -20.04 | -21.36 |
| music | 3 | -20.70 | -23.89 | -30.00 | -100.00 | -8.77 | -9.77 | -11.22 |
| footsteps | 3 | -25.51 | -30.00 | -30.00 | -100.00 | -9.26 | -12.91 | -13.71 |
| chair_scrape | 4 | -29.88 | -29.21 | -30.00 | -100.00 | -12.44 | -12.12 | -13.35 |
| alarm | 1 | -30.00 | -30.00 | -30.00 | -100.00 | -22.26 | -24.20 | -28.60 |
| traffic | 4 | -26.01 | -29.80 | -30.00 | -100.00 | -16.91 | -21.01 | -22.94 |

## SI-SDRi (dB) by the loudest sound that was not selected

| Loudest other sound | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| speech | 14 | +3.53 | -4.15 | +9.05 | +23.61 | +4.50 | +2.66 | +2.37 |
| music | 20 | -0.58 | -2.85 | +5.14 | +19.83 | +1.46 | +1.80 | +1.58 |
| footsteps | 25 | +2.76 | -3.86 | +8.18 | +20.41 | +5.11 | +3.64 | +3.43 |
| dishes | 14 | -0.33 | -6.91 | +8.87 | +26.89 | +2.00 | +3.23 | +2.98 |
| chair_scrape | 16 | -0.12 | -6.08 | +5.29 | +23.70 | +1.01 | +0.75 | +0.72 |
| alarm | 26 | +1.49 | -7.84 | +10.65 | +31.53 | +1.40 | +1.03 | +0.84 |
| traffic | 19 | +2.21 | -4.25 | +8.61 | +23.16 | +3.20 | +2.70 | +2.24 |
| bg | 349 | +2.51 | -4.73 | +8.09 | +24.62 | +3.50 | +3.13 | +2.70 |

## Benchmark `absent` (200 recipes)

Every class in turn selected while it is not in the mixture (1-3 other classes, 1-3 background sounds). The correct output is silence.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | - | - | - | - | - | - | +0.00 |
| mask_baseline + timeline | - | - | - | - | - | - | -26.12 |
| mask_baseline + live | - | - | - | - | - | - | -26.53 |
| mask_baseline + oracle | - | - | - | - | - | - | -30.00 |
| IRM oracle | - | - | - | - | - | - | -100.00 |
| previous neural model (whole file) | - | - | - | - | - | - | -12.60 |
| neural_tse without the gate (whole file) | - | - | - | - | - | - | -14.81 |
| neural_tse offline (whole file) | - | - | - | - | - | - | -16.26 |
| neural_tse streaming (as live) | - | - | - | - | - | - | -16.26 |
| neural_tse streaming, 15 dB | - | - | - | - | - | - | -10.59 |
| neural_tse streaming, 40 dB | - | - | - | - | - | - | -18.23 |

Suppression (dB) by selected class:

| Selected class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| speech | 29 | -28.17 | -28.54 | -30.00 | -100.00 | -15.08 | -17.47 | -19.16 |
| music | 29 | -26.29 | -26.05 | -30.00 | -100.00 | -11.76 | -14.42 | -15.84 |
| footsteps | 29 | -25.81 | -26.95 | -30.00 | -100.00 | -9.99 | -10.42 | -10.95 |
| dishes | 29 | -27.10 | -27.06 | -30.00 | -100.00 | -15.95 | -15.01 | -16.71 |
| chair_scrape | 28 | -26.53 | -26.80 | -30.00 | -100.00 | -11.88 | -12.62 | -13.75 |
| alarm | 28 | -23.56 | -24.26 | -30.00 | -100.00 | -13.37 | -18.75 | -20.95 |
| traffic | 28 | -25.25 | -25.96 | -30.00 | -100.00 | -10.05 | -15.05 | -16.54 |

Suppression (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| speech | 20 | -24.45 | -24.54 | -30.00 | -100.00 | -14.47 | -15.48 | -16.89 |
| music | 19 | -27.50 | -29.13 | -30.00 | -100.00 | -11.27 | -11.85 | -12.51 |
| footsteps | 19 | -30.00 | -30.00 | -30.00 | -100.00 | -14.29 | -17.08 | -19.59 |
| dishes | 13 | -27.86 | -27.87 | -30.00 | -100.00 | -12.69 | -14.47 | -15.87 |
| chair_scrape | 22 | -26.52 | -25.48 | -30.00 | -100.00 | -13.22 | -15.81 | -17.37 |
| alarm | 17 | -22.85 | -25.91 | -30.00 | -100.00 | -10.30 | -10.55 | -11.09 |
| traffic | 18 | -25.13 | -25.35 | -30.00 | -100.00 | -10.64 | -12.20 | -13.13 |
| bg | 72 | -25.77 | -26.00 | -30.00 | -100.00 | -12.80 | -16.22 | -17.93 |

## Benchmark `pair_absent` (100 recipes)

Only the second class of the pair (sustained clips, no background), the first one selected: the owner's 'music playing, Speech selected'.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | - | - | - | - | - | - | +0.00 |
| mask_baseline + timeline | - | - | - | - | - | - | -29.57 |
| mask_baseline + live | - | - | - | - | - | - | -29.25 |
| mask_baseline + oracle | - | - | - | - | - | - | -30.00 |
| IRM oracle | - | - | - | - | - | - | -100.00 |
| previous neural model (whole file) | - | - | - | - | - | - | -15.26 |
| neural_tse without the gate (whole file) | - | - | - | - | - | - | -19.58 |
| neural_tse offline (whole file) | - | - | - | - | - | - | -20.60 |
| neural_tse streaming (as live) | - | - | - | - | - | - | -20.60 |
| neural_tse streaming, 15 dB | - | - | - | - | - | - | -12.32 |
| neural_tse streaming, 40 dB | - | - | - | - | - | - | -24.00 |

Suppression (dB) by selected class:

| Selected class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| speech | 100 | -29.57 | -29.25 | -30.00 | -100.00 | -15.26 | -19.58 | -20.60 |

Suppression (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| music | 100 | -29.57 | -29.25 | -30.00 | -100.00 | -15.26 | -19.58 | -20.60 |

## Benchmark `pair_a` (100 recipes)

Both classes of the pair, the first one selected.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | +0.63 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +7.71 | +7.09 | +5.37 | +7.85 | 4 | 20 % | - |
| mask_baseline + live | -4.01 | -4.64 | -1.01 | +0.05 | 39 | 33 % | - |
| mask_baseline + oracle | +11.59 | +10.96 | +10.03 | +11.21 | 0 | 0 % | - |
| IRM oracle | +31.89 | +31.26 | +19.52 | +31.36 | 0 | 0 % | - |
| previous neural model (whole file) | +5.96 | +5.33 | +4.80 | +6.66 | 4 | 0 % | - |
| neural_tse without the gate (whole file) | +5.96 | +5.33 | +5.10 | +7.09 | 3 | 0 % | - |
| neural_tse offline (whole file) | +5.62 | +4.99 | +4.97 | +6.78 | 3 | 0 % | - |
| neural_tse streaming (as live) | +5.62 | +4.99 | +4.97 | +6.78 | 3 | 0 % | - |
| neural_tse streaming, 15 dB | +5.34 | +4.71 | +4.73 | +6.26 | 2 | 4 % | - |
| neural_tse streaming, 40 dB | +5.38 | +4.76 | +4.69 | +6.79 | 6 | 0 % | - |

## Benchmark `pair_b` (100 recipes)

Both classes of the pair, the second one selected.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | +0.64 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +0.68 | +0.04 | -0.99 | +2.58 | 22 | 9 % | - |
| mask_baseline + live | -6.24 | -6.88 | -4.48 | -0.96 | 49 | 13 % | - |
| mask_baseline + oracle | +3.76 | +3.11 | -0.07 | +4.25 | 5 | 0 % | - |
| IRM oracle | +31.14 | +30.49 | +19.81 | +30.55 | 0 | 0 % | - |
| previous neural model (whole file) | +2.75 | +2.11 | +1.42 | +4.46 | 15 | 0 % | - |
| neural_tse without the gate (whole file) | +0.91 | +0.27 | +0.12 | +3.98 | 27 | 0 % | - |
| neural_tse offline (whole file) | +0.70 | +0.05 | -0.03 | +3.95 | 28 | 0 % | - |
| neural_tse streaming (as live) | +0.70 | +0.05 | -0.03 | +3.95 | 28 | 0 % | - |
| neural_tse streaming, 15 dB | +1.88 | +1.24 | +0.99 | +3.72 | 12 | 1 % | - |
| neural_tse streaming, 40 dB | +0.05 | -0.59 | -0.21 | +3.93 | 32 | 0 % | - |

## Benchmark `librispeech` (100 recipes)

The target is LibriSpeech test-clean read speech (16 kHz recordings), with one other class and the usual background.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | -0.88 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +6.64 | +7.52 | +6.06 | +8.34 | 0 | 0 % | - |
| mask_baseline + live | -6.27 | -5.39 | -4.00 | +1.47 | 42 | 8 % | - |
| mask_baseline + oracle | +8.38 | +9.26 | +8.49 | +9.97 | 0 | 0 % | - |
| IRM oracle | +23.39 | +24.27 | +17.67 | +24.39 | 0 | 0 % | - |
| previous neural model (whole file) | +7.18 | +8.06 | +7.33 | +9.00 | 0 | 0 % | - |
| neural_tse without the gate (whole file) | +8.05 | +8.93 | +8.25 | +9.85 | 0 | 0 % | - |
| neural_tse offline (whole file) | +7.95 | +8.83 | +8.25 | +9.78 | 0 | 0 % | - |
| neural_tse streaming (as live) | +7.95 | +8.83 | +8.25 | +9.78 | 0 | 0 % | - |
| neural_tse streaming, 15 dB | +6.52 | +7.40 | +7.12 | +8.37 | 0 | 0 % | - |
| neural_tse streaming, 40 dB | +8.03 | +8.90 | +8.17 | +9.87 | 1 | 0 % | - |

SI-SDRi (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | baseline oracle | IRM | neural v1 | neural, no gate | neural streaming |
|---|---|---|---|---|---|---|---|---|
| music | 10 | +8.75 | -4.00 | +9.66 | +28.12 | +5.67 | +8.13 | +8.10 |
| footsteps | 11 | +6.98 | -4.79 | +9.00 | +27.54 | +8.15 | +8.56 | +8.45 |
| dishes | 10 | +8.10 | -9.59 | +9.93 | +26.53 | +8.16 | +8.68 | +9.03 |
| chair_scrape | 10 | +7.19 | -10.58 | +7.41 | +32.37 | +4.76 | +5.88 | +5.88 |
| alarm | 11 | +7.89 | -0.26 | +10.74 | +29.16 | +8.28 | +9.75 | +9.75 |
| traffic | 5 | +6.47 | -4.45 | +7.94 | +17.74 | +9.99 | +10.41 | +10.42 |
| bg | 43 | +7.34 | -5.11 | +9.27 | +19.64 | +9.05 | +9.60 | +9.31 |
