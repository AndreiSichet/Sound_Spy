# Extraction evaluation: hybrid (test)

500 recipes (483 with a target, 17 with the target class absent), rendered at 48000 Hz, attenuation 30 dB, detector det_mn10_v1 on cuda, extractor tse_waveformer_v1 + DeepFilterNet3 on cuda + cpu (onnx). 3000 s of audio in 1205 s (render 14 s, detector 45 s, baseline x3 67 s).

## Overall

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | -0.11 | +0.00 | +0.00 | +0.00 | 0 | 0 % | +0.00 |
| mask_baseline + timeline | +2.08 | +2.19 | +0.00 | +4.17 | 53 | 28 % | -26.60 |
| mask_baseline + live | -4.96 | -4.85 | -0.96 | +0.80 | 175 | 32 % | -28.69 |
| mask_baseline + oracle | +7.97 | +8.09 | +7.25 | +8.77 | 5 | 0 % | -30.00 |
| IRM oracle | +24.41 | +24.52 | +18.24 | +24.66 | 0 | 0 % | -100.00 |
| Waveformer v1 (whole file, no gate) | +3.17 | +3.28 | +3.32 | +5.51 | 51 | 0 % | -12.48 |
| Waveformer v1 + hybrid gate | +2.88 | +3.00 | +3.35 | +5.60 | 56 | 2 % | -19.60 |
| DeepFilterNet3 alone (Speech only) | +6.24 | +6.08 | +6.69 | +7.90 | 5 | 4 % | -19.27 |
| DeepFilterNet3 + hybrid gate (Speech only) | +6.77 | +6.60 | +7.27 | +8.04 | 5 | 10 % | -34.30 |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | +5.00 | +4.83 | +1.18 | +6.06 | 1 | 24 % | -27.51 |
| cascade + hybrid gate (Speech only) | +6.00 | +5.83 | +5.53 | +6.08 | 1 | 29 % | -42.94 |
| hybrid offline (whole file, centred detector) | +3.03 | +3.14 | +3.18 | +5.71 | 58 | 3 % | -22.51 |
| hybrid streaming (20 ms chunks, centred detector) | +3.03 | +3.14 | +3.18 | +5.71 | 58 | 3 % | -22.51 |
| hybrid with the live detector's lag | +1.86 | +1.97 | +2.48 | +5.11 | 68 | 4 % | -20.87 |
| hybrid offline, 15 dB | +3.68 | +3.79 | +3.48 | +5.54 | 27 | 7 % | -14.31 |
| hybrid offline, 40 dB | +2.56 | +2.67 | +3.08 | +5.69 | 64 | 1 % | -26.02 |

## Speech selected alone (51 recipes with a target, 2 without)

| System | SI-SDRi (dB) | worse by > 5 dB | interruption | suppression, absent |
|---|---|---|---|---|
| mixture (no processing) | +0.00 | 0 | 0.0 % | +0.00 |
| mask_baseline + timeline | +6.17 | 3 | 22.5 % | -30.00 |
| mask_baseline + live | -8.23 | 29 | 79.7 % | -30.00 |
| mask_baseline + oracle | +10.43 | 0 | 2.9 % | -30.00 |
| IRM oracle | +24.85 | 0 | 0.0 % | -100.00 |
| Waveformer v1 (whole file, no gate) | +4.85 | 3 | 28.7 % | -9.20 |
| Waveformer v1 + hybrid gate | +5.22 | 3 | 28.7 % | -9.52 |
| DeepFilterNet3 alone (Speech only) | +6.08 | 5 | 39.8 % | -19.27 |
| DeepFilterNet3 + hybrid gate (Speech only) | +6.60 | 5 | 39.8 % | -34.30 |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | +4.83 | 1 | 57.3 % | -27.51 |
| cascade + hybrid gate (Speech only) | +5.83 | 1 | 57.3 % | -42.94 |
| hybrid offline (whole file, centred detector) | +6.60 | 5 | 39.8 % | -34.30 |
| hybrid streaming (20 ms chunks, centred detector) | +6.60 | 5 | 39.8 % | -34.30 |
| hybrid with the live detector's lag | +3.80 | 9 | 43.6 % | -34.29 |
| hybrid offline, live default dB | +7.44 | 0 | 35.6 % | -20.79 |
| hybrid offline, high dB | +5.49 | 8 | 40.4 % | -40.52 |

## SI-SDRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 93 | +3.74 | -6.66 | +3.60 | +3.80 | +6.08 | +6.60 | +4.83 | +5.83 | +4.56 | +2.97 |
| music | 95 | +0.64 | -5.30 | +2.65 | +2.59 | - | - | - | - | +2.59 | +1.62 |
| footsteps | 94 | +1.07 | -4.08 | +4.41 | +4.49 | - | - | - | - | +4.49 | +3.85 |
| dishes | 88 | +3.46 | -5.26 | +3.84 | +3.72 | - | - | - | - | +3.72 | +2.28 |
| chair_scrape | 96 | -0.32 | -3.02 | +0.16 | -1.40 | - | - | - | - | -1.40 | -2.06 |
| alarm | 82 | +3.80 | -4.63 | +3.75 | +3.82 | - | - | - | - | +3.82 | +2.61 |
| traffic | 84 | -0.39 | -4.86 | +2.46 | +2.27 | - | - | - | - | +2.27 | +2.11 |

## SI-SDRi (dB) by target level relative to the rest

| Target vs rest | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <= -5 dB | 118 | +3.09 | -4.13 | +3.89 | +3.86 | +9.40 | +10.81 | +7.68 | +10.22 | +4.18 | +2.76 |
| -5 .. 5 dB | 253 | +2.32 | -4.26 | +3.20 | +2.70 | +5.46 | +5.94 | +3.65 | +4.40 | +2.93 | +1.70 |
| >= 5 dB | 112 | +0.94 | -6.91 | +2.79 | +2.74 | +4.15 | +3.97 | +4.17 | +4.21 | +2.51 | +1.74 |

## SI-SDRi (dB) by number of foreground sources

| Foreground sources | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 165 | +3.76 | -4.36 | +4.19 | +3.84 | +6.49 | +6.50 | +4.94 | +6.02 | +4.12 | +2.81 |
| 2 | 167 | +1.87 | -4.38 | +2.66 | +2.41 | +4.98 | +5.87 | +3.12 | +3.95 | +2.61 | +1.38 |
| 3 | 151 | +0.83 | -5.90 | +2.95 | +2.71 | +6.72 | +7.66 | +6.76 | +7.83 | +2.66 | +1.69 |

## SI-SDRi (dB) by how long the target sounds

| Target active | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| < 1 s | 196 | +5.44 | -5.55 | +5.07 | +5.06 | +7.03 | +7.86 | +5.86 | +7.09 | +5.31 | +3.03 |
| 1 .. 3 s | 203 | +1.06 | -4.26 | +2.37 | +1.89 | +6.06 | +6.15 | +3.86 | +4.60 | +1.91 | +1.40 |
| >= 3 s | 84 | -2.66 | -4.63 | +1.27 | +0.85 | -1.48 | -1.79 | +0.23 | +0.31 | +1.06 | +0.84 |

## SI-SDRi (dB) for one or several target classes

| Selection | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 target class | 334 | +3.16 | -4.89 | +3.89 | +3.52 | +6.08 | +6.60 | +4.83 | +5.83 | +3.73 | +2.13 |
| 2+ target classes | 149 | +0.03 | -4.74 | +1.91 | +1.82 | - | - | - | - | +1.82 | +1.59 |

## SNRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 93 | +5.31 | +0.34 | +5.61 | +5.76 | +7.90 | +8.04 | +6.06 | +6.08 | +6.35 | +5.58 |
| music | 95 | +3.15 | +1.07 | +5.17 | +5.20 | - | - | - | - | +5.20 | +4.66 |
| footsteps | 94 | +3.79 | +1.53 | +5.91 | +6.03 | - | - | - | - | +6.03 | +5.61 |
| dishes | 88 | +4.95 | +0.28 | +5.71 | +5.67 | - | - | - | - | +5.67 | +4.96 |
| chair_scrape | 96 | +1.84 | +0.48 | +3.64 | +3.65 | - | - | - | - | +3.65 | +3.29 |
| alarm | 82 | +6.05 | +2.05 | +6.02 | +6.34 | - | - | - | - | +6.34 | +5.86 |
| traffic | 84 | +2.87 | +0.88 | +4.24 | +4.23 | - | - | - | - | +4.23 | +4.12 |

## Suppression (dB) when the target class is absent, by selected class

| Selected class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 2 | -30.00 | -30.00 | -9.20 | -9.52 | -19.27 | -34.30 | -27.51 | -42.94 | -34.30 | -34.29 |
| music | 3 | -20.70 | -23.89 | -8.77 | -11.37 | - | - | - | - | -11.37 | -11.40 |
| footsteps | 3 | -25.51 | -30.00 | -9.26 | -9.89 | - | - | - | - | -9.89 | -10.02 |
| chair_scrape | 4 | -29.88 | -29.21 | -12.44 | -27.45 | - | - | - | - | -27.45 | -20.20 |
| alarm | 1 | -30.00 | -30.00 | -22.26 | -41.45 | - | - | - | - | -41.45 | -41.45 |
| traffic | 4 | -26.01 | -29.80 | -16.91 | -24.77 | - | - | - | - | -24.77 | -24.93 |

## SI-SDRi (dB) by the loudest sound that was not selected

| Loudest other sound | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 14 | +3.53 | -4.15 | +4.50 | +3.96 | - | - | - | - | +3.96 | +4.33 |
| music | 20 | -0.58 | -2.85 | +1.46 | -0.03 | +6.80 | +6.81 | +3.74 | +3.99 | +0.35 | -0.21 |
| footsteps | 25 | +2.76 | -3.86 | +5.11 | +3.85 | +4.01 | +4.83 | +6.24 | +6.65 | +2.73 | +1.15 |
| dishes | 14 | -0.33 | -6.91 | +2.00 | +2.37 | +5.98 | +6.12 | +3.31 | +3.31 | +3.07 | +0.12 |
| chair_scrape | 16 | -0.12 | -6.08 | +1.01 | +1.01 | - | - | - | - | +1.01 | +0.61 |
| alarm | 26 | +1.49 | -7.84 | +1.40 | +1.11 | +3.50 | +6.09 | +1.83 | +4.41 | +1.62 | -2.71 |
| traffic | 19 | +2.21 | -4.25 | +3.20 | +3.19 | +4.09 | +5.00 | +5.63 | +6.01 | +3.19 | +2.56 |
| bg | 349 | +2.51 | -4.73 | +3.50 | +3.31 | +7.10 | +7.21 | +5.26 | +6.27 | +3.51 | +2.51 |

## Benchmark `absent` (200 recipes)

Every class in turn selected while it is not in the mixture (1-3 other classes, 1-3 background sounds). The correct output is silence.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | - | - | - | - | - | - | +0.00 |
| mask_baseline + timeline | - | - | - | - | - | - | -26.12 |
| mask_baseline + live | - | - | - | - | - | - | -26.53 |
| mask_baseline + oracle | - | - | - | - | - | - | -30.00 |
| IRM oracle | - | - | - | - | - | - | -100.00 |
| Waveformer v1 (whole file, no gate) | - | - | - | - | - | - | -12.60 |
| Waveformer v1 + hybrid gate | - | - | - | - | - | - | -16.84 |
| DeepFilterNet3 alone (Speech only) | - | - | - | - | - | - | -20.00 |
| DeepFilterNet3 + hybrid gate (Speech only) | - | - | - | - | - | - | -29.12 |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | - | - | - | - | - | - | -28.48 |
| cascade + hybrid gate (Speech only) | - | - | - | - | - | - | -43.72 |
| hybrid offline (whole file, centred detector) | - | - | - | - | - | - | -18.18 |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | - | - | - |
| hybrid with the live detector's lag | - | - | - | - | - | - | -19.15 |
| hybrid offline, 15 dB | - | - | - | - | - | - | -12.72 |
| hybrid offline, 40 dB | - | - | - | - | - | - | -20.11 |

Suppression (dB) by selected class:

| Selected class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 29 | -28.17 | -28.54 | -15.08 | -19.95 | -20.00 | -29.12 | -28.48 | -43.72 | -29.12 | -31.16 |
| music | 29 | -26.29 | -26.05 | -11.76 | -14.57 | - | - | - | - | -14.57 | -14.94 |
| footsteps | 29 | -25.81 | -26.95 | -9.99 | -12.58 | - | - | - | - | -12.58 | -12.80 |
| dishes | 29 | -27.10 | -27.06 | -15.95 | -23.69 | - | - | - | - | -23.69 | -23.95 |
| chair_scrape | 28 | -26.53 | -26.80 | -11.88 | -15.57 | - | - | - | - | -15.57 | -16.02 |
| alarm | 28 | -23.56 | -24.26 | -13.37 | -18.67 | - | - | - | - | -18.67 | -21.51 |
| traffic | 28 | -25.25 | -25.96 | -10.05 | -12.78 | - | - | - | - | -12.78 | -13.42 |

Suppression (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 20 | -24.45 | -24.54 | -14.47 | -19.18 | - | - | - | - | -19.18 | -19.50 |
| music | 19 | -27.50 | -29.13 | -11.27 | -14.65 | -20.99 | -28.62 | -26.59 | -39.36 | -19.99 | -20.67 |
| footsteps | 19 | -30.00 | -30.00 | -14.29 | -22.78 | -25.52 | -55.52 | -29.97 | -59.97 | -23.12 | -23.46 |
| dishes | 13 | -27.86 | -27.87 | -12.69 | -16.70 | - | - | - | - | -16.70 | -19.46 |
| chair_scrape | 22 | -26.52 | -25.48 | -13.22 | -18.96 | -24.06 | -35.43 | -28.10 | -41.02 | -23.37 | -25.03 |
| alarm | 17 | -22.85 | -25.91 | -10.30 | -11.32 | -24.86 | -37.31 | -28.91 | -43.94 | -15.04 | -15.78 |
| traffic | 18 | -25.13 | -25.35 | -10.64 | -13.16 | -19.05 | -30.55 | -29.89 | -50.60 | -14.36 | -14.45 |
| bg | 72 | -25.77 | -26.00 | -12.80 | -16.82 | -16.47 | -22.11 | -28.83 | -43.53 | -16.49 | -17.62 |

## Benchmark `pair_absent` (100 recipes)

Only the second class of the pair (sustained clips, no background), the first one selected: the owner's 'music playing, Speech selected'.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | - | - | - | - | - | - | +0.00 |
| mask_baseline + timeline | - | - | - | - | - | - | -29.57 |
| mask_baseline + live | - | - | - | - | - | - | -29.25 |
| mask_baseline + oracle | - | - | - | - | - | - | -30.00 |
| IRM oracle | - | - | - | - | - | - | -100.00 |
| Waveformer v1 (whole file, no gate) | - | - | - | - | - | - | -15.26 |
| Waveformer v1 + hybrid gate | - | - | - | - | - | - | -24.80 |
| DeepFilterNet3 alone (Speech only) | - | - | - | - | - | - | -25.94 |
| DeepFilterNet3 + hybrid gate (Speech only) | - | - | - | - | - | - | -48.13 |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | - | - | - | - | - | - | -27.46 |
| cascade + hybrid gate (Speech only) | - | - | - | - | - | - | -51.06 |
| hybrid offline (whole file, centred detector) | - | - | - | - | - | - | -48.13 |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | - | - | - |
| hybrid with the live detector's lag | - | - | - | - | - | - | -48.66 |
| hybrid offline, 15 dB | - | - | - | - | - | - | -25.76 |
| hybrid offline, 40 dB | - | - | - | - | - | - | -60.45 |

Suppression (dB) by selected class:

| Selected class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 100 | -29.57 | -29.25 | -15.26 | -24.80 | -25.94 | -48.13 | -27.46 | -51.06 | -48.13 | -48.66 |

Suppression (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| music | 100 | -29.57 | -29.25 | -15.26 | -24.80 | -25.94 | -48.13 | -27.46 | -51.06 | -48.13 | -48.66 |

## Benchmark `pair_a` (100 recipes)

Both classes of the pair, the first one selected.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | +0.63 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +7.71 | +7.09 | +5.37 | +7.85 | 4 | 20 % | - |
| mask_baseline + live | -4.01 | -4.64 | -1.01 | +0.05 | 39 | 33 % | - |
| mask_baseline + oracle | +11.59 | +10.96 | +10.03 | +11.21 | 0 | 0 % | - |
| IRM oracle | +31.89 | +31.26 | +19.52 | +31.36 | 0 | 0 % | - |
| Waveformer v1 (whole file, no gate) | +5.96 | +5.33 | +4.80 | +6.66 | 4 | 0 % | - |
| Waveformer v1 + hybrid gate | +6.13 | +5.50 | +4.92 | +6.86 | 5 | 2 % | - |
| DeepFilterNet3 alone (Speech only) | +7.44 | +6.81 | +7.13 | +7.92 | 10 | 4 % | - |
| DeepFilterNet3 + hybrid gate (Speech only) | +8.22 | +7.59 | +7.72 | +8.23 | 10 | 7 % | - |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | +3.19 | +2.56 | +2.43 | +4.44 | 13 | 9 % | - |
| cascade + hybrid gate (Speech only) | +4.38 | +3.75 | +4.21 | +4.56 | 13 | 13 % | - |
| hybrid offline (whole file, centred detector) | +8.22 | +7.59 | +7.72 | +8.23 | 10 | 7 % | - |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | 0 | - | - |
| hybrid with the live detector's lag | +4.84 | +4.22 | +5.17 | +6.50 | 18 | 10 % | - |
| hybrid offline, 15 dB | +8.87 | +8.25 | +7.79 | +8.38 | 1 | 10 % | - |
| hybrid offline, 40 dB | +7.48 | +6.85 | +7.46 | +8.11 | 12 | 5 % | - |

## Benchmark `pair_b` (100 recipes)

Both classes of the pair, the second one selected.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | +0.64 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +0.68 | +0.04 | -0.99 | +2.58 | 22 | 9 % | - |
| mask_baseline + live | -6.24 | -6.88 | -4.48 | -0.96 | 49 | 13 % | - |
| mask_baseline + oracle | +3.76 | +3.11 | -0.07 | +4.25 | 5 | 0 % | - |
| IRM oracle | +31.14 | +30.49 | +19.81 | +30.55 | 0 | 0 % | - |
| Waveformer v1 (whole file, no gate) | +2.75 | +2.11 | +1.42 | +4.46 | 15 | 0 % | - |
| Waveformer v1 + hybrid gate | +2.81 | +2.17 | +1.41 | +4.53 | 15 | 0 % | - |
| DeepFilterNet3 alone (Speech only) | - | - | - | - | 0 | - | - |
| DeepFilterNet3 + hybrid gate (Speech only) | - | - | - | - | 0 | - | - |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | - | - | - | - | 0 | - | - |
| cascade + hybrid gate (Speech only) | - | - | - | - | 0 | - | - |
| hybrid offline (whole file, centred detector) | +2.81 | +2.17 | +1.41 | +4.53 | 15 | 0 % | - |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | 0 | - | - |
| hybrid with the live detector's lag | +2.23 | +1.59 | +1.13 | +4.16 | 15 | 0 % | - |
| hybrid offline, 15 dB | +3.02 | +2.38 | +1.66 | +4.14 | 6 | 1 % | - |
| hybrid offline, 40 dB | +2.66 | +2.01 | +1.36 | +4.55 | 16 | 0 % | - |

## Benchmark `librispeech` (100 recipes)

The target is LibriSpeech test-clean read speech (16 kHz recordings), with one other class and the usual background.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Worse than input by > 5 dB | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|---|
| mixture (no processing) | -0.88 | +0.00 | +0.00 | +0.00 | 0 | 0 % | - |
| mask_baseline + timeline | +6.64 | +7.52 | +6.06 | +8.34 | 0 | 0 % | - |
| mask_baseline + live | -6.27 | -5.39 | -4.00 | +1.47 | 42 | 8 % | - |
| mask_baseline + oracle | +8.38 | +9.26 | +8.49 | +9.97 | 0 | 0 % | - |
| IRM oracle | +23.39 | +24.27 | +17.67 | +24.39 | 0 | 0 % | - |
| Waveformer v1 (whole file, no gate) | +7.18 | +8.06 | +7.33 | +9.00 | 0 | 0 % | - |
| Waveformer v1 + hybrid gate | +7.44 | +8.32 | +7.40 | +9.25 | 0 | 0 % | - |
| DeepFilterNet3 alone (Speech only) | +10.93 | +11.80 | +11.32 | +12.06 | 0 | 0 % | - |
| DeepFilterNet3 + hybrid gate (Speech only) | +11.20 | +12.08 | +11.77 | +12.28 | 0 | 0 % | - |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | +8.71 | +9.59 | +8.92 | +10.20 | 1 | 2 % | - |
| cascade + hybrid gate (Speech only) | +8.96 | +9.84 | +8.94 | +10.30 | 1 | 3 % | - |
| hybrid offline (whole file, centred detector) | +11.20 | +12.08 | +11.77 | +12.28 | 0 | 0 % | - |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | 0 | - | - |
| hybrid with the live detector's lag | +9.36 | +10.23 | +10.84 | +10.97 | 2 | 0 % | - |
| hybrid offline, 15 dB | +9.57 | +10.45 | +10.59 | +10.92 | 0 | 1 % | - |
| hybrid offline, 40 dB | +11.23 | +12.11 | +11.75 | +12.31 | 0 | 0 % | - |

SI-SDRi (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| music | 10 | +8.75 | -4.00 | +5.67 | +5.91 | +12.28 | +12.35 | +9.36 | +9.40 | +12.35 | +12.61 |
| footsteps | 11 | +6.98 | -4.79 | +8.15 | +8.53 | +13.43 | +13.73 | +10.66 | +10.71 | +13.73 | +10.84 |
| dishes | 10 | +8.10 | -9.59 | +8.16 | +8.50 | +10.70 | +10.85 | +10.46 | +10.70 | +10.85 | +7.24 |
| chair_scrape | 10 | +7.19 | -10.58 | +4.76 | +4.93 | +10.77 | +10.88 | +8.77 | +8.87 | +10.88 | +8.91 |
| alarm | 11 | +7.89 | -0.26 | +8.28 | +8.40 | +13.38 | +13.56 | +9.19 | +9.36 | +13.56 | +11.43 |
| traffic | 5 | +6.47 | -4.45 | +9.99 | +10.34 | +10.44 | +11.01 | +7.42 | +7.52 | +11.01 | +6.50 |
| bg | 43 | +7.34 | -5.11 | +9.05 | +9.32 | +11.53 | +11.90 | +9.71 | +10.13 | +11.90 | +10.66 |
