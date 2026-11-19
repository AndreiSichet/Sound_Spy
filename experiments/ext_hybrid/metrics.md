# Extraction evaluation: hybrid (test)

500 recipes (483 with a target, 17 with the target class absent), rendered at 48000 Hz, attenuation 30 dB, detector det_mn10_v1 on cuda, extractor tse_waveformer_v1 + DeepFilterNet3 on cuda + cpu (onnx). 3000 s of audio in 1182 s (render 23 s, detector 40 s, baseline x3 60 s).

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
| DeepFilterNet3 alone (Speech only) | +7.13 | +6.97 | +6.37 | +8.30 | 4 | 0 % | -23.83 |
| DeepFilterNet3 + hybrid gate (Speech only) | +7.29 | +7.13 | +7.97 | +8.44 | 4 | 2 % | -38.97 |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | +5.74 | +5.57 | +2.17 | +6.64 | 1 | 18 % | -27.32 |
| cascade + hybrid gate (Speech only) | +6.34 | +6.17 | +4.14 | +6.68 | 1 | 28 % | -43.02 |
| hybrid offline (whole file, centred detector) | +3.09 | +3.20 | +3.34 | +5.76 | 57 | 2 % | -23.06 |
| hybrid streaming (20 ms chunks, centred detector) | +3.09 | +3.20 | +3.34 | +5.76 | 57 | 2 % | -23.06 |
| hybrid with the live detector's lag | +1.95 | +2.06 | +2.59 | +5.13 | 65 | 4 % | -21.42 |
| hybrid offline, 15 dB | +3.71 | +3.82 | +3.48 | +5.57 | 28 | 6 % | -14.52 |
| hybrid offline, 40 dB | +2.65 | +2.76 | +3.12 | +5.73 | 62 | 1 % | -26.94 |

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
| DeepFilterNet3 alone (Speech only) | +6.97 | 4 | 39.1 % | -23.83 |
| DeepFilterNet3 + hybrid gate (Speech only) | +7.13 | 4 | 39.1 % | -38.97 |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | +5.57 | 1 | 53.8 % | -27.32 |
| cascade + hybrid gate (Speech only) | +6.17 | 1 | 53.8 % | -43.02 |
| hybrid offline (whole file, centred detector) | +7.13 | 4 | 39.1 % | -38.97 |
| hybrid streaming (20 ms chunks, centred detector) | +7.13 | 4 | 39.1 % | -38.97 |
| hybrid with the live detector's lag | +4.70 | 6 | 42.6 % | -38.94 |
| hybrid offline, live default dB | +7.78 | 1 | 34.5 % | -22.52 |
| hybrid offline, high dB | +6.37 | 6 | 39.6 % | -48.35 |

## SI-SDRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 93 | +3.74 | -6.66 | +3.60 | +3.80 | +6.97 | +7.13 | +5.57 | +6.17 | +4.85 | +3.46 |
| music | 95 | +0.64 | -5.30 | +2.65 | +2.59 | - | - | - | - | +2.59 | +1.62 |
| footsteps | 94 | +1.07 | -4.08 | +4.41 | +4.49 | - | - | - | - | +4.49 | +3.85 |
| dishes | 88 | +3.46 | -5.26 | +3.84 | +3.72 | - | - | - | - | +3.72 | +2.28 |
| chair_scrape | 96 | -0.32 | -3.02 | +0.16 | -1.40 | - | - | - | - | -1.40 | -2.06 |
| alarm | 82 | +3.80 | -4.63 | +3.75 | +3.82 | - | - | - | - | +3.82 | +2.61 |
| traffic | 84 | -0.39 | -4.86 | +2.46 | +2.27 | - | - | - | - | +2.27 | +2.11 |

## SI-SDRi (dB) by target level relative to the rest

| Target vs rest | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <= -5 dB | 118 | +3.09 | -4.13 | +3.89 | +3.86 | +12.48 | +13.13 | +9.91 | +11.71 | +4.44 | +3.06 |
| -5 .. 5 dB | 253 | +2.32 | -4.26 | +3.20 | +2.70 | +5.63 | +5.91 | +3.90 | +4.33 | +2.93 | +1.74 |
| >= 5 dB | 112 | +0.94 | -6.91 | +2.79 | +2.74 | +4.24 | +3.80 | +4.38 | +4.20 | +2.49 | +1.73 |

## SI-SDRi (dB) by number of foreground sources

| Foreground sources | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 165 | +3.76 | -4.36 | +4.19 | +3.84 | +6.97 | +6.88 | +5.78 | +6.20 | +4.17 | +2.84 |
| 2 | 167 | +1.87 | -4.38 | +2.66 | +2.41 | +5.78 | +6.05 | +3.73 | +4.30 | +2.62 | +1.44 |
| 3 | 151 | +0.83 | -5.90 | +2.95 | +2.71 | +8.42 | +8.87 | +7.48 | +8.43 | +2.76 | +1.90 |

## SI-SDRi (dB) by how long the target sounds

| Target active | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| < 1 s | 196 | +5.44 | -5.55 | +5.07 | +5.06 | +7.56 | +8.00 | +6.77 | +7.49 | +5.33 | +3.14 |
| 1 .. 3 s | 203 | +1.06 | -4.26 | +2.37 | +1.89 | +7.40 | +7.01 | +4.34 | +4.82 | +1.97 | +1.47 |
| >= 3 s | 84 | -2.66 | -4.63 | +1.27 | +0.85 | +0.58 | +0.56 | +0.61 | +0.67 | +1.17 | +0.97 |

## SI-SDRi (dB) for one or several target classes

| Selection | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 target class | 334 | +3.16 | -4.89 | +3.89 | +3.52 | +6.97 | +7.13 | +5.57 | +6.17 | +3.81 | +2.27 |
| 2+ target classes | 149 | +0.03 | -4.74 | +1.91 | +1.82 | - | - | - | - | +1.82 | +1.59 |

## SNRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 93 | +5.31 | +0.34 | +5.61 | +5.76 | +8.30 | +8.44 | +6.64 | +6.68 | +6.57 | +5.69 |
| music | 95 | +3.15 | +1.07 | +5.17 | +5.20 | - | - | - | - | +5.20 | +4.66 |
| footsteps | 94 | +3.79 | +1.53 | +5.91 | +6.03 | - | - | - | - | +6.03 | +5.61 |
| dishes | 88 | +4.95 | +0.28 | +5.71 | +5.67 | - | - | - | - | +5.67 | +4.96 |
| chair_scrape | 96 | +1.84 | +0.48 | +3.64 | +3.65 | - | - | - | - | +3.65 | +3.29 |
| alarm | 82 | +6.05 | +2.05 | +6.02 | +6.34 | - | - | - | - | +6.34 | +5.86 |
| traffic | 84 | +2.87 | +0.88 | +4.24 | +4.23 | - | - | - | - | +4.23 | +4.12 |

## Suppression (dB) when the target class is absent, by selected class

| Selected class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 2 | -30.00 | -30.00 | -9.20 | -9.52 | -23.83 | -38.97 | -27.32 | -43.02 | -38.97 | -38.94 |
| music | 3 | -20.70 | -23.89 | -8.77 | -11.37 | - | - | - | - | -11.37 | -11.40 |
| footsteps | 3 | -25.51 | -30.00 | -9.26 | -9.89 | - | - | - | - | -9.89 | -10.02 |
| chair_scrape | 4 | -29.88 | -29.21 | -12.44 | -27.45 | - | - | - | - | -27.45 | -20.20 |
| alarm | 1 | -30.00 | -30.00 | -22.26 | -41.45 | - | - | - | - | -41.45 | -41.45 |
| traffic | 4 | -26.01 | -29.80 | -16.91 | -24.77 | - | - | - | - | -24.77 | -24.93 |

## SI-SDRi (dB) by the loudest sound that was not selected

| Loudest other sound | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 14 | +3.53 | -4.15 | +4.50 | +3.96 | - | - | - | - | +3.96 | +4.33 |
| music | 20 | -0.58 | -2.85 | +1.46 | -0.03 | +7.42 | +7.43 | +4.09 | +4.30 | +0.44 | -0.12 |
| footsteps | 25 | +2.76 | -3.86 | +5.11 | +3.85 | +7.70 | +8.31 | +8.00 | +8.38 | +3.57 | +2.05 |
| dishes | 14 | -0.33 | -6.91 | +2.00 | +2.37 | +5.76 | +5.86 | +3.47 | +3.48 | +3.03 | +0.06 |
| chair_scrape | 16 | -0.12 | -6.08 | +1.01 | +1.01 | - | - | - | - | +1.01 | +0.61 |
| alarm | 26 | +1.49 | -7.84 | +1.40 | +1.11 | +4.21 | +4.67 | +1.57 | +3.50 | +1.29 | -2.48 |
| traffic | 19 | +2.21 | -4.25 | +3.20 | +3.19 | +7.10 | +7.59 | +5.55 | +5.84 | +3.60 | +3.06 |
| bg | 349 | +2.51 | -4.73 | +3.50 | +3.31 | +7.38 | +7.38 | +6.16 | +6.65 | +3.52 | +2.53 |

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
| DeepFilterNet3 alone (Speech only) | - | - | - | - | - | - | -19.36 |
| DeepFilterNet3 + hybrid gate (Speech only) | - | - | - | - | - | - | -27.55 |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | - | - | - | - | - | - | -27.24 |
| cascade + hybrid gate (Speech only) | - | - | - | - | - | - | -41.51 |
| hybrid offline (whole file, centred detector) | - | - | - | - | - | - | -17.95 |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | - | - | - |
| hybrid with the live detector's lag | - | - | - | - | - | - | -18.91 |
| hybrid offline, 15 dB | - | - | - | - | - | - | -12.64 |
| hybrid offline, 40 dB | - | - | - | - | - | - | -19.82 |

Suppression (dB) by selected class:

| Selected class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 29 | -28.17 | -28.54 | -15.08 | -19.95 | -19.36 | -27.55 | -27.24 | -41.51 | -27.55 | -29.51 |
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
| music | 19 | -27.50 | -29.13 | -11.27 | -14.65 | -15.21 | -21.69 | -23.95 | -36.22 | -18.16 | -18.83 |
| footsteps | 19 | -30.00 | -30.00 | -14.29 | -22.78 | -26.18 | -56.18 | -29.97 | -59.97 | -23.15 | -23.50 |
| dishes | 13 | -27.86 | -27.87 | -12.69 | -16.70 | - | - | - | - | -16.70 | -19.46 |
| chair_scrape | 22 | -26.52 | -25.48 | -13.22 | -18.96 | -22.53 | -32.59 | -27.67 | -40.42 | -22.73 | -24.29 |
| alarm | 17 | -22.85 | -25.91 | -10.30 | -11.32 | -23.72 | -38.32 | -27.92 | -42.68 | -15.21 | -15.95 |
| traffic | 18 | -25.13 | -25.35 | -10.64 | -13.16 | -19.72 | -21.22 | -29.54 | -41.03 | -12.80 | -12.86 |
| bg | 72 | -25.77 | -26.00 | -12.80 | -16.82 | -18.01 | -24.40 | -27.47 | -42.47 | -16.87 | -18.02 |

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
| DeepFilterNet3 alone (Speech only) | - | - | - | - | - | - | -24.35 |
| DeepFilterNet3 + hybrid gate (Speech only) | - | - | - | - | - | - | -45.22 |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | - | - | - | - | - | - | -26.45 |
| cascade + hybrid gate (Speech only) | - | - | - | - | - | - | -49.48 |
| hybrid offline (whole file, centred detector) | - | - | - | - | - | - | -45.22 |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | - | - | - |
| hybrid with the live detector's lag | - | - | - | - | - | - | -46.12 |
| hybrid offline, 15 dB | - | - | - | - | - | - | -24.85 |
| hybrid offline, 40 dB | - | - | - | - | - | - | -56.00 |

Suppression (dB) by selected class:

| Selected class | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 100 | -29.57 | -29.25 | -15.26 | -24.80 | -24.35 | -45.22 | -26.45 | -49.48 | -45.22 | -46.12 |

Suppression (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| music | 100 | -29.57 | -29.25 | -15.26 | -24.80 | -24.35 | -45.22 | -26.45 | -49.48 | -45.22 | -46.12 |

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
| DeepFilterNet3 alone (Speech only) | +6.89 | +6.27 | +7.25 | +7.40 | 11 | 2 % | - |
| DeepFilterNet3 + hybrid gate (Speech only) | +7.62 | +7.00 | +7.52 | +7.72 | 10 | 6 % | - |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | +3.81 | +3.18 | +2.32 | +4.68 | 8 | 7 % | - |
| cascade + hybrid gate (Speech only) | +4.68 | +4.06 | +4.22 | +4.81 | 9 | 11 % | - |
| hybrid offline (whole file, centred detector) | +7.62 | +7.00 | +7.52 | +7.72 | 10 | 6 % | - |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | 0 | - | - |
| hybrid with the live detector's lag | +4.65 | +4.03 | +5.03 | +6.11 | 16 | 9 % | - |
| hybrid offline, 15 dB | +8.33 | +7.71 | +7.64 | +7.87 | 2 | 10 % | - |
| hybrid offline, 40 dB | +6.90 | +6.28 | +7.32 | +7.61 | 13 | 4 % | - |

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
| DeepFilterNet3 alone (Speech only) | +11.30 | +12.18 | +12.59 | +12.23 | 0 | 0 % | - |
| DeepFilterNet3 + hybrid gate (Speech only) | +11.61 | +12.49 | +12.71 | +12.48 | 0 | 0 % | - |
| Waveformer v1 -> DeepFilterNet3 (Speech only) | +9.29 | +10.17 | +10.09 | +10.41 | 1 | 1 % | - |
| cascade + hybrid gate (Speech only) | +9.48 | +10.36 | +10.11 | +10.53 | 1 | 1 % | - |
| hybrid offline (whole file, centred detector) | +11.61 | +12.49 | +12.71 | +12.48 | 0 | 0 % | - |
| hybrid streaming (20 ms chunks, centred detector) | - | - | - | - | 0 | - | - |
| hybrid with the live detector's lag | +9.78 | +10.65 | +11.23 | +11.21 | 2 | 0 % | - |
| hybrid offline, 15 dB | +9.91 | +10.79 | +11.11 | +11.10 | 0 | 0 % | - |
| hybrid offline, 40 dB | +11.67 | +12.55 | +12.71 | +12.52 | 0 | 0 % | - |

SI-SDRi (dB) by the loudest other sound:

| Loudest other sound | recipes | baseline offline | baseline live | v1 | v1 + gate | DFN3 | DFN3 + gate | cascade | cascade + gate | hybrid | hybrid live |
|---|---|---|---|---|---|---|---|---|---|---|---|
| music | 10 | +8.75 | -4.00 | +5.67 | +5.91 | +11.37 | +11.44 | +9.28 | +9.32 | +11.44 | +11.84 |
| footsteps | 11 | +6.98 | -4.79 | +8.15 | +8.53 | +12.92 | +13.12 | +10.04 | +10.08 | +13.12 | +10.73 |
| dishes | 10 | +8.10 | -9.59 | +8.16 | +8.50 | +11.20 | +11.71 | +10.95 | +11.21 | +11.71 | +8.39 |
| chair_scrape | 10 | +7.19 | -10.58 | +4.76 | +4.93 | +12.11 | +12.22 | +9.31 | +9.48 | +12.22 | +10.29 |
| alarm | 11 | +7.89 | -0.26 | +8.28 | +8.40 | +13.84 | +14.01 | +10.78 | +10.98 | +14.01 | +11.57 |
| traffic | 5 | +6.47 | -4.45 | +9.99 | +10.34 | +11.84 | +12.04 | +9.12 | +9.25 | +12.04 | +6.44 |
| bg | 43 | +7.34 | -5.11 | +9.05 | +9.32 | +12.04 | +12.48 | +10.40 | +10.64 | +12.48 | +11.22 |
