# Extraction evaluation: neural_tse (test)

500 recipes (483 with a target, 17 with the target class absent), rendered at 48000 Hz, attenuation 30 dB, detector det_mn10_v1 on cuda, extractor tse_waveformer_v1 on cuda. 3000 s of audio in 1370 s (render 23 s, detector 38 s, baseline x3 63 s, neural x2 1123 s).

## Overall

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|
| mixture (no processing) | -0.11 | +0.00 | +0.00 | +0.00 | 0 % | +0.00 |
| mask_baseline + timeline | +2.08 | +2.19 | +0.00 | +4.17 | 28 % | -26.60 |
| mask_baseline + live | -4.96 | -4.85 | -0.96 | +0.80 | 32 % | -28.69 |
| mask_baseline + oracle | +7.97 | +8.09 | +7.25 | +8.77 | 0 % | -30.00 |
| IRM oracle | +24.41 | +24.52 | +18.24 | +24.66 | 0 % | -100.00 |
| neural_tse offline (whole file) | +3.17 | +3.28 | +3.32 | +5.51 | 0 % | -12.48 |
| neural_tse streaming (as live) | +3.17 | +3.28 | +3.32 | +5.51 | 0 % | -12.48 |
| neural_tse streaming, 15 dB | +3.21 | +3.32 | +3.17 | +5.06 | 2 % | -8.49 |

## Overall, the same recipes in simulated rooms (reverberant copy)

Every event convolved with its own room impulse response (RT60 0.2-0.8 s, seed 4242); the target is the reverberant target.

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|
| mixture (no processing) | -0.12 | +0.00 | +0.00 | +0.00 | 0 % | +0.00 |
| mask_baseline + timeline | +0.58 | +0.70 | +0.00 | +3.57 | 33 % | -24.64 |
| mask_baseline + live | -5.32 | -5.20 | -0.17 | +0.82 | 36 % | -25.28 |
| mask_baseline + oracle | +7.60 | +7.72 | +6.94 | +8.52 | 0 % | -30.00 |
| IRM oracle | +22.50 | +22.62 | +17.67 | +22.77 | 0 % | -100.00 |
| neural_tse offline (whole file) | +2.80 | +2.92 | +3.11 | +5.29 | 0 % | -12.16 |
| neural_tse streaming (as live) | +2.80 | +2.92 | +3.11 | +5.29 | 0 % | -12.16 |
| neural_tse streaming, 15 dB | +2.86 | +2.98 | +3.19 | +4.90 | 1 % | -8.26 |

## SI-SDRi (dB) by target class, reverberant copy

| Target class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural streaming |
|---|---|---|---|---|---|---|
| speech | 93 | +1.26 | -5.17 | +7.85 | +21.30 | +3.45 |
| music | 95 | +0.34 | -6.43 | +7.23 | +21.43 | +2.06 |
| footsteps | 94 | -0.39 | -3.65 | +6.40 | +22.36 | +3.98 |
| dishes | 88 | +1.06 | -7.26 | +7.76 | +23.15 | +3.49 |
| chair_scrape | 96 | -0.93 | -4.23 | +7.74 | +22.05 | -0.05 |
| alarm | 82 | +1.82 | -4.74 | +8.05 | +22.41 | +3.43 |
| traffic | 84 | -1.66 | -5.44 | +5.61 | +18.82 | +2.08 |

## SI-SDRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural streaming |
|---|---|---|---|---|---|---|
| speech | 93 | +3.74 | -6.66 | +7.80 | +22.89 | +3.60 |
| music | 95 | +0.64 | -5.30 | +7.50 | +23.49 | +2.65 |
| footsteps | 94 | +1.07 | -4.08 | +6.63 | +23.26 | +4.41 |
| dishes | 88 | +3.46 | -5.26 | +8.21 | +25.50 | +3.84 |
| chair_scrape | 96 | -0.32 | -3.02 | +7.79 | +23.54 | +0.16 |
| alarm | 82 | +3.80 | -4.63 | +8.63 | +23.92 | +3.75 |
| traffic | 84 | -0.39 | -4.86 | +6.07 | +20.57 | +2.46 |

## SI-SDRi (dB) by target level relative to the rest

| Target vs rest | recipes | baseline offline | baseline live | baseline oracle | IRM | neural streaming |
|---|---|---|---|---|---|---|
| <= -5 dB | 118 | +3.09 | -4.13 | +8.86 | +24.33 | +3.89 |
| -5 .. 5 dB | 253 | +2.32 | -4.26 | +8.40 | +24.47 | +3.20 |
| >= 5 dB | 112 | +0.94 | -6.91 | +6.56 | +24.84 | +2.79 |

## SI-SDRi (dB) by number of foreground sources

| Foreground sources | recipes | baseline offline | baseline live | baseline oracle | IRM | neural streaming |
|---|---|---|---|---|---|---|
| 1 | 165 | +3.76 | -4.36 | +9.78 | +29.27 | +4.19 |
| 2 | 167 | +1.87 | -4.38 | +7.37 | +23.31 | +2.66 |
| 3 | 151 | +0.83 | -5.90 | +7.02 | +20.67 | +2.95 |

## SI-SDRi (dB) by how long the target sounds

| Target active | recipes | baseline offline | baseline live | baseline oracle | IRM | neural streaming |
|---|---|---|---|---|---|---|
| < 1 s | 196 | +5.44 | -5.55 | +11.53 | +32.55 | +5.07 |
| 1 .. 3 s | 203 | +1.06 | -4.26 | +7.50 | +20.70 | +2.37 |
| >= 3 s | 84 | -2.66 | -4.63 | +1.48 | +15.02 | +1.27 |

## SI-SDRi (dB) for one or several target classes

| Selection | recipes | baseline offline | baseline live | baseline oracle | IRM | neural streaming |
|---|---|---|---|---|---|---|
| 1 target class | 334 | +3.16 | -4.89 | +9.17 | +26.79 | +3.89 |
| 2+ target classes | 149 | +0.03 | -4.74 | +5.66 | +19.44 | +1.91 |

## SNRi (dB) by target class

| Target class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural streaming |
|---|---|---|---|---|---|---|
| speech | 93 | +5.31 | +0.34 | +8.59 | +23.09 | +5.61 |
| music | 95 | +3.15 | +1.07 | +8.33 | +23.60 | +5.17 |
| footsteps | 94 | +3.79 | +1.53 | +7.59 | +23.44 | +5.91 |
| dishes | 88 | +4.95 | +0.28 | +8.71 | +25.62 | +5.71 |
| chair_scrape | 96 | +1.84 | +0.48 | +8.40 | +23.64 | +3.64 |
| alarm | 82 | +6.05 | +2.05 | +9.44 | +24.07 | +6.02 |
| traffic | 84 | +2.87 | +0.88 | +7.17 | +20.77 | +4.24 |

## Suppression (dB) when the target class is absent, by selected class

| Selected class | recipes | baseline offline | baseline live | baseline oracle | IRM | neural streaming |
|---|---|---|---|---|---|---|
| speech | 2 | -30.00 | -30.00 | -30.00 | -100.00 | -9.20 |
| music | 3 | -20.70 | -23.89 | -30.00 | -100.00 | -8.77 |
| footsteps | 3 | -25.51 | -30.00 | -30.00 | -100.00 | -9.26 |
| chair_scrape | 4 | -29.88 | -29.21 | -30.00 | -100.00 | -12.44 |
| alarm | 1 | -30.00 | -30.00 | -30.00 | -100.00 | -22.26 |
| traffic | 4 | -26.01 | -29.80 | -30.00 | -100.00 | -16.91 |
