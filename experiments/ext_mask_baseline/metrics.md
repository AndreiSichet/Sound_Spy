# Extraction evaluation: mask_baseline (test)

500 recipes (483 with a target, 17 with the target class absent), rendered at 48000 Hz, attenuation 30 dB, detector det_mn10_v1 on cuda. 3000 s of audio in 230 s (render 20 s, detector 44 s, extractor x3 65 s).

## Overall

| System | SI-SDR (dB) | SI-SDRi mean (dB) | SI-SDRi median (dB) | SNRi (dB) | Target missed | Suppression, target absent (dB) |
|---|---|---|---|---|---|---|
| mixture (no processing) | -0.11 | +0.00 | +0.00 | +0.00 | 0 % | +0.00 |
| mask_baseline + timeline | +2.08 | +2.19 | +0.00 | +4.17 | 28 % | -26.60 |
| mask_baseline + live | -4.96 | -4.85 | -0.96 | +0.80 | 32 % | -28.69 |
| mask_baseline + oracle | +7.97 | +8.09 | +7.25 | +8.77 | 0 % | -30.00 |
| IRM oracle | +24.41 | +24.52 | +18.24 | +24.66 | 0 % | -100.00 |

## SI-SDRi (dB) by target class

| Target class | recipes | mask_baseline + timeline | mask_baseline + live | mask_baseline + oracle | IRM oracle |
|---|---|---|---|---|---|
| speech | 93 | +3.74 | -6.66 | +7.80 | +22.89 |
| music | 95 | +0.64 | -5.30 | +7.50 | +23.49 |
| footsteps | 94 | +1.07 | -4.08 | +6.63 | +23.26 |
| dishes | 88 | +3.46 | -5.26 | +8.21 | +25.50 |
| chair_scrape | 96 | -0.32 | -3.02 | +7.79 | +23.54 |
| alarm | 82 | +3.80 | -4.63 | +8.63 | +23.92 |
| traffic | 84 | -0.39 | -4.86 | +6.07 | +20.57 |

## SI-SDRi (dB) by target level relative to the rest

| Target vs rest | recipes | mask_baseline + timeline | mask_baseline + live | mask_baseline + oracle | IRM oracle |
|---|---|---|---|---|---|
| <= -5 dB | 118 | +3.09 | -4.13 | +8.86 | +24.33 |
| -5 .. 5 dB | 253 | +2.32 | -4.26 | +8.40 | +24.47 |
| >= 5 dB | 112 | +0.94 | -6.91 | +6.56 | +24.84 |

## SI-SDRi (dB) by number of foreground sources

| Foreground sources | recipes | mask_baseline + timeline | mask_baseline + live | mask_baseline + oracle | IRM oracle |
|---|---|---|---|---|---|
| 1 | 165 | +3.76 | -4.36 | +9.78 | +29.27 |
| 2 | 167 | +1.87 | -4.38 | +7.37 | +23.31 |
| 3 | 151 | +0.83 | -5.90 | +7.02 | +20.67 |

## SI-SDRi (dB) by how long the target sounds

| Target active | recipes | mask_baseline + timeline | mask_baseline + live | mask_baseline + oracle | IRM oracle |
|---|---|---|---|---|---|
| < 1 s | 196 | +5.44 | -5.55 | +11.53 | +32.55 |
| 1 .. 3 s | 203 | +1.06 | -4.26 | +7.50 | +20.70 |
| >= 3 s | 84 | -2.66 | -4.63 | +1.48 | +15.02 |

## SNRi (dB) by target class

| Target class | recipes | mask_baseline + timeline | mask_baseline + live | mask_baseline + oracle | IRM oracle |
|---|---|---|---|---|---|
| speech | 93 | +5.31 | +0.34 | +8.59 | +23.09 |
| music | 95 | +3.15 | +1.07 | +8.33 | +23.60 |
| footsteps | 94 | +3.79 | +1.53 | +7.59 | +23.44 |
| dishes | 88 | +4.95 | +0.28 | +8.71 | +25.62 |
| chair_scrape | 96 | +1.84 | +0.48 | +8.40 | +23.64 |
| alarm | 82 | +6.05 | +2.05 | +9.44 | +24.07 |
| traffic | 84 | +2.87 | +0.88 | +7.17 | +20.77 |

## Suppression (dB) when the target class is absent, by selected class

| Selected class | recipes | mask_baseline + timeline | mask_baseline + live | mask_baseline + oracle | IRM oracle |
|---|---|---|---|---|---|
| speech | 2 | -30.00 | -30.00 | -30.00 | -100.00 |
| music | 3 | -20.70 | -23.89 | -30.00 | -100.00 |
| footsteps | 3 | -25.51 | -30.00 | -30.00 | -100.00 |
| chair_scrape | 4 | -29.88 | -29.21 | -30.00 | -100.00 |
| alarm | 1 | -30.00 | -30.00 | -30.00 | -100.00 |
| traffic | 4 | -26.01 | -29.80 | -30.00 | -100.00 |
