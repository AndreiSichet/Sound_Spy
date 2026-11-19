# Real-room benchmark, session 20261119-105248-00c1

Takes: {'speech': ['speech_01.wav'], 'music': ['music_01.wav'], 'music_pop': ['music_pop_01.wav'], 'room': ['room_01.wav']}. Speech selected. Mixtures: SI-SDRi (dB) at 30 dB; music only: suppression (dB) and speech only: SI-SDR against the input (dB) / interruption rate at 40 dB.

| System | mix -5 dB | mix +0 dB | mix +5 dB | mix +10 dB | music only | speech only SI-SDR | speech only interruption |
|---|---|---|---|---|---|---|---|
| input (bypass) | +0.00 | +0.00 | +0.00 | +0.00 | 0.0 | +100.0 | 0.0 % |
| baseline offline | +1.48 | +1.44 | +1.45 | +0.76 | -23.8 | +14.7 | 2.4 % |
| baseline live | -4.20 | -4.63 | -1.60 | -3.19 | -27.5 | +8.0 | 19.8 % |
| Waveformer v1 | -1.70 | +0.95 | +1.99 | +0.88 | -19.1 | +14.0 | 4.8 % |
| v1 + hybrid gate | -1.91 | +0.93 | +1.97 | +0.85 | -39.1 | +14.0 | 4.8 % |
| DeepFilterNet3 alone | +13.62 | +11.66 | +9.00 | +6.25 | -23.5 | +17.5 | 0.2 % |
| hybrid (offline detector) | +13.17 | +11.34 | +8.88 | +6.22 | -43.5 | +17.5 | 0.2 % |
| hybrid (live detector) | +13.06 | +11.21 | +8.89 | +6.18 | -43.6 | +17.4 | 0.5 % |
