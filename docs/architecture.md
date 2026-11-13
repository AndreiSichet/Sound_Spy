# Architecture

## Pipeline

```
 configs/devices/<profile>.yaml
            │ (device, raw_channels, geometry)
            ▼
 ┌────────────────────── io/ (module 1) ──────────────────────┐
 │ AudioSource:  MicSource | FileSource | SimulatedArraySource │
 │               read(seconds) / stream()  ──►  AudioBuffer     │
 │ storage: WAV (native rate, all channels) + JSON sidecar     │
 └────────────────────────────┬───────────────────────────────┘
                              │ AudioBuffer (frames, channels), float32
                              ▼
               dsp/channels.py  ChannelReducer
               (mean | select | beamform in module 3)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
   classification/ (module 2)        enhancement/ (module 2)
   features → model → sliding-       Enhancer.enhance(buffer,
   window inference → timeline ────► target_label, timeline)
                                               │
                                               ▼
                                     AudioBuffer → outputs/
```

`pipeline.py` wires these stages together; the `soundfocus` CLI exposes each stage.
Module 3 (localization) will feed a direction to the `ChannelReducer` beamforming mode and to the
spatial UI (module 4); module 5 extends enhancement with spatial target extraction.

## Rules, explained

1. **One audio format (`AudioBuffer`, shape `(frames, channels)`).** A single, always-2-D
   container means mono laptop audio and 4-channel array audio go through the same functions.
   Assuming one channel anywhere would break as soon as the array arrives.
2. **Hardware lives in config.** Device names, channel indices and mic geometry differ per
   microphone. Keeping them in `configs/devices/` means moving from the laptop mic to the
   ReSpeaker XVF3800 is a config switch (`active_device_profile`), not a code change.
3. **All audio comes through `AudioSource`.** Live mic, WAV replay and simulated scenes are
   interchangeable, so everything above `io/` can be developed and tested with files or synthetic
   data, and `sounddevice` stays isolated in one package.
4. **Channels are reduced only in `ChannelReducer`.** Downmixing is a design decision (mean now,
   beamforming toward the selected source later). Doing it in one place keeps that decision
   swappable; a stray `audio[:, 0]` in a loader would silently discard spatial information.
5. **Raw recordings are immutable.** Recordings at native rate with all channels plus metadata
   can always be re-processed with a better pipeline; derived copies live in `data/processed/`.
6. **Config over constants.** Experiments change sample rates, windows and hyperparameters;
   keeping them in YAML makes runs reproducible and comparable.
7. **Tests never need hardware.** A faked `sounddevice` keeps the suite runnable on any machine
   and in CI; GPU/dataset tests are opt-in (`--run-gpu`, `--run-data`).
8. **Windows-first.** The development machine is Windows 11; `pathlib` and PowerShell-documented
   commands avoid path and shell surprises.
9. **Git is read-only for Claude Code.** The owner reviews and commits every change.
10. **Scope discipline.** Each step implements only its spec; ideas go into the report.
11. **Every step ends with a report** in `reports/`.
