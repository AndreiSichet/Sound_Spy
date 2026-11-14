# Architecture

## Runtime picture

```
                 ┌──────────────────── Python backend (FastAPI, 127.0.0.1:8765) ────────────────────┐
 Microphone ──► AudioSource ──► Engine (worker thread, chunk by chunk)                               │
 (or WAV file)   (io/)          │                                                                   │
                                ├─► SpatialFilter     passthrough now │ beamformer (tabs 2/3)       │
                                ├─► Extractor         keeps target classes, attenuates the rest ──► AudioSink ──► headphones
                                ├─► Detector  (side)  classes present + confidence   ─┐             │
                                ├─► Localizer (side)  source directions (tabs 2/3)     ├─► events ──► WebSocket
                                └─► Analyzer  (side)  levels, spectrogram columns     ─┘             │
                                                                                                    │
 REST API: devices, config, session start/stop, target, recordings, processing ◄── SQLite (SQLAlchemy)
                 └──────────────────────────────────────────────────────────────────────────────────┘
                               ▲  HTTP + WebSocket (Vite dev proxy: /api, /ws)
                               │
                 React + TypeScript app (Vite, localhost:5173 in dev): tabs, controls, visualizations
```

### Layers

| Layer | Package | May import |
|---|---|---|
| Contracts | `soundspy.core` | numpy only |
| Engine | `soundspy.engine` | `core` |
| Stages | `io`, `dsp`, `analysis`, `detection`, `extraction`, `spatial` | `core`, `engine`, `dsp` |
| Offline jobs | `data`, `training`, `cli` | the library |
| API | `soundspy.api` (FastAPI, SQLAlchemy) | everything above |
| UI | `frontend/` | only `/api/*` and `/ws/session` |

The library never imports `soundspy.api`, FastAPI, Starlette or SQLAlchemy
(`backend/tests/test_smoke.py` enforces this both statically and at import time).

### Core contracts (`soundspy.core`)

- `AudioBuffer(samples, sample_rate, metadata)`: float32 `(frames, channels)`, always 2-D,
  validated on construction; `num_frames`, `num_channels`, `duration_s`. No channel reduction.
- `Direction(azimuth_deg, width_deg)`: azimuth normalized to [0, 360), width in (0, 360].
- `TargetSpec(labels, direction, source_id, attenuation_db=30)`: one spec for all three tabs —
  Semantic sets `labels` (empty = pass everything), Spatial adds `direction`, Sources sets
  `source_id`. `validate(label_set)` rejects unknown labels.
- Events (all carry `t`, seconds since session start): `Detection`, `SoundSource`,
  `LevelFrame`, `SpectrogramFrame`, `EngineStatus`.
- `LabelSet` / `ClassDef`: ordered classes from `configs/classes.yaml`. Each class has an
  internal ID (the key; used by code, models and the DB; never renamed), a `display_name` (UI),
  a `description` (tooltip) and `audioset` names (dataset building only, never shown to users).
  Key order = multi-hot index: append new classes, never reorder.

### Engine (`soundspy.engine`)

- `Processor` (main path): `configure(sr, ch) -> out_ch`, `set_target`, `process(chunk)`, `reset`.
- `SideProcessor` (observer): `configure`, `set_target`, `observe(chunk) -> [Event]`, `reset`.
- `ProcessorChain(main, side)`: side processors observe the **input** chunk; main processors run
  in order; channel counts propagate through `configure`.
- `Mode` → stage kinds: SEMANTIC = extractor + detector + analyzer; SPATIAL adds spatial filter +
  localizer; SOURCES adds tracker. Spatial/Sources need a profile with mic geometry.
- Offline processing = feeding a file through the same chain chunk by chunk.

## Concurrency model

```
sounddevice input callback ──► bounded input queue ──► engine worker thread ──► output queue ──► sounddevice output stream
                                                              │
                                                              └─ side events ──► loop.call_soon_threadsafe ──► asyncio loop ──► WebSocket broadcast
```

- The `sounddevice` callback only copies blocks into a bounded queue (no processing in the
  callback). What happens when the worker falls behind (e.g. drop oldest + report in the
  status) is decided in step 03.
- The **engine worker thread** chunks the input (`engine/buffers.py`), runs the
  `ProcessorChain` and pushes output chunks to the sink's queue.
- Side-branch results (detections, levels, spectrogram columns, sources) are handed to the
  asyncio event loop with `loop.call_soon_threadsafe` and broadcast by the WebSocket
  `ConnectionManager` to all connected clients.
- **One active session at a time.** Target changes arrive via REST (`PUT /api/session/target`)
  or WebSocket (`set_target`) and are applied atomically by the worker at the next chunk
  boundary (`ProcessorChain.set_target` is called only from the worker thread).

The session/worker is implemented in step 03; in step 01 the engine status is always `idle`.

## WebSocket protocol (`/ws/session`)

Every message, in both directions, uses one envelope:

```json
{"type": "status" | "levels" | "spectrogram" | "detections" | "sources" | "error" | "set_target",
 "t": 12.34,
 "payload": { ... }}
```

- `t`: seconds since session start (0.0 when no session is running).
- `payload` = the event's fields without `t`:

| type | direction | payload |
|---|---|---|
| `status` | server → client | `EngineStatus`: `state`, `mode`, `device`, `sample_rate`, `channels`, `message` |
| `levels` | server → client | `LevelFrame`: `rms_dbfs[]`, `peak_dbfs[]`, `clipping[]` (per channel) |
| `spectrogram` | server → client | `SpectrogramFrame`: `magnitudes_db[]`, `freq_max_hz` |
| `detections` | server → client | `{"items": [Detection, ...]}` |
| `sources` | server → client | `{"items": [SoundSource, ...]}` |
| `error` | server → client | `{"message": "..."}` |
| `set_target` | client → server | `TargetSpec`: `labels[]`, `direction`, `source_id`, `attenuation_db` |

On connect the server sends a `status` message, then a `status` heartbeat every
`api.ws_heartbeat_s` seconds (2 s). Unknown or malformed client messages get an `error` reply;
the client ignores unknown types with a console warning. In step 01 only `status` and `error`
are produced (`set_target` answers "not implemented (step-03)").

## REST API

| Route | Step |
|---|---|
| `GET /api/health`, `GET /api/config` | 01 (implemented) |
| `GET /api/devices` | 02 |
| `POST /api/session/start`, `POST /api/session/stop`, `PUT /api/session/target` | 03 |
| `GET /api/recordings`, `GET /api/recordings/{id}`, `GET /api/recordings/{id}/audio` | 03 |
| `POST /api/recordings/{id}/process` | 06 |

Unimplemented routes return HTTP 501 `{"detail": "not implemented (step-XX)"}`.
Interactive docs: http://127.0.0.1:8765/docs.

## Database

SQLite at `data/soundspy.db` (SQLAlchemy 2.x, URL in `configs/default.yaml`), tables created on
startup by `init_db()`:

- `recordings`: id, created_at, path (unique), duration_s, sample_rate, channels, device,
  profile, label?, notes.
- `processing_runs`: id, created_at, recording_id → recordings, mode, target_json, method
  (`mask_baseline` | `neural_tse`), output_path?, metrics_json?, status (`pending` | `done` |
  `failed`), error?.

Audio files stay on disk; the database stores paths and metadata. No migrations tool yet.

## Rules, explained

1. **One audio format.** One always-2-D container means mono laptop audio and 4-channel array
   audio flow through the same functions.
2. **Hardware lives in device profiles.** Moving from the laptop mic to the ReSpeaker XVF3800 is a
   config switch (`active_device_profile`), not a code change.
3. **`AudioSource` in, `AudioSink` out.** Live mic, WAV replay and simulated scenes are
   interchangeable; `sounddevice` stays isolated in `io/`.
4. **Channel reduction only in `ChannelReducer` or a `SpatialFilter`.** Downmixing is a design
   decision (mean now, beamforming later); a stray `audio[:, 0]` silently discards spatial data.
5. **Processors / SideProcessors, chunk-based.** Offline and live share one code path, so real
   time needs no redesign.
6. **`TargetSpec` only.** One way to express the user's choice across all tabs.
7. **Classes from `classes.yaml` via `LabelSet`.** Class order defines model vector indices.
8. **Library vs API.** The library stays testable and usable from the CLI and notebooks.
9. **Immutable raw recordings.** They can always be reprocessed with a better pipeline.
10. **Config over constants.** Reproducible, comparable experiments.
11. **No hardware in tests.** The suite runs anywhere.
12. **Windows-first.** The development machine is Windows 11.
13–16. Git read-only for Claude Code, license hygiene, scope discipline, a report per step.

## Step plan (numbers used in `TODO(...)` markers)

| Step | Content | Tab |
|---|---|---|
| 01 | Architecture restructure + app shell | - |
| 02 | Audio capture: devices, profiles, sources/sinks, storage, levels, ChannelReducer; CLI `devices`, `check-mic`, `record`; `GET /api/devices` | - |
| 03 | Live session: engine worker, chunker, speaker sink, Analyzer; WS levels + spectrogram; record button; Semantic tab live monitor | 1 |
| 04 | Data: ontology, collectors (ESC-50, FSD50K, own recordings), splits, foreground/background mixtures; `record-dataset` | - |
| 05 | Detection: features, multi-label fine-tuning of a pretrained model, chunked inference, detections over WS | 1 |
| 06 | Extraction baseline (detector-guided masking) + SI-SDR metrics; offline processing with A/B playback | 1 |
| 07 | Neural TSE (Waveformer-style, label-conditioned, streaming); live extraction in the Semantic tab | 1 |
| 08 | End-to-end evaluation, robustness, polish, documentation for the university report | 1 |
| module-3/4 | Simulated array (pyroomacoustics), DOA, beamforming, Spatial tab | 2 |
| module-5 | Source tracking + detection fusion, Sources tab | 3 |
