# Architecture

The precise technical reference: layers, contracts, protocols, step plan. For a plain-language
explanation of how the app works (with a glossary), read
[`project_insights.md`](../project_insights.md) first.

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
| `GET /api/devices` | 02 (implemented): 200 with `resolved_input: null` + `error` if the profile does not resolve; 503 if PortAudio is unavailable |
| `GET /api/recordings` | 02 (implemented): re-indexes `data/raw`, newest first |
| `POST /api/session/start`, `POST /api/session/stop`, `PUT /api/session/target` | 03 |
| `GET /api/recordings/{id}`, `GET /api/recordings/{id}/audio` | 03 |
| `POST /api/recordings/{id}/process` | 06 |

Unimplemented routes return HTTP 501 `{"detail": "not implemented (step-XX)"}`.
Interactive docs: http://127.0.0.1:8765/docs.

## Audio capture: from device to chunks

```
backend/configs/devices/<profile>.yaml
        │  device / hostapi / sample_rate / raw_channels / geometry   (validated: config.ConfigError)
        ▼
io/devices.resolve_input(profile) ──► ResolvedInput
        │  device (WASAPI twin preferred), open_channels = max(raw_channels)+1 or all,
        │  channel_indices to keep, sample_rate (native unless set), wasapi_auto_convert
        ▼
io/mic_source.MicSource ── one sd.InputStream, callback on PortAudio's thread:
        │     indata[:, raw_channels]  (hardware selection, not a downmix)
        │     drop leading digital zeros after open (≤ warmup_max_s; WASAPI start-up silence)
        ├──► read(seconds): fills a preallocated buffer, exactly round(seconds·sr) frames
        └──► chunks(n):     cuts fixed n-frame chunks into a bounded deque;
                            full → drop the OLDEST chunk (stats.dropped_chunks)
        no audio for 2 s → MicStallError; close() / leaving chunks() always stops the stream

io/file_source.FileSource ── same interface from a WAV (raw_channels applied if a profile is
        given; last chunk zero-padded with metadata["valid_frames"]; realtime= and loop= options)
io/file_sink.FileSink ── writes chunks incrementally (32-bit float WAV), honours valid_frames
```

**Host-API twins.** On Windows each physical device is listed once per host API. When the
profile names no host API, the twin on the most preferred one is chosen: WASAPI → DirectSound →
MME → WDM-KS (then Core Audio, ALSA). MME truncates names to 31 characters, so twins are matched
case-insensitively on the first 31 characters. WASAPI streams are opened with
`WasapiSettings(auto_convert=True)` so a non-native `sample_rate` is accepted.

**Drop-oldest.** In live use fresh audio matters more than complete audio: a consumer that falls
behind gets the newest chunks, and the drop is counted rather than hidden.

## Recordings: the sidecar is the source of truth

`io/storage.save_recording` writes `<label|rec>_<YYYYmmdd-HHMMSS-mmm>.wav` (32-bit float, native
rate, every kept channel) and a JSON sidecar with the same stem (device, host API, profile,
channel indices, label, notes, overflows, app version, …). Files are created exclusively, so
nothing is ever overwritten.

The database is only an **index** of the sidecars. `api/services/recordings.sync_recordings` runs
on API startup and on every `GET /api/recordings`: it adds a row for every unindexed sidecar and
**deletes** rows whose WAV is gone (a later sync re-adds the file if it comes back). The CLI
(`soundspy record`) writes files only and never touches the database, which keeps the library free
of SQLAlchemy and means a database can always be rebuilt from `data/raw/`. Paths are stored
relative to the project root; times in UTC (SQLite keeps no time zone).

## Database

SQLite at `data/soundspy.db` (SQLAlchemy 2.x, URL in `configs/default.yaml`), tables created on
startup by `init_db()`:

- `recordings`: id, created_at, path (unique), duration_s, sample_rate, channels, device,
  profile, label?, notes. An index of the sidecars in `data/raw/` (see above).
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
13–17. Git read-only for Claude Code; adapted third-party code keeps its license header and is
listed in the "Third-party code" section of `project_insights.md`; scope discipline; the step
report is given in chat; `project_insights.md` is updated every step.

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
