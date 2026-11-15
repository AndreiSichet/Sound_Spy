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

### Session lifecycle (implemented in step 03)

```
SessionManager.start(request)                                   (API thread pool, under a lock)
   ├─ build_chain(mode)        SEMANTIC: [ChannelReducer(mean) → extractor=Passthrough] + [Analyzer]
   ├─ source                   MicSource(resolve_input(profile), capture)  |  FileSource(path, realtime, loop)
   ├─ output                   SpeakerSink(resolve_output(output.device), output)  (None if unavailable)
   └─ Session.start()          source.open → chain.configure(sr, ch) → output.open(sr, out_ch)
                               → worker thread "soundspy-session" → status event

worker loop, per chunk (engine.chunk_ms = 20 ms):
   apply pending target / monitor  →  chain.process(chunk)  →  output.write(out) if monitoring
   →  recorder.write(chunk) (raw input, under _rec_lock)  →  emit events  →  status every 1 s,
   warning if p95 RTF > engine.rtf_warning over engine.rtf_window_s

Session.stop()  (idempotent)    set stop flag → source.close() (unblocks the worker) → join
                                → close output → finish an active recording → status "idle"
worker error                    state "error" + message → status event → release devices;
                                never propagates to the API process
app shutdown (lifespan exit)    SessionManager.shutdown() stops the session: no thread outlives the app
```

**Thread → asyncio bridge.** `Session` calls `on_event(event)` on the worker thread.
`SessionManager._on_event` only calls `loop.call_soon_threadsafe(self._publish, event)` (the loop
captured in the lifespan) and returns. `_publish` runs on the event loop, converts the event to a
`WsMessage` and puts it into every client's queue. Each WebSocket client has a bounded
`ClientQueue` (`api.ws_client_queue` = 64) drained by its own sender task; when full, the oldest
`levels` / `waveform` / `spectrogram` message is dropped, `status` / `warning` / `error` never.

**Input backpressure.** `MicSource` drops the oldest chunk when its queue
(`capture.queue_chunks` = 32) is full and counts it (`dropped_chunks`). **Output.** `SpeakerSink`
plays after `output.prebuffer_chunks` (2), drops the oldest above `output.max_queue_chunks` (8),
plays silence and counts an underrun when empty, then re-prebuffers; gain → 1→N upmix →
peak limiter (instant attack, `limiter_release_ms` release, `limiter_ceiling_dbfs` ceiling, final
hard clip).

## WebSocket protocol (`/ws/session`)

Every message, in both directions, uses one envelope:

```json
{"type": "status" | "levels" | "waveform" | "spectrogram" | "detections" | "sources"
         | "warning" | "error" | "set_target",
 "t": 12.34,
 "payload": { ... }}
```

- `t`: seconds since session start (0.0 when no session is running).
- `payload` = the event's fields without `t`:

| type | direction | payload | rate (step 03 defaults) |
|---|---|---|---|
| `status` | server → client | `EngineStatus`: `state`, `mode`, `device`, `sample_rate`, `channels`, `message`, `source`, `output_device`, `monitoring`, `muted`, `gain_db`, `recording`, `recording_seconds`, `latency_ms`, `chunk_ms`, `chunks_processed`, `proc_ms_mean`, `proc_ms_p95`, `rtf_p95`, `input_overflows`, `output_underruns`, `dropped_chunks`, `output_dropped`, `detector` (`{model, device, hop_s, latency_ms, skipped}` or null, step 05) | on connect, on every change, 1/s while running (`engine.status_interval_s`), 1 per 2 s heartbeat when idle |
| `levels` | server → client | `LevelFrame`: `rms_dbfs[]`, `peak_dbfs[]`, `clipping[]` (per input channel, 0.1 dB) | 20/s (`analysis.levels_hz`) |
| `waveform` | server → client | `WaveformFrame`: `mins[]`, `maxs[]` (64 pairs, mono mix) | 25/s (`analysis.waveform_hz`) |
| `spectrogram` | server → client | `SpectrogramFrame`: `magnitudes_db[]` (128 mel bands, floor −100), `freq_min_hz`, `freq_max_hz` | 25/s (`analysis.spectrogram_hz`) |
| `detections` | server → client | `DetectionFrame`: `probabilities` (every class, smoothed) and `active` (class IDs, threshold + hysteresis); `t` = end of the analysed 1 s window | every `detection.live.hop_s` (0.25 s) while a detector runs (step 05) |
| `sources` | server → client | `{"items": [SoundSource, ...]}` | module 5 |
| `warning` | server → client | `{"code", "message"}` (e.g. `rtf_high`) | when needed, ≤ 1 per `rtf_window_s` |
| `error` | server → client | `{"message": "..."}` | reply to a bad client message |
| `set_target` | client → server | `TargetSpec`: `labels[]`, `direction`, `source_id`, `attenuation_db` | validated against `LabelSet`, stored, forwarded; replied with `status` |

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
| `GET /api/session` | 03 (implemented): status, idle if none |
| `POST /api/session/start` | 03 (implemented): `{mode, source: {type: mic\|file, path, loop}, monitor}`; 409 running, 400 bad mode/source (file paths must be inside the project root), 503 no audio backend |
| `POST /api/session/stop` | 03 (implemented): idempotent |
| `PUT /api/session/target` | 03 (implemented): `TargetSpec`, 422 unknown label |
| `PUT /api/session/monitor` | 03 (implemented): `{enabled, gain_db ∈ [−40, 12], muted}`; 409 no output device |
| `POST /api/session/record/start`, `/record/stop` | 03 (implemented): 409 no session / (not) recording; stop syncs the index and returns the `Recording` |
| `GET /api/recordings/{id}`, `GET /api/recordings/{id}/audio` | 03 (implemented): `audio/wav`, 404 if missing |
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

## Training data (step 04, `soundspy.data`)

Storage (`paths` in `default.yaml`; large data never inside the repository):

| key | default | content |
|---|---|---|
| `external` | `D:/SoundSpyData/external` | `<dataset>/` = extracted data + `archives/` + `SOURCE.json` + `.done` |
| `processed` | `D:/SoundSpyData/processed` | `clip_index.parquet` (+ `.csv`), `activity/<dataset>.npz` |
| `mixtures` | `D:/SoundSpyData/mixtures` | `<split>.jsonl` recipes |
| `model_cache` | `D:/SoundSpyData/models` | `TORCH_HOME`/`HF_HOME`/`XDG_CACHE_HOME` sub-folders (set at CLI/API start if unset) |
| `custom` | `data/custom` | `<class>/<class>_NNNN.wav` + sidecar (`class`, `session_id`) |

Path aliases (`data/aliases.py`): `esc50://` → `<external>/esc50/ESC-50-master`,
`fsd50k://` → `<external>/fsd50k`, `custom://` → `<custom>`. The index stores only aliases.

Clip index columns: `clip_id, dataset, path, duration_s, sample_rate, channels, mids (list),
raw_labels (list), role, split, group, license`. `role` ∈ `fg:<class_key>` | `bg` | `excluded`.

Role rule (`data/roles.py`): `F(c)` = descendants of the class mids, `A(c)` = their ancestors.
`M'` = clip mids minus mids that are ancestors of another mid of the clip. `fg:c` iff
`M' ⊆ F(c) ∪ A(c)` and `M' ∩ F(c) ≠ ∅` for exactly one `c`; several → `excluded` (overlap);
`bg` iff no mid of `M` is in any `F(c)`; else `excluded`. Custom clips take the role of their
recorded class (`background` → `bg`).

Splits (`data/splits.py`): FSD50K eval → test; dev train/val from the proposed split made
uploader-disjoint (`majority_split`: each uploader to the side with most of its clips; ties →
train; 2,010 clips moved on the real data); ESC-50 folds from
`splits.esc50`; custom by session (`split_groups`, seeded). `prepare` fails if a `(dataset, group)`
spans two splits. Recipes use clips of their own split only.

Activity: RMS envelope in dBFS per `activity.hop_s`; active = `env ≥ peak − threshold` and
`> −90 dBFS`. Stored per dataset as concatenated `env_db` + `offsets` + `clip_ids` + header facts.

Recipe (one JSON object per line):

```json
{"id": "train-000001", "duration_s": 6.0,
 "events": [{"clip_id": "...", "role": "fg|bg", "class": "<key>|null",
             "clip_offset_s": 0.0, "onset_s": 0.0, "dur_s": 0.0, "gain_db": 0.0}],
 "present": ["<key>"], "target": ["<key>"], "target_to_rest_db": 0.0}
```

`render(recipe, sample_rate, library, labels, *, reference_rms_dbfs, peak_limit,
activity_rate_hz=10) -> RenderedMixture`: segments read + downmixed (`ChannelReducer`) +
resampled (`dsp/resample.py`, `resample_poly`); each normalised to `reference_rms_dbfs` over its
active frames, `gain_db` applied, summed into target (fg events of target classes) or rest;
target scaled to `target_to_rest_db` (RMS over the whole mixture; skipped if either side is
silent); `mixture = target + rest` in float32; if `peak > peak_limit`, target and rest are scaled
by the same factor. Returns `mixture`, `target`, `rest` (`AudioBuffer (n, 1)`), `present` /
`target_hot` multi-hots, `class_activity (frames, classes)`. Deterministic. Spatial rendering:
`TODO(module-3)` in `render`.

## Detection (step 05, `soundspy.detection`, `soundspy.training`)

`DetectorNet`: waveform `(B, n @ 32 kHz)` → zero-pad to `input_context_s` (10 s) → `EfficientATMel`
(`dsp/features.py`, parity-tested against EfficientAT) → backbone `features` → `classifier[:5]`
(1280-d) → `head` (`Linear(1280, len(LabelSet))`). The backbone's own 527-row AudioSet layer is
frozen and used for head initialisation (mean of the rows of each class's `audioset` names) and
the zero-shot baseline (max over rows in `F(c)`). Row lists live in the checkpoint
(`ModelSpec.rows`), so inference never needs the ontology.

Checkpoint (`torch.save` dict): `version`, `spec` (`ModelSpec` JSON), `model` (state dict), plus
`epoch`/`val_map` (best) or optimizer/scheduler/scaler/`batch_size`/`pos_weight` (last).
`thresholds.json` (`{"thresholds": {label: float}}`) sits two folders above `best.pt`.

Windows: 1 s cut from a rendered recipe; label = class active ≥ `min_active_s` (0.1 s) in the
window (`render()` activity at `activity_rate_hz` = 100). `DetectionDataset` items are
`(index, epoch)` from `EpochSampler` (training) or `index` (eval, fixed epoch); seeded by
`(seed, index, epoch)`. Workers: one BLAS/OpenMP thread each, `timeout = loader_timeout_s`.

Live: `Detector(SideProcessor)` in the `SEMANTIC` chain (name `detector`): mono ring buffer of
`window_s` at the native rate; a window every `hop_s` samples-exactly; one inference thread
(`soundspy-detector`), skip-when-busy (`skipped`); `Smoother` (EMA `ema_alpha`, hysteresis);
emits `DetectionFrame`. The inference thread warms the model up itself and `configure` waits
for it (CUDA initialises per thread, ~3 s); on the CPU it limits itself to `live.cpu_threads`
PyTorch threads (per-thread setting; idle PyTorch threads otherwise busy-wait). `ProcessorChain.close()` stops the thread when the session releases. No
checkpoint / load failure → chain without detector + `chain.warnings` → session emits
`WarningEvent(code="detector_unavailable")`. Offline: `detect_timeline()` (same windows/hop/
smoothing; overlapping or touching segments of a class merged), `POST /api/recordings/{id}/detect` (413 above `offline.max_seconds`, 503 no model).

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
| 04 | Data: ontology, collectors (ESC-50, FSD50K, own recordings), splits, foreground/background mixtures; `record-dataset` — **done** (data on D:, recipes in JSONL, `soundspy data …`) | - |
| 05 | Detection: features, multi-label fine-tuning of a pretrained model, chunked inference, detections over WS — **done** (EfficientAT mn10_as, `detection/`, `training/`) | 1 |
| 06 | Extraction baseline (detector-guided masking) + SI-SDR metrics; offline processing with A/B playback | 1 |
| 07 | Neural TSE (Waveformer-style, label-conditioned, streaming); live extraction in the Semantic tab | 1 |
| 08 | End-to-end evaluation, robustness, polish, documentation for the university report | 1 |
| module-3/4 | Simulated array (pyroomacoustics), DOA, beamforming, Spatial tab | 2 |
| module-5 | Source tracking + detection fusion, Sources tab | 3 |
