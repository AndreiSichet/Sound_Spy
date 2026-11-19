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
- `TargetSpec(labels, direction, source_id, attenuation_db=30)` (the API fills in
  `extraction.default_attenuation_db` = 15 when a client sends none): one spec for all three tabs —
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
         | "run" | "warning" | "error" | "set_target" | "set_bypass" | "set_method"
         | "set_gate",
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
| `run` | server → client | `RunModel` (`run_id`, `recording_id`, `labels`, `attenuation_db`, `method`, `status`, `error`, `output_path`, `duration_s`, `processing_seconds`, `info`) | on every state change of an offline processing run (step 06); never dropped |
| `set_bypass` | client → server | `{"enabled": bool}` | stored, forwarded to the session; replied with `status` (step 06) |
| `set_method` | client → server | `{"method": "neural_tse" \| "mask_baseline"}` | the extraction method of the **next** session; replied with `status`, or `error` while a session runs (step 07) |
| `set_gate` | client → server | `{"enabled": bool}` | the neural output gate of the **next** session; replied with `status`, or `error` while a session runs (step 07b) |
| `warning` | server → client | `{"code", "message"}` (e.g. `rtf_high`) | when needed, ≤ 1 per `rtf_window_s` |
| `error` | server → client | `{"message": "..."}` | reply to a bad client message |
| `set_target` | client → server | `TargetSpec`: `labels[]`, `direction`, `source_id`, `attenuation_db` | validated against `LabelSet`, stored, forwarded; replied with `status` |

Step 06: `levels` / `waveform` / `spectrogram` payloads carry `source`: `"input"` (the input
`Analyzer`) or `"output"` (the `post_side` analyzer on the chain output), so each exists twice at
its rate. `status` carries `bypass` and `extraction` (`{method, latency_ms, gating,
target_active}`); `latency_ms` includes the chain's algorithmic latency.

Step 07: the output analyzer no longer produces `waveform` (output = `levels` + `spectrogram`).
`status` carries `extraction_method` (the method chosen for sessions) next to
`extraction.method` (what really runs: after a fallback they differ), and for the neural method
`extraction` also has `device`, `model`, `model_ms`, `model_ms_p95` (time of one model call)
and `compiled`.
`warning` code `extraction_fallback`: neural was chosen but its model is unavailable, the
baseline runs. `set_target` without `attenuation_db` uses `extraction.default_attenuation_db`.

Step 07b: `status` carries `extraction_gate` (the gate chosen for sessions) and
`output_drained` (how often the speaker queue was emptied back to its cushion); for the neural
method `extraction` also has `gate` (on / off) and `gate_closed` (share of model steps with more
than half of the gate's depth applied).

On connect the server sends a `status` message, then a `status` heartbeat every
`api.ws_heartbeat_s` seconds (2 s). Unknown or malformed client messages get an `error` reply;
the client ignores unknown types with a console warning. In step 01 only `status` and `error`
are produced (`set_target` answers "not implemented (step-03)").

## REST API

| Route | Step |
|---|---|
| `GET /api/health`, `GET /api/config` | 01 (implemented); step 07: config also returns `extraction_methods` (`[{name, display_name, available, reason}]`), `default_method` (after the fallback) and `default_attenuation_db`; step 07b: `default_gate` |
| `GET /api/devices` | 02 (implemented): 200 with `resolved_input: null` + `error` if the profile does not resolve; 503 if PortAudio is unavailable |
| `GET /api/recordings` | 02 (implemented): re-indexes `data/raw`, newest first |
| `GET /api/session` | 03 (implemented): status, idle if none |
| `POST /api/session/start` | 03 (implemented): `{mode, source: {type: mic\|file, path, loop}, monitor, method?}`; 409 running, 400 bad mode/source (file paths must be inside the project root), 503 no audio backend. `method` (step 07) and `gate` (07b) apply to this session only |
| `POST /api/session/stop` | 03 (implemented): idempotent |
| `PUT /api/session/target` | 03 (implemented): `TargetSpec`, 422 unknown label; `attenuation_db` optional (default 15, echoed back) |
| `PUT /api/session/monitor` | 03 (implemented): `{enabled, gain_db ∈ [−40, 12], muted}`; 409 no output device |
| `POST /api/session/record/start`, `/record/stop` | 03 (implemented): 409 no session / (not) recording; stop syncs the index and returns the `Recording` |
| `GET /api/recordings/{id}`, `GET /api/recordings/{id}/audio` | 03 (implemented): `audio/wav`, 404 if missing |
| `POST /api/recordings/{id}/detect` | 05 (implemented): offline detection timeline; 413 above `detection.offline.max_seconds`, 503 no model |
| `PUT /api/session/bypass` | 06 (implemented): `{enabled}`; kept between sessions |
| `PUT /api/session/method` | 07 (implemented): `{method: neural_tse \| mask_baseline}`; kept between sessions; 409 while a session runs |
| `PUT /api/session/gate` | 07b (implemented): `{enabled}`: the neural output gate; kept between sessions; 409 while a session runs |
| `POST /api/recordings/{id}/process` | 06 (implemented): `{labels (>= 1), attenuation_db?, method?}` → 202 + `RunModel` (`pending`), processed by the one-job worker; 404 file gone, 413 above `extraction.offline.max_seconds`, 422 unknown label, 503 the method's model is missing (neural checkpoint, or the detection model for the baseline). Step 07: `method` = `neural_tse` \| `mask_baseline`, default = the configured method if it can run, else the baseline. Step 07b: `gate?` (neural; default `extraction.neural_tse.gate.enabled`) |
| `GET /api/recordings/{id}/runs` | 06 (implemented): runs of a recording, newest first |
| `GET /api/runs/{run_id}`, `GET /api/runs/{run_id}/audio` | 06 (implemented): 404 unknown; audio 409 unless `done` |

Unimplemented routes return HTTP 501 `{"detail": "not implemented (step-XX)"}` (none since step 06).
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
on API startup and on every `GET /api/recordings`: it adds a row for every unindexed sidecar,
marks a row **`missing`** while its WAV is gone but its sidecar remains (step 06; cleared when the
file returns), and deletes a row whose sidecar is gone (nothing describes it any more). The CLI
(`soundspy record`) writes files only and never touches the database, which keeps the library free
of SQLAlchemy and means a database can always be rebuilt from `data/raw/`. Paths are stored
relative to the project root; times in UTC (SQLite keeps no time zone).

## Database

SQLite at `data/soundspy.db` (SQLAlchemy 2.x, URL in `configs/default.yaml`). Since step 06 the
database is a **disposable index of sidecar files**: nothing is stored only in it.

- `recordings`: id, created_at, path (unique), duration_s, sample_rate, channels, device,
  profile, label?, notes, `missing`. Index of the sidecars in `data/raw/`.
- `processing_runs`: id, `run_id` (unique; the sidecar's stem), created_at, recording_id? →
  recordings (`SET NULL`), recording_path, labels (JSON), attenuation_db, method, activity_mode,
  status (`pending` | `running` | `done` | `failed`), error?, output_path?, duration_s?,
  processing_seconds?, info (JSON). Index of the run sidecars in `outputs/processed/`.
- `meta`: key/value; holds `schema_version`.

**No migrations.** `db/models.SCHEMA_VERSION` (2) is bumped whenever a table changes.
`init_db()` compares it with the stored version: equal → nothing; different, or tables without a
version → `rebuild_db()` (reflect + drop every table, create the current schema), after which
`services/index.sync_all` re-reads the sidecars. `soundspy db rebuild` does the same on demand.
All indexing holds `INDEX_LOCK` (request threads and the run worker both index).

## Extraction (step 06, `soundspy.extraction`)

`dsp/stft.py`: `StftSpec(n_fft, hop)`, sqrt-Hann analysis + synthesis, frame `k` covers
`[k*hop - (n_fft - hop), k*hop + hop)`. `StreamingSTFT.push(chunk)` → `(k, channels, bins)`;
`StreamingISTFT.push(frames)` + `read(n)` returns exactly `n` samples; unity gain = input delayed
by `StftSpec.latency` = `n_fft` samples for any chunking. Offline `stft` / `istft`: same frames,
zero delay.

`MaskBaselineExtractor(Processor)` (1 channel; `configs/extraction/mask_baseline.yaml`): per
frame, `active = ActivitySource.active(labels, centre_s, now_s)`; gate position `o` (open ramp
`open_ms`, hold `hold_ms`, close ramp `close_ms`); noise `N` = recursive average
(`noise_alpha`) over frames confirmed inactive (`lag_s` guard) if younger than
`noise_max_age_s`, else `minstat_bias` × minimum of the smoothed power over `minstat_window_s`;
decision-directed `xi` (`dd_alpha`), Wiener gain `xi / (1 + xi)` floored at
`g_min = 10^(-attenuation_db / 20)`, smoothed (`gain_smoothing`); frame gain
`exp(m * (o * ln G + (1 - o) * ln g_min))`, `m` = ramp into/out of pass-through (empty labels =
unity gain, still through the STFT). No activity source → gate open, minimum statistics only.
`latency_samples = n_fft`.

`ActivitySource`: `LiveDetectorActivity(DetectionState)` (current decision; `lag_s` =
`noise_guard_s`), `TimelineActivity(labels, times, active, clock="centre" | "now")` (step
function; built by `centred_timeline` = offline, windows centred, zero-phase EMA + hysteresis,
and `causal_timeline` = the live detector's windows/EMA/hysteresis + `delay_s`),
`OracleActivity(class_activity, rate_hz)`. `engine/detection_state.DetectionState` is written by
`Detector._drain` and built per chain by `engine/factory.py`.

Chain (`SEMANTIC`): `main = [reducer, extractor]`, `side = [analyzer(input), detector]`,
`post_side = [analyzer(output)]`, `dry_after="reducer"`: `chain.dry` is the reducer's output
delayed by the latency of the later stages (aligned with the output); `Session.set_bypass(True)`
sends it to the sink instead of the output. `Processor.latency_samples` (default 0) sums to
`chain.latency_samples`, added to `Session.latency_ms`. While a session runs, long-lived objects
are frozen for the garbage collector (`gc.freeze()` at start, `gc.unfreeze()` on release).

Offline: `extraction/offline.py` (`run_offline` = block-wise `process` + flush, delay-compensated;
`extract_file`; `start_run` / `execute_run`), `io/runs.py` (run sidecars, atomic rewrite under a
lock, retry on a locked file), `api/services/runs.RunManager` (queue + one `soundspy-runs` thread,
`run` WS messages). Evaluation: `training/evaluate_extraction.py`, five systems (mixture,
timeline, live, oracle, IRM), outputs in `experiments/ext_<method>/` and
`outputs/eval/ext_<method>/`.

## Neural extraction (step 07)

**Model** (`extraction/models/waveformer/model.py`, adapted from Waveformer's `dcc_tf.py`, MIT):
`Net(label_len, L=32, enc_dim=512, num_enc_layers=10, dec_dim=256, num_dec_layers=1,
dec_buf_len=13, dec_chunk_size=13, out_buf_len=4)`, 44.1 kHz, mono in / mono out. Input conv
(kernel `3L`, stride `L`, lookahead `L`) → 10 dilated causal depthwise-separable conv layers
(dilation 1 … 512, state = their context) → × label embedding
(`Linear(label_len, 512) → LN → ReLU → Linear(512, enc_dim) → LN → ReLU`) → causal transformer
decoder over chunks of 13 latent frames (context: the previous 13) → mask × encoder output →
transposed conv with overlap buffer → `tanh`. `Net.stream_sizes = (hop 416, context 32,
lookahead 32)` samples; `Net.stream_step(x, label, buffers)` consumes
`context + k·hop + lookahead` samples and returns `k·hop`; `init_buffers` gives the three state
buffers. `extraction/neural_model.py`: `ensure_pretrained` (download, SHA-256),
`adapt_label_layer` (41 → 7 inputs from `label_init`), `build_finetune_model`,
`parameter_groups` (backbone / label), `save_checkpoint` / `load_checkpoint`
(`{version, model, labels, state, extra}`, `weights_only`).

**`NeuralTSEExtractor(Extractor)`** (`extraction/neural_tse.py`, 1 channel): `dry` FIFO
(latency) ‖ `StreamingResampler` session → model rate → input FIFO → every `block_chunks`
whole steps one `NeuralCore.step` (lock, `inference_mode`) → `StreamingResampler` back →
estimate FIFO primed with zeros. `y = dry + w · (1 − g) · (est − dry)`, `g = 10^(−dB/20)`, `w` =
0 … 1 fade (`ramp_ms`) between pass-through (empty labels: model not run, state reset) and
extraction. `latency_samples = ceil((lookahead + block − 1) · sr / model_sr) + filter delay`
(48 kHz: 487 + 22 = 509). `on_target_change`: `keep` | `reset` the model state. `start()` =
warm-up on the audio thread; with `live.compile` it first traces the step for calls of
`block_chunks … block_chunks + 2` steps (`NeuralCore.compile_steps`: `torch.jit.trace` +
`freeze`, checked against the plain step, run under `torch.jit.optimized_execution(False)`;
other step counts run untraced). `NeuralCore` is cached per `(checkpoint, mtime, device)`;
`extract_whole` = the same maths over a whole signal without streaming (evaluation).
`dsp/resample.StreamingResampler(source_rate, target_rate, channels, taps)`:
`y[m] = Σ_j h[(m·down) mod up + j·up] · x[(m·down) div up − j]`, `ceil(n·up/down)` outputs after
`n` inputs, delay `(taps − 1) / (2·down)` output samples; `taps_for_whole_delay` picks the second
filter's length.

**Method selection** (`extraction/offline.py`): `resolve_method(config, requested)` →
`(method, warning)`: `neural_tse` without a checkpoint file → `mask_baseline` + warning;
`build_extractor(config, activity, method=)` builds exactly that method; `engine/factory.
build_chain(..., method=)` adds the `extraction_fallback` warning (also when the checkpoint
cannot be loaded) and `gating_unavailable` only for the baseline. The neural extractor is
built before the detector: `factory.detector_device` returns `"cpu"` for the detector when
the extractor's core is on CUDA, `live.detector_on_cpu` is true and `detection.live.device`
is `auto` (one model per GPU in a live session). `Processor.start()` /
`ProcessorChain.start()`: called by the session worker on the audio thread before the first
chunk. Run sidecars store `method`, `parameters` (neural: checkpoint path + modification time,
model config, `block_chunks`) and `activity_mode` (`none` for neural).

**Training data** (`data/cache.py`, `data/reverb.py`): `soundspy data cache --task extraction`
renders every recipe once at the model rate into
`<processed>/cache/extraction_<rate>/<split>/shard_NNN.npy` (`(n, 2, samples)` float16: mixture,
target) + `shard_NNN.meta.npz` (`target_hot`, `present_hot`, `empty`, `reverb`, `target_start`,
`target_end`) + `index.json` (signature, `shards_done`). Shards are written as `.part` and
renamed; reruns render only missing shards; another signature is refused without `--force`;
free space is checked first. `recipe_rirs(recipe_id, n_events, rate, cfg, seed)`: with
probability `reverb.prob` a shoebox room (`sample_room`) and one unit-energy RIR per event
(`simulate_rirs`, pyroomacoustics); `mixtures.render(..., rirs=)` convolves each event before
summing, so `mixture == target + rest` holds and the target is reverberant.
`ExtractionCache.pair(i)` → memory-mapped `(2, samples)` view.

**Training** (`training/extraction_data.py`, `training/train_extraction.py`): crops of
`train_crop_s` addressed by `(index, epoch)` (placed on the target's stretch with probability
`crop_target_prob`), gain and noise augmentation, `RotatingSampler` (`epoch_recipes` crops per
epoch, every recipe once per `train_recipes / epoch_recipes` epochs). Loss per example:
`−(snr_weight · SNR + si_snr_weight · SI-SNR)` if the target sounds, else
`empty_weight · max(10·log10(|est|² / |mix|²), empty_floor_db)`. AdamW (two learning rates),
warm-up + cosine per update, AMP, gradient clipping, batch size found by measuring peak memory,
accumulation to `effective_batch_size`. Validation on fixed whole examples, on the system output
at the evaluation attenuation; `best.pt` by SI-SDRi; early stopping; `--resume` from `last.pt`.
Result (`experiments/tse_waveformer_v1`): best epoch 16 of 26, val SI-SDRi +3.72 dB; test
+3.28 dB (streaming = offline), +2.92 dB on the reverberant copy.

**Evaluation** (`training/evaluate_extraction.py`, `--method neural_tse`): the five step-06
systems + `neural_offline` (`extract_whole`), `neural_stream` (the live extractor in 20 ms
chunks) and `neural_live_db` (the stream output re-mixed to the live attenuation), on the same
recipes; breakdowns as in step 06 plus one / several target classes; a second pass on the
**reverberant copy** of the same recipes (rooms from `evaluation.reverb_seed`). Outputs in
`experiments/ext_neural_tse/` and `outputs/eval/ext_neural_tse/`.

## Neural extraction v2 (step 07b)

**Output gate** (`extraction/gate.py`, `gate` in `neural_tse.yaml`): `OutputGate(cfg, model_rate,
block=hop)` works per model step on the estimate `y` and the model input `x` of the same samples:
block mean squares → one-pole envelopes (rise `attack_ms`, fall `release_ms`) → `r = L_y − L_x`
(dB) → wanted gain `0` (`r ≥ open_db`) … `−depth_db` (`r ≤ close_db`), linear in dB → the gain
follows upwards with the attack coefficient, downwards with the release coefficient after
`hold_ms` without a reason to close → linear ramp inside the block. In `NeuralTSEExtractor` it
sits between the model and the up-resampler (no latency); reset with the model state.
`whole_estimates(core, x, sr, labels, gates)` runs the model once and returns the ungated
estimate plus one per gate setting (same filters as streaming); `extract_whole(…, gate)` and
`residual_mix` build the evaluation's whole-file outputs. `build_neural_extractor(config,
gate=None|bool)`, `build_extractor(..., gate=)`, `build_chain(..., gate=)`,
`SessionManager.set_gate`, `start_run(..., gate=)`; run sidecars' `parameters` hold
`model_name` and the full `gate` settings.

**Speaker queue draining** (`io/speaker_sink.py`): after each `write`, if the queue has been
longer than `prebuffer_chunks + 1` for `output.drain_after_s` (1 s), the oldest chunks are
dropped down to `prebuffer_chunks`, the first kept chunk's first `drain_fade_ms` (5 ms) are
crossfaded from the chunk that would have played, and `drained` / `drained_chunks` count it.

**Loss** (`training/extraction_data.py`): `quality_sum / B + leak_weight · Σ_silent-frames
(max(10·log10(E_out/E_mix), leak_floor_db) − leak_floor_db) / (B · F)`, frames of
`leak_frame_ms`, a frame is silent when the target's mean square is below `leak_silence_dbfs`;
quality as in step 07 for examples whose target sounds. Validation adds `leak_frames_db` (the
same frame measure over the silent frames of examples with a target, floor −60) and
`val_empty_examples` target-absent examples; the start model is validated as epoch 0;
`require_leak_not_worse`: an epoch becomes `best.pt` only if its target-absent leak is ≤ the
start's + 0.25 dB; `time_budget_h` (or `--time-budget-h`) stops before an epoch that would not
fit; resumed time is counted.

**Data.** LibriSpeech (`data/collectors/librispeech.py`, alias `librispeech://` →
`<external>/librispeech/LibriSpeech`; `fg:<datasets.librispeech.class>`, split = official
subset from `splits.librispeech`, group = speaker; not counted in the FSD50K validation
rebalance). `data/clip_cache.py`: `soundspy data cache --task clips` writes
`<processed>/cache/clips_<rate>/<dataset>/<path>.wav` (PCM_16, mono) for
`neural_tse.data.clip_cache` datasets/splits; `ClipLibrary.load_segment` reads such a copy when
it exists for the wanted rate. Room bank (`data/reverb.py`): `soundspy data cache --task rirs`
→ `rir_bank_<rate>.npz` (offsets, room facts, signature) + `.npy` (float16 responses, tails cut
at `trim_db`); `RirBank.draw(rng, n)` = one room, distinct positions. Recipe sampling
(`data/mixtures.py`): `RecipeSampler(library, labels, mix, split).sample(rng, id, spec)` is the
only sampler (`generate_recipes` = a seeded loop over it; the frozen files are reproduced
exactly); `MixtureConfig` adds `partial_target_prob` / `partial_switch` (`Recipe.switch_s`),
`hard_negative_prob` / `hard_negative_pairs`, `source_mix` and `lowpass_prob` / `lowpass_hz`;
`DataConfig.dynamic` = `mixtures` + `mixtures.dynamic`. `render(..., lowpass_hz=)` filters
events before levelling; `meta["class_energy"]`. `data/dynamic.py: DynamicMixer.example(epoch,
index)` (seeded by `(seed, epoch, index)`; crop around a random active target moment, or around
the switch of a partial target); `training/datasets.py: DynamicExtractionSet`.

**Evaluation** (`--name`, `--no-reverb`, `--no-benchmarks`): extra systems `neural_v1`
(`evaluation.compare_checkpoint`, whole file), `neural_nogate`, `neural_high_db`
(`evaluation.high_attenuation_db`); `worse` (count of SI-SDRi < −5 dB) and `dominant` (loudest
non-selected class, from `class_energy`) per recipe; benchmark sets `absent`, `pair_absent`,
`pair_a`, `pair_b`, `librispeech` (`evaluation.benchmarks`), frozen in
`<mixtures>/bench_<name>.jsonl` (first line = the settings). `soundspy evaluate gate`
(`training/tune_gate.py`): the model once over the validation set, a grid of gate settings,
SI-SDRi / worse / suppression at 30 and 40 dB, `gate_tuning.json|md` next to the checkpoint.

Result: `experiments/tse_waveformer_v2` (`leak_floor_db` −50) stopped at epoch 15, val +2.32 dB;
`experiments/tse_waveformer_v2b` (`leak_floor_db` −30, the configured checkpoint) best epoch 20 of
30, val SI-SDRi +3.17 dB, target-absent leak −17.8 dB (v1 +3.72 / −14.1). Gate −12 / −18 / 15 dB
(tuned on val, cost 0.30 dB). Test (`experiments/ext_neural_tse_v2/`): +2.51 dB with the gate,
+2.88 without, suppression −17.1 / −15.6 dB (v1 +3.28 / −12.5; v1 + gate,
`experiments/ext_neural_tse_v1_gate/`: +2.97 / −14.0); streaming = offline.

## Hybrid extraction (step 07c)

Method `hybrid` (default, `configs/extraction/hybrid.yaml`): `HybridExtractor`
(`extraction/hybrid.py`) with routes `off` (the input), `speech_enhancer` (exactly
`speech.labels` selected: DeepFilterNet3), `waveformer` (anything else, including speech + other
classes: `WaveformerStream`), optionally `cascade` (`speech.route: cascade`: Waveformer, then
DeepFilterNet3). Every route's estimate is delayed to the slowest route's latency (FIFO pads), so
the extractor has one fixed `latency_samples` (1919 at 48 kHz). A selection change starts the
new route's network at once; its weight ramps up (`live.ramp_ms`) only one latency later (its
estimate is valid from then), the old one ramps down and rests afterwards. Resting routes are
not called at all; a route that starts again is `reset()` and its pad re-primed (saves ~1 ms
per chunk on the live Waveformer route). `y = G · (est + g · (x − est))`, `G` from
the gate. `build_hybrid_extractor(config, activity, device=, gate=)` notes a missing route
(`notes`, warning `hybrid_route_unavailable`); neither route → `HybridUnavailable` →
`resolve_method` / `build_chain` fall back to `mask_baseline` with `extraction_fallback`.
`detector_device` looks at `HybridExtractor.core` (the Waveformer's). `start()` pins the audio
thread to the performance cores (`engine/cpu.py`, `GetSystemCpuSetInformation` →
`SetThreadAffinityMask`; no-op on single-class CPUs and off Windows).

**`WaveformerStream`** (`extraction/neural_tse.py`): the resampling bridge, model steps and
optional 07b output gate taken out of `NeuralTSEExtractor` (which now mixes on top of it);
`process(x, labels | None)`, `latency`, `start()`, `request_reset()`, `model_ms()`.

**Speech enhancer.** `extraction/models/deepfilternet/` (MIT / Apache-2.0, headers kept):
`dsp.py` = libDF's STFT (Vorbis window 960, hop 480, `wnorm = 2·hop/fft²`, unnormalised inverse),
ERB widths (`erb_fb`), ERB dB features with running mean (init −60…−90 dB) / 40, unit-normalised
96 complex bins (init 0.001…0.0001), `alpha = 0.99`; `model.py` = DfNet3 with the checkpoint's
module names, `forward` (whole sequence, as upstream) and `stream(spec, erb, low, state)` with
state = (2 feature frames ×2, 4 spectra, 4 c0 frames, 3 GRU states); the first
`conv_lookahead` frames of a stream only enter the spectrum window (`push_spectra`), resting
frames move the windows and clear the recurrent state (`push_features`); `deep_filter_real`
(no complex tensors, ONNX-exportable); `fuse_for_inference` replaces depthwise 1×1 convolutions
by `ChannelScale` (PyTorch ran them as 64 calls each on one CPU thread). `speech_model.py`:
`ensure_weights` (zip + SHA-256 from `hybrid.yaml`, unpacked to
`<model_cache>/deepfilternet/DeepFilterNet3/`), `read_params(config.ini)`, `load_model` (strict,
`weights_only`), `export_onnx` (opset 17, dynamic frame count, cached next to the model as
`DeepFilterNet3_stream_v<EXPORT_VERSION>_<sha12>.onnx`). `speech_enhancer.py`: `SpeechCore`
(backend `onnx`: ONNX Runtime CPU session, NumPy state, checked against PyTorch within 1e-4 at
load; `torch`: any device; `whole()` always PyTorch), `SpeechEnhancer.process(x, active)`
(latency `(hop − 1) + conv_lookahead·hop + (fft − hop)` = 1919), `enhance_whole`,
`load_speech_core` (cached). `soundspy data download deepfilternet` downloads, unpacks and
exports.

**`HybridGate`** (`extraction/hybrid_gate.py`, `gate` in `hybrid.yaml`): blocks of `block_ms`
counted from the stream start; per completed block: one-pole envelopes of est and input
(`attack_ms` / `release_ms`) → `ratio_db`; *rise* = ratio > `reopen_ratio_db` and input above
`min_level_dbfs` and no warm-up left; presence from the `ActivitySource` every 10 blocks
(`active`, and `absent(labels, centre, now, close_prob_factor)`: every selected class's smoothed
probability below factor × its threshold); close when confidently absent for `close_hold_ms` and
no rise for `close_hold_ms`, ramp `close_ms`; otherwise open, ramp `open_ms`; the next block
ramps linearly (in dB) to the new position; closed gain `10^(−attenuation_db/20)`.
`restart(warmup)` after a speech-enhancer route starts: rises ignored for `warmup_ms`;
`start_closed` (with an activity source) starts a stream closed. `ActivitySource.absent`
(default: not active); `LiveDetectorActivity(state, lag_s, thresholds)` (the factory passes the
detector core's thresholds); `TimelineActivity(..., probabilities=, thresholds=)` (both timeline
builders store their smoothed probabilities). Offline (`extract_buffer`) uses the centred
timeline; without a detector the gate stays open (note in the run info).

**API / UI.** `/api/config`: `hybrid` in `extraction_methods`, `default_gates` per method;
`default_gate` = the default method's. `SessionManager.selected_gate` = the chosen method's own
default unless switched; choosing another method resets the switch to that method's default.
Status `extraction`: `route`, `routes`, `gate_state` (open | closed | off), `gate_closed`,
`gate_reopened`, `speech_enhancer` / `waveformer` (model, device, latency_ms, model_ms(_p95)),
`performance_cores`, `notes`. Frontend: "Hybrid" in the selector, the Gate switch for Hybrid
and Neural, session bar `Hybrid · speech enhancer · cpu (onnx) · gate closed · +39.98 ms`.

**Evaluation** (`training/evaluate_hybrid.py`, `--method hybrid`): systems `wf`, `wf_hgate`,
`dfn`, `dfn_hgate`, `cascade`, `cascade_hgate` (Speech-only selections have the last four),
`hybrid_offline` / `hybrid_live` (centred / causal timeline), `hybrid_stream` (20 ms chunks
through `HybridExtractor`, main set only), `hybrid_live_db`, `hybrid_high_db`; whole-file
outputs via `gate_output` (the same gate over `latency` zeros + the signal); per-recipe
`interrupt` (`metrics.interruption_rate`); `speech_only` summary. `summarize` / CSV accept
systems missing for some recipes. `soundspy evaluate hybrid-gate`: the configured route once
over `--n` val recipes + `--bench-n` val copies of the pair benchmarks
(`<mixtures>/bench_val_<name>.jsonl`, `benchmark_sets(split=, counts=)`), every combination of
`tuning`, live (causal) timeline; `experiments/ext_hybrid/gate_tuning.json|md`.
Real-room benchmark: `soundspy record-benchmark` → `data/benchmark/<session>/<part>_NN.wav`
(sidecar `benchmark_session`, `part`); `training/real_benchmark.py` (`build_cases`,
`score_case`, `model_systems`, `evaluate_real`) → `experiments/ext_hybrid/real_room.json|md`.

Result (`experiments/ext_hybrid/`): gate 400 ms / 0.5 / −6 dB, warm-up 1 s (val), speech input gain +10 dB (`SpeechEnhancer(input_gain_db=)`, `enhance_whole(input_gain_db=)`, `HybridCores.speech_gain_db`). Test +3.20 dB offline = streaming, +2.06 with the live detector (v1 +3.28); music only with Speech selected −45.2 dB at 30 dB, −56.0 at 40 dB; speech over music +7.0 offline / +4.0 live (v1 +5.33); real room at 0 dB +12.0 / +11.8 (classical), +10.7 (pop), speech-only interruption 0.2 %; owner's classical recording −25.4 dB (the first 2 s of the stream). Live: Speech route RTF p95 0.39, Waveformer route 0.72 (worst 0.86, as Neural alone), 144 ms.

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
`fsd50k://` → `<external>/fsd50k`, `custom://` → `<custom>`, `librispeech://` →
`<external>/librispeech/LibriSpeech` (07b). The index stores only aliases.

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
18. **Large data on D:.** Datasets, indexes and model caches live under the configured `paths`.
19. **The database is a disposable index of sidecar files.** Nothing is stored only in the DB,
    so it can be rebuilt at any time and schema changes need no migrations.

## Step plan (numbers used in `TODO(...)` markers)

| Step | Content | Tab |
|---|---|---|
| 01 | Architecture restructure + app shell | - |
| 02 | Audio capture: devices, profiles, sources/sinks, storage, levels, ChannelReducer; CLI `devices`, `check-mic`, `record`; `GET /api/devices` | - |
| 03 | Live session: engine worker, chunker, speaker sink, Analyzer; WS levels + spectrogram; record button; Semantic tab live monitor | 1 |
| 04 | Data: ontology, collectors (ESC-50, FSD50K, own recordings), splits, foreground/background mixtures; `record-dataset` — **done** (data on D:, recipes in JSONL, `soundspy data …`) | - |
| 05 | Detection: features, multi-label fine-tuning of a pretrained model, chunked inference, detections over WS — **done** (EfficientAT mn10_as, `detection/`, `training/`) | 1 |
| 06 | Extraction baseline (detector-guided masking) + SI-SDR metrics; offline processing with A/B playback — **done** (detector-gated Wiener filter, `extraction/`, run sidecars, rebuildable DB) | 1 |
| 07 | Neural TSE (Waveformer fine-tuned, label-conditioned, streaming); live extraction in the Semantic tab — **done** (`extraction/neural_tse.py`, cache + reverb, method selector) | 1 |
| 07b | Neural extraction v2: silent-frame loss, output gate, LibriSpeech, dynamic mixing, speaker-queue draining — **done** | 1 |
| 07c | Hybrid extraction: DeepFilterNet3 for Speech, Waveformer v1 for the rest, detector-closed / output-reopened gate, real-room benchmark — **done** | 1 |
| 08 | End-to-end evaluation, robustness, polish, documentation for the university report | 1 |
| module-3/4 | Simulated array (pyroomacoustics), DOA, beamforming, Spatial tab | 2 |
| module-5 | Source tracking + detection fusion, Sources tab | 3 |
