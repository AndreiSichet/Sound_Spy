// TypeScript mirrors of the backend schemas (backend/src/soundspy/api/schemas.py).
// Hand-written for now; `npm run gen:api` will generate them from /openapi.json later.

export type Mode = 'semantic' | 'spatial' | 'sources'

export interface HealthResponse {
  status: 'ok'
  version: string
}

export interface ClassInfo {
  /** Internal ID (used in TargetSpec.labels); never shown to users. */
  name: string
  /** What the UI shows on toggle buttons. */
  display_name: string
  /** Tooltip text. */
  description: string
}

export interface ModeInfo {
  mode: Mode
  available: boolean
  reason: string | null
}

/** How the selected sounds are extracted (step 07). */
export type ExtractionMethod = 'neural_tse' | 'mask_baseline'

export interface MethodInfo {
  name: ExtractionMethod
  /** What the selector shows ("Neural", "Baseline"). */
  display_name: string
  available: boolean
  /** Why not, when unavailable (e.g. the model is not trained yet). */
  reason: string | null
}

export interface ConfigResponse {
  classes: ClassInfo[]
  modes: ModeInfo[]
  device_profile: string
  /** Step 07. Optional so that an older backend still works. */
  extraction_methods?: MethodInfo[]
  /** The method a session uses unless another is chosen. */
  default_method?: string
  /** Where the attenuation slider starts (live default: 15 dB). */
  default_attenuation_db?: number
}

export interface Direction {
  azimuth_deg: number
  width_deg: number
}

export interface TargetSpec {
  labels: string[]
  direction: Direction | null
  source_id: string | null
  attenuation_db: number
}

// --- WebSocket payloads (event fields without `t`, which lives in the envelope) ---

export interface EngineStatus {
  state: 'idle' | 'running' | 'error'
  mode: string | null
  device: string | null
  sample_rate: number | null
  channels: number | null
  message: string
  source?: string | null
  output_device?: string | null
  monitoring?: boolean
  muted?: boolean
  gain_db?: number
  recording?: boolean
  recording_seconds?: number
  latency_ms?: number | null
  chunk_ms?: number | null
  chunks_processed?: number
  proc_ms_mean?: number | null
  proc_ms_p95?: number | null
  rtf_p95?: number | null
  input_overflows?: number
  output_underruns?: number
  dropped_chunks?: number
  output_dropped?: number
  /** Present while a detection model runs in the session (step 05). */
  detector?: DetectorStatus | null
  /** The speakers get the unprocessed (delay-matched) signal instead of the output (step 06). */
  bypass?: boolean
  /** The session's extractor (step 06): what is really running. */
  extraction?: ExtractionStatus | null
  /** The method chosen for sessions (step 07); `extraction.method` can differ after a
   *  fallback (no neural model -> baseline). */
  extraction_method?: string | null
}

export interface ExtractionStatus {
  method: string
  /** Its algorithmic delay; already part of EngineStatus.latency_ms. */
  latency_ms: number
  /** A detector tells it whether the selected sounds are present. */
  gating: boolean
  target_active: boolean
  /** Neural method only (step 07): where the model runs and how long one call takes. */
  device?: string | null
  model?: string | null
  model_ms?: number | null
  model_ms_p95?: number | null
  /** The network's step runs as a traced (TorchScript) module. */
  compiled?: boolean | null
}

/** Which side of the processing chain a visual frame shows. */
export type FrameSource = 'input' | 'output'

export interface DetectorStatus {
  model: string
  device: string
  hop_s: number
  /** Mean inference time per window. */
  latency_ms: number | null
  /** Windows skipped because the model was still busy with the previous one. */
  skipped: number
}

/** Live detection (WebSocket `detections`, one per hop; `t` = end of the analysed window). */
export interface DetectionFrame {
  /** Smoothed probability per class ID (every class). */
  probabilities: Record<string, number>
  /** Class IDs currently switched on (threshold + hysteresis). */
  active: string[]
}

export interface LevelFrame {
  rms_dbfs: number[]
  peak_dbfs: number[]
  clipping: boolean[]
  source?: FrameSource
}

export interface WaveformFrame {
  mins: number[]
  maxs: number[]
  source?: FrameSource
}

export interface SpectrogramFrame {
  magnitudes_db: number[]
  freq_max_hz: number
  freq_min_hz?: number
  source?: FrameSource
}

export interface WarningPayload {
  code: string
  message: string
}

// --- REST: session and recordings ---

export interface SourceRequest {
  type: 'mic' | 'file'
  path?: string
  loop?: boolean
}

export interface SessionStartRequest {
  mode: Mode
  source: SourceRequest
  monitor?: boolean
  method?: ExtractionMethod
}

export interface MonitorRequest {
  enabled: boolean
  gain_db: number
  muted: boolean
}

export interface RecordStartRequest {
  label?: string | null
  notes?: string
}

export interface Recording {
  id: number
  created_at: string
  path: string
  duration_s: number
  sample_rate: number
  channels: number
  device: string
  profile: string
  label: string | null
  notes: string
  /** The sidecar is there but the WAV file is gone: listed, but not playable. */
  missing?: boolean
}

export interface ProcessRequest {
  labels: string[]
  attenuation_db?: number
  method?: ExtractionMethod
}

/** One offline processing of a recording (step 06). */
export interface Run {
  run_id: string
  created_at: string
  recording_id: number | null
  recording_path: string
  labels: string[]
  attenuation_db: number
  method: string
  activity_mode?: string
  status: 'pending' | 'running' | 'done' | 'failed'
  error: string | null
  output_path: string | null
  duration_s: number | null
  processing_seconds: number | null
  info?: Record<string, unknown>
}

export interface Detection {
  label: string
  probability: number
  t_start: number
  t_end: number
}

/** Offline detection of a recording (POST /api/recordings/{id}/detect). */
export interface DetectionTimeline {
  duration_s: number
  hop_s: number
  window_s: number
  labels: string[]
  /** End time of each analysed window (seconds). */
  times: number[]
  probabilities: Record<string, number[]>
  segments: Detection[]
}

export interface SoundSource {
  id: string
  label: string
  direction: Direction
  activity: number
  last_seen: number
}

export interface ErrorPayload {
  message: string
}

// --- WebSocket envelope: {"type", "t", "payload"} ---

export interface Envelope<T extends string, P> {
  type: T
  t: number
  payload: P
}

export type ServerMessage =
  | Envelope<'status', EngineStatus>
  | Envelope<'levels', LevelFrame>
  | Envelope<'waveform', WaveformFrame>
  | Envelope<'spectrogram', SpectrogramFrame>
  | Envelope<'detections', DetectionFrame>
  | Envelope<'sources', { items: SoundSource[] }>
  | Envelope<'run', Run>
  | Envelope<'warning', WarningPayload>
  | Envelope<'error', ErrorPayload>

export type ClientMessage =
  | Envelope<'set_target', TargetSpec>
  | Envelope<'set_bypass', { enabled: boolean }>
  | Envelope<'set_method', { method: ExtractionMethod }>

export type ServerMessageType = ServerMessage['type']

export const SERVER_MESSAGE_TYPES: readonly ServerMessageType[] = [
  'status',
  'levels',
  'waveform',
  'spectrogram',
  'detections',
  'sources',
  'run',
  'warning',
  'error',
]
