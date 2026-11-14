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

export interface ConfigResponse {
  classes: ClassInfo[]
  modes: ModeInfo[]
  device_profile: string
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
}

export interface LevelFrame {
  rms_dbfs: number[]
  peak_dbfs: number[]
  clipping: boolean[]
}

export interface WaveformFrame {
  mins: number[]
  maxs: number[]
}

export interface SpectrogramFrame {
  magnitudes_db: number[]
  freq_max_hz: number
  freq_min_hz?: number
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
}

export interface Detection {
  label: string
  probability: number
  t_start: number
  t_end: number
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
  | Envelope<'detections', { items: Detection[] }>
  | Envelope<'sources', { items: SoundSource[] }>
  | Envelope<'warning', WarningPayload>
  | Envelope<'error', ErrorPayload>

export type ClientMessage = Envelope<'set_target', TargetSpec>

export type ServerMessageType = ServerMessage['type']

export const SERVER_MESSAGE_TYPES: readonly ServerMessageType[] = [
  'status',
  'levels',
  'waveform',
  'spectrogram',
  'detections',
  'sources',
  'warning',
  'error',
]
