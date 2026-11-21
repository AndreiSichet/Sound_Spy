// Global UI state (Zustand): connection, active tab, backend config, latest status, recordings
// and their processing runs, toasts, the latest detection (4 per second, for the class chips),
// the selected classes and the attenuation of everything else.
// High-rate visual frames and the detection history are NOT here: they live in
// components/viz/liveFrames.ts.
import { create } from 'zustand'
import { DEFAULT_ATTENUATION_DB } from '../api/target.ts'
import type { SocketState } from '../api/ws.ts'
import type {
  ConfigResponse,
  DetectionFrame,
  EngineStatus,
  MeasuredLatency,
  Mode,
  Recording,
  Run,
} from '../api/types.ts'

export type BackendState = 'unknown' | 'up' | 'down'

export interface Toast {
  id: number
  kind: 'warning' | 'error' | 'info'
  message: string
}

export interface AppState {
  backend: BackendState
  backendVersion: string | null
  ws: SocketState
  activeTab: Mode
  config: ConfigResponse | null
  status: EngineStatus | null
  recordings: Recording[]
  toasts: Toast[]
  detection: (DetectionFrame & { t: number }) | null
  /** Class IDs the user selected as the target (sent as TargetSpec.labels). */
  selected: string[]
  /** How much non-target sound is turned down, in dB (TargetSpec.attenuation_db). */
  attenuation: number
  /** The user moved the slider: the backend's default no longer replaces the value. */
  attenuationTouched: boolean
  /** Processing runs by recording id, newest first. */
  runs: Record<number, Run[]>
  /** Step 08: the latest `soundspy measure-latency` result (null: never measured). */
  measuredLatency: MeasuredLatency | null
  setMeasuredLatency: (latency: MeasuredLatency | null) => void
  setBackend: (state: BackendState, version?: string | null) => void
  setWsState: (state: SocketState) => void
  setActiveTab: (tab: Mode) => void
  setConfig: (config: ConfigResponse | null) => void
  setStatus: (status: EngineStatus) => void
  setRecordings: (recordings: Recording[]) => void
  pushToast: (kind: Toast['kind'], message: string) => void
  dismissToast: (id: number) => void
  setDetection: (frame: (DetectionFrame & { t: number }) | null) => void
  setSelected: (labels: string[]) => void
  setAttenuation: (db: number) => void
  setRuns: (recordingId: number, runs: Run[]) => void
  /** Insert or update one run (from a `run` WebSocket message or a request). */
  upsertRun: (run: Run) => void
}

export const initialAppState = {
  backend: 'unknown' as BackendState,
  backendVersion: null,
  ws: 'disconnected' as SocketState,
  activeTab: 'semantic' as Mode,
  config: null,
  status: null,
  recordings: [] as Recording[],
  toasts: [] as Toast[],
  detection: null as (DetectionFrame & { t: number }) | null,
  selected: [] as string[],
  attenuation: DEFAULT_ATTENUATION_DB,
  attenuationTouched: false,
  runs: {} as Record<number, Run[]>,
  measuredLatency: null as MeasuredLatency | null,
}

const MAX_TOASTS = 4
let nextToastId = 1

export const useAppStore = create<AppState>()((set) => ({
  ...initialAppState,
  setBackend: (backend, version = null) => set({ backend, backendVersion: version }),
  setWsState: (ws) => set({ ws }),
  setActiveTab: (activeTab) => set({ activeTab }),
  setConfig: (config) =>
    set((state) => {
      const db = config?.default_attenuation_db
      // the slider starts at the backend's live default, unless the user already moved it
      return typeof db === 'number' && !state.attenuationTouched
        ? { config, attenuation: db }
        : { config }
    }),
  setStatus: (status) => set({ status }),
  setRecordings: (recordings) => set({ recordings }),
  pushToast: (kind, message) =>
    set((state) => {
      // The same warning repeating (e.g. every 2 s) should not stack up.
      if (state.toasts.some((t) => t.kind === kind && t.message === message)) return state
      const toast = { id: nextToastId++, kind, message }
      return { toasts: [...state.toasts, toast].slice(-MAX_TOASTS) }
    }),
  dismissToast: (id) => set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
  setDetection: (detection) => set({ detection }),
  setSelected: (selected) => set({ selected }),
  setAttenuation: (attenuation) => set({ attenuation, attenuationTouched: true }),
  setRuns: (recordingId, runs) => set((state) => ({ runs: { ...state.runs, [recordingId]: runs } })),
  setMeasuredLatency: (measuredLatency) => set({ measuredLatency }),
  upsertRun: (run) =>
    set((state) => {
      if (run.recording_id === null) return state
      const current = state.runs[run.recording_id] ?? []
      const known = current.some((r) => r.run_id === run.run_id)
      const next = known
        ? current.map((r) => (r.run_id === run.run_id ? run : r))
        : [run, ...current] // newest first
      return { runs: { ...state.runs, [run.recording_id]: next } }
    }),
}))
