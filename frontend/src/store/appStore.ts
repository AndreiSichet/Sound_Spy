// Global UI state (Zustand): connection, active tab, backend config, latest status, recordings,
// toasts. High-rate visual frames are NOT here: they live in components/viz/liveFrames.ts.
import { create } from 'zustand'
import type { SocketState } from '../api/ws.ts'
import type { ConfigResponse, EngineStatus, Mode, Recording } from '../api/types.ts'

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
  setBackend: (state: BackendState, version?: string | null) => void
  setWsState: (state: SocketState) => void
  setActiveTab: (tab: Mode) => void
  setConfig: (config: ConfigResponse | null) => void
  setStatus: (status: EngineStatus) => void
  setRecordings: (recordings: Recording[]) => void
  pushToast: (kind: Toast['kind'], message: string) => void
  dismissToast: (id: number) => void
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
}

const MAX_TOASTS = 4
let nextToastId = 1

export const useAppStore = create<AppState>()((set) => ({
  ...initialAppState,
  setBackend: (backend, version = null) => set({ backend, backendVersion: version }),
  setWsState: (ws) => set({ ws }),
  setActiveTab: (activeTab) => set({ activeTab }),
  setConfig: (config) => set({ config }),
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
}))
