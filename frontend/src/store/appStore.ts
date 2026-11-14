// Global UI state (Zustand): connection state, active tab, backend config, latest status.
import { create } from 'zustand'
import type { SocketState } from '../api/ws.ts'
import type { ConfigResponse, EngineStatus, Mode } from '../api/types.ts'

export type BackendState = 'unknown' | 'up' | 'down'

export interface AppState {
  backend: BackendState
  backendVersion: string | null
  ws: SocketState
  activeTab: Mode
  config: ConfigResponse | null
  status: EngineStatus | null
  setBackend: (state: BackendState, version?: string | null) => void
  setWsState: (state: SocketState) => void
  setActiveTab: (tab: Mode) => void
  setConfig: (config: ConfigResponse | null) => void
  setStatus: (status: EngineStatus) => void
}

export const initialAppState = {
  backend: 'unknown' as BackendState,
  backendVersion: null,
  ws: 'disconnected' as SocketState,
  activeTab: 'semantic' as Mode,
  config: null,
  status: null,
}

export const useAppStore = create<AppState>()((set) => ({
  ...initialAppState,
  setBackend: (backend, version = null) => set({ backend, backendVersion: version }),
  setWsState: (ws) => set({ ws }),
  setActiveTab: (activeTab) => set({ activeTab }),
  setConfig: (config) => set({ config }),
  setStatus: (status) => set({ status }),
}))
