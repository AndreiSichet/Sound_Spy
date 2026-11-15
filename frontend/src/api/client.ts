// Small fetch wrapper for the backend REST API (/api/*, proxied by Vite in dev).
import type {
  ConfigResponse,
  DetectionTimeline,
  EngineStatus,
  HealthResponse,
  MonitorRequest,
  ProcessRequest,
  RecordStartRequest,
  Recording,
  Run,
  SessionStartRequest,
  TargetSpec,
} from './types.ts'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { Accept: 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = 'invalid request'
    } catch {
      // body was not JSON; keep statusText
    }
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  return apiFetch<T>(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
}

export function recordingAudioUrl(id: number): string {
  return `/api/recordings/${id}/audio`
}

export function runAudioUrl(runId: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/audio`
}

export const api = {
  health: () => apiFetch<HealthResponse>('/health'),
  config: () => apiFetch<ConfigResponse>('/config'),
  session: () => apiFetch<EngineStatus>('/session'),
  startSession: (body: SessionStartRequest) => send<EngineStatus>('POST', '/session/start', body),
  stopSession: () => send<EngineStatus>('POST', '/session/stop'),
  setTarget: (target: TargetSpec) => send<TargetSpec>('PUT', '/session/target', target),
  setMonitor: (body: MonitorRequest) => send<EngineStatus>('PUT', '/session/monitor', body),
  recordStart: (body: RecordStartRequest) =>
    send<EngineStatus>('POST', '/session/record/start', body),
  recordStop: () => send<Recording>('POST', '/session/record/stop'),
  recordings: () => apiFetch<Recording[]>('/recordings'),
  detectRecording: (id: number) => send<DetectionTimeline>('POST', `/recordings/${id}/detect`),
  setBypass: (enabled: boolean) => send<EngineStatus>('PUT', '/session/bypass', { enabled }),
  processRecording: (id: number, body: ProcessRequest) =>
    send<Run>('POST', `/recordings/${id}/process`, body),
  recordingRuns: (id: number) => apiFetch<Run[]>(`/recordings/${id}/runs`),
  run: (runId: string) => apiFetch<Run>(`/runs/${encodeURIComponent(runId)}`),
}
