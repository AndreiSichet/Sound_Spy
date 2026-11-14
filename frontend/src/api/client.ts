// Small fetch wrapper for the backend REST API (/api/*, proxied by Vite in dev).
import type { ConfigResponse, HealthResponse } from './types.ts'

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
    } catch {
      // body was not JSON; keep statusText
    }
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

export const api = {
  health: () => apiFetch<HealthResponse>('/health'),
  config: () => apiFetch<ConfigResponse>('/config'),
}
