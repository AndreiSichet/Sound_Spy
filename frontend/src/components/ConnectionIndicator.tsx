// Backend health (polls /api/health) + WebSocket state, shown as a coloured dot and a label.
import { useEffect } from 'react'
import { api } from '../api/client.ts'
import { useAppStore } from '../store/appStore.ts'

export const HEALTH_POLL_MS = 5000

type Indicator = { label: string; dot: string; text: string }

export function ConnectionIndicator({ pollMs = HEALTH_POLL_MS }: { pollMs?: number }) {
  const backend = useAppStore((s) => s.backend)
  const version = useAppStore((s) => s.backendVersion)
  const ws = useAppStore((s) => s.ws)
  const setBackend = useAppStore((s) => s.setBackend)

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        const health = await api.health()
        if (!cancelled) setBackend('up', health.version)
      } catch {
        if (!cancelled) setBackend('down')
      }
    }
    void check()
    const id = setInterval(() => void check(), pollMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [pollMs, setBackend])

  const indicator = describe(backend, ws)
  return (
    <div
      role="status"
      aria-label="Backend connection"
      data-state={indicator.label}
      className="flex items-center gap-2 rounded-full border border-slate-800 px-3 py-1 text-sm"
      title={`backend: ${backend}, websocket: ${ws}`}
    >
      <span className={`h-2.5 w-2.5 rounded-full ${indicator.dot}`} aria-hidden="true" />
      <span className={indicator.text}>{indicator.label}</span>
      {backend === 'up' && version && <span className="text-slate-500">v{version}</span>}
    </div>
  )
}

function describe(backend: string, ws: string): Indicator {
  if (backend === 'up' && ws === 'connected') {
    return { label: 'connected', dot: 'bg-emerald-400', text: 'text-emerald-300' }
  }
  if (backend === 'down' && ws !== 'connected') {
    const label = ws === 'reconnecting' ? 'disconnected · reconnecting' : 'disconnected'
    return { label, dot: 'bg-red-500', text: 'text-red-300' }
  }
  if (backend === 'unknown' && ws !== 'connected') {
    return { label: 'connecting', dot: 'bg-slate-500', text: 'text-slate-300' }
  }
  return { label: 'reconnecting', dot: 'bg-amber-400', text: 'text-amber-300' }
}
