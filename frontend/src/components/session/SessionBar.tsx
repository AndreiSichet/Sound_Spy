// Start/Stop, source choice (microphone or a saved recording, looped), and live session figures:
// state, input device / rate / channels, latency, real-time factor and the drop/overflow counters.
import { useState } from 'react'
import { ApiError, api } from '../../api/client.ts'
import type { SourceRequest } from '../../api/types.ts'
import { liveFrames } from '../viz/liveFrames.ts'
import { useAppStore } from '../../store/appStore.ts'

function errorText(error: unknown): string {
  return error instanceof ApiError ? error.message : String(error)
}

function Stat({ label, value, warn = false }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-slate-500">{label}</span>
      <span className={`text-sm tabular-nums ${warn ? 'text-amber-300' : 'text-slate-200'}`}>
        {value}
      </span>
    </div>
  )
}

export function SessionBar() {
  const status = useAppStore((s) => s.status)
  const recordings = useAppStore((s) => s.recordings)
  const backend = useAppStore((s) => s.backend)
  const pushToast = useAppStore((s) => s.pushToast)
  const setStatus = useAppStore((s) => s.setStatus)
  const [source, setSource] = useState('mic')
  const [busy, setBusy] = useState(false)
  const running = status?.state === 'running'

  const start = async () => {
    setBusy(true)
    const recording = recordings.find((r) => `file:${r.id}` === source)
    const request: SourceRequest = recording
      ? { type: 'file', path: recording.path, loop: true }
      : { type: 'mic' }
    try {
      liveFrames.reset()
      setStatus(await api.startSession({ mode: 'semantic', source: request }))
    } catch (error) {
      pushToast('error', `Could not start: ${errorText(error)}`)
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    setBusy(true)
    try {
      setStatus(await api.stopSession())
      setRecordingsSoon()
    } catch (error) {
      pushToast('error', `Could not stop: ${errorText(error)}`)
    } finally {
      setBusy(false)
    }
  }

  const setRecordingsSoon = () => {
    void api.recordings().then(useAppStore.getState().setRecordings).catch(() => undefined)
  }

  const rtf = status?.rtf_p95
  const problems =
    (status?.input_overflows ?? 0) + (status?.output_underruns ?? 0) + (status?.dropped_chunks ?? 0)
  return (
    <section aria-label="Session" className="space-y-3 rounded-md border border-slate-800 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void (running ? stop() : start())}
          disabled={busy || backend !== 'up'}
          className={`rounded-md px-4 py-2 text-sm font-semibold disabled:opacity-50 ${
            running ? 'bg-rose-600 hover:bg-rose-500' : 'bg-emerald-600 hover:bg-emerald-500'
          }`}
        >
          {running ? 'Stop' : 'Start'}
        </button>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          Source
          <select
            value={source}
            onChange={(e) => setSource(e.target.value)}
            disabled={running}
            aria-label="Source"
            className="rounded border border-slate-700 bg-slate-900 px-2 py-1"
          >
            <option value="mic">Microphone</option>
            {recordings.map((r) => (
              <option key={r.id} value={`file:${r.id}`}>
                File: {r.path.split('/').pop()}
              </option>
            ))}
          </select>
        </label>
        <span
          data-testid="session-state"
          className={`text-sm ${
            status?.state === 'running'
              ? 'text-emerald-300'
              : status?.state === 'error'
                ? 'text-rose-300'
                : 'text-slate-400'
          }`}
        >
          {status?.state ?? 'idle'}
          {status?.message && status.state !== 'running' ? ` – ${status.message}` : ''}
        </span>
      </div>
      {running && (
        <div className="flex flex-wrap gap-x-6 gap-y-2" data-testid="session-stats">
          <Stat label="Input" value={status?.device ?? '–'} />
          <Stat
            label="Format"
            value={`${status?.sample_rate ?? '–'} Hz · ${status?.channels ?? '–'} ch`}
          />
          <Stat label="Latency" value={status?.latency_ms != null ? `${status.latency_ms} ms` : '–'} />
          <Stat
            label="RTF p95"
            value={rtf != null ? rtf.toFixed(3) : '–'}
            warn={rtf != null && rtf > 0.8}
          />
          <Stat
            label="Overflow / underrun / dropped"
            value={`${status?.input_overflows ?? 0} / ${status?.output_underruns ?? 0} / ${
              status?.dropped_chunks ?? 0
            }`}
            warn={problems > 0}
          />
        </div>
      )}
    </section>
  )
}
