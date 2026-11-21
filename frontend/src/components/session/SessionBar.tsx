// Start/Stop, source choice (microphone or a saved recording, looped), and live session figures:
// state, input device / rate / channels, latency, real-time factor, the drop/overflow counters,
// the detector, and the extractor: its method, where its model runs (GPU / CPU), how long one
// model call takes and the delay it adds (step 07).
// Step 08: "warming" (the models are being prepared; Stop already works), an error message with
// Restart, and one summary line while running: the route, the gate (in words, not only colour)
// and the delay, labelled "measured" (the latest `soundspy measure-latency` for this route) or
// "estimated" (the session's own sum of its parts).
import { useState } from 'react'
import { ApiError, api } from '../../api/client.ts'
import { ROUTE_NAMES, extractionSummary, latencyLabel, methodName } from '../../api/target.ts'
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

function GateIndicator({ state }: { state: string | null | undefined }) {
  const text = state === 'closed' ? 'Gate closed' : state === 'open' ? 'Gate open' : 'No gate'
  const dot =
    state === 'closed' ? 'bg-slate-400' : state === 'open' ? 'bg-emerald-400' : 'bg-slate-700'
  return (
    <span className="flex items-center gap-1.5" data-testid="gate-indicator">
      <span aria-hidden className={`inline-block h-2 w-2 rounded-full ${dot}`} />
      {text}
    </span>
  )
}

export function SessionBar() {
  const status = useAppStore((s) => s.status)
  const recordings = useAppStore((s) => s.recordings)
  const backend = useAppStore((s) => s.backend)
  const measured = useAppStore((s) => s.measuredLatency)
  const pushToast = useAppStore((s) => s.pushToast)
  const setStatus = useAppStore((s) => s.setStatus)
  const [source, setSource] = useState('mic')
  const [busy, setBusy] = useState(false)
  const state = status?.state ?? 'idle'
  const running = state === 'running'
  const active = running || state === 'warming'

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
  const latency = latencyLabel(status, measured)
  const extraction = status?.extraction
  const stateText =
    state === 'warming'
      ? `Preparing models…${status?.message && status.message !== 'preparing' ? ` ${status.message}` : ''}`
      : state
  return (
    <section aria-label="Session" className="space-y-3 rounded-md border border-slate-800 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void (active ? stop() : start())}
          disabled={busy || backend !== 'up'}
          className={`rounded-md px-4 py-2 text-sm font-semibold disabled:opacity-50 ${
            active ? 'bg-rose-600 hover:bg-rose-500' : 'bg-emerald-600 hover:bg-emerald-500'
          }`}
        >
          {active ? 'Stop' : 'Start'}
        </button>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          Source
          <select
            value={source}
            onChange={(e) => setSource(e.target.value)}
            disabled={active}
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
          role="status"
          aria-live="polite"
          className={`text-sm ${
            running
              ? 'text-emerald-300'
              : state === 'error'
                ? 'text-rose-300'
                : state === 'warming'
                  ? 'text-sky-300'
                  : 'text-slate-400'
          }`}
        >
          {stateText}
          {status?.message && state !== 'running' && state !== 'warming'
            ? ` – ${status.message}`
            : ''}
        </span>
        {state === 'error' && (
          <button
            type="button"
            onClick={() => void start()}
            disabled={busy || backend !== 'up'}
            className="rounded-md border border-rose-400 px-3 py-1 text-sm text-rose-100"
          >
            Restart
          </button>
        )}
      </div>
      {running && (
        <p
          className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-slate-200"
          data-testid="session-summary"
        >
          <span>
            {methodName(extraction?.method)}
            {extraction?.route ? ` · ${ROUTE_NAMES[extraction.route] ?? extraction.route}` : ''}
          </span>
          {extraction?.method === 'hybrid' && <GateIndicator state={extraction.gate_state} />}
          {latency && (
            <span title={latency.title} data-testid="latency-label">
              Delay {latency.text}
            </span>
          )}
          {status?.bypass && <span className="text-amber-300">Bypass: unprocessed</span>}
        </p>
      )}
      {running && (
        <details className="text-sm">
          <summary className="cursor-pointer text-xs text-slate-500">Details</summary>
          <div className="mt-2 flex flex-wrap gap-x-6 gap-y-2" data-testid="session-stats">
            <Stat label="Input" value={status?.device ?? '–'} />
            <Stat
              label="Format"
              value={`${status?.sample_rate ?? '–'} Hz · ${status?.channels ?? '–'} ch`}
            />
            <Stat
              label="Latency"
              value={status?.latency_ms != null ? `${status.latency_ms} ms` : '–'}
            />
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
            {status?.detector ? (
              <>
                <Stat
                  label="Detector"
                  value={`${status.detector.model} · ${status.detector.device}`}
                />
                <Stat
                  label="Inference"
                  value={
                    status.detector.latency_ms != null ? `${status.detector.latency_ms} ms` : '–'
                  }
                />
                <Stat
                  label="Skipped windows"
                  value={String(status.detector.skipped)}
                  warn={status.detector.skipped > 0}
                />
              </>
            ) : (
              <Stat label="Detector" value="off" />
            )}
            {extraction && (
              <Stat
                label="Extraction"
                value={extractionSummary(extraction, status?.bypass)}
                warn={extraction.method === 'mask_baseline' && !extraction.gating}
              />
            )}
            {(extraction?.method === 'neural_tse' || extraction?.method === 'hybrid') && (
              <Stat
                label="Model time"
                value={
                  extraction.model_ms != null
                    ? `${extraction.model_ms} ms${
                        extraction.model_ms_p95 != null ? ` (p95 ${extraction.model_ms_p95})` : ''
                      }`
                    : '–'
                }
              />
            )}
            {status?.warmup_s != null && (
              <Stat label="Start" value={`${status.warmup_s.toFixed(1)} s`} />
            )}
          </div>
        </details>
      )}
    </section>
  )
}
