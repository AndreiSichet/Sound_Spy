// Saved recordings (GET /api/recordings), newest first. Each can be played in the browser,
// analysed (Detect: the offline detection timeline, step 05) and processed (Process: offline
// extraction of the currently selected classes, step 06) with the method chosen in "Process
// with" (Neural or Baseline, step 07; it starts on the session's method). Processing runs are
// listed under their recording with their status and method; a finished run gets an A/B player
// (original vs processed), so the two methods can be compared on the same recording.
// A recording whose file is gone is still listed (greyed, with its runs) but cannot be played.
// The browser only ever plays SAVED files; live sound goes to the speakers from the backend.
import { useEffect, useState } from 'react'
import { ApiError, api, recordingAudioUrl, runAudioUrl } from '../../api/client.ts'
import { hasGate, isExtractionMethod, methodName } from '../../api/target.ts'
import type { DetectionTimeline, ExtractionMethod, Recording, Run } from '../../api/types.ts'
import { useAppStore } from '../../store/appStore.ts'
import { OfflineTimeline } from '../detection/OfflineTimeline.tsx'
import { ABPlayer } from './ABPlayer.tsx'

const POLL_MS = 1000

function isRun(value: unknown): value is Run {
  return typeof value === 'object' && value !== null && typeof (value as Run).run_id === 'string'
}

function RunRow({ run, recording }: { run: Run; recording: Recording }) {
  const config = useAppStore((s) => s.config)
  const names = run.labels.map(
    (label) => config?.classes.find((c) => c.name === label)?.display_name ?? label,
  )
  const colour =
    run.status === 'done'
      ? 'text-emerald-300'
      : run.status === 'failed'
        ? 'text-rose-300'
        : 'text-amber-300'
  return (
    <li className="space-y-1 rounded bg-slate-900/60 px-2 py-1" data-testid={`run-${run.run_id}`}>
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <span className={`font-semibold ${colour}`} data-testid={`run-status-${run.run_id}`}>
          {run.status === 'pending'
            ? 'waiting…'
            : run.status === 'running'
              ? 'processing…'
              : run.status}
        </span>
        <span className="text-slate-200">{names.join(', ')}</span>
        <span className="text-slate-400">rest −{run.attenuation_db} dB</span>
        <span className="rounded bg-slate-800 px-1.5 text-slate-300" title={run.method}>
          {methodName(run.method)}
          {hasGate(run.method) && typeof run.info?.gate === 'boolean'
            ? run.info.gate
              ? ' · gate'
              : ' · no gate'
            : ''}
        </span>
        {run.processing_seconds != null && (
          <span className="text-slate-500">{run.processing_seconds.toFixed(1)} s to process</span>
        )}
        <span className="text-slate-500">{new Date(run.created_at).toLocaleTimeString()}</span>
      </div>
      {run.status === 'failed' && run.error && (
        <p className="text-xs text-rose-300">{run.error}</p>
      )}
      {run.status === 'done' && (
        <ABPlayer
          id={run.run_id}
          originalUrl={recording.missing ? null : recordingAudioUrl(recording.id)}
          processedUrl={runAudioUrl(run.run_id)}
          duration={run.duration_s ?? recording.duration_s}
          downloadName={`${run.run_id}.wav`}
        />
      )}
    </li>
  )
}

export function RecordingsList() {
  const recordings = useAppStore((s) => s.recordings)
  const setRecordings = useAppStore((s) => s.setRecordings)
  const backend = useAppStore((s) => s.backend)
  const config = useAppStore((s) => s.config)
  const selected = useAppStore((s) => s.selected)
  const attenuation = useAppStore((s) => s.attenuation)
  const runs = useAppStore((s) => s.runs)
  const setRuns = useAppStore((s) => s.setRuns)
  const upsertRun = useAppStore((s) => s.upsertRun)
  const pushToast = useAppStore((s) => s.pushToast)
  const [playing, setPlaying] = useState<number | null>(null)
  const [timelines, setTimelines] = useState<Record<number, DetectionTimeline>>({})
  const [detecting, setDetecting] = useState<number | null>(null)
  /** null = follow the session's method (the selector next to the attenuation slider). */
  const [picked, setPicked] = useState<ExtractionMethod | null>(null)
  const sessionMethod = useAppStore((s) => s.status?.extraction_method)
  // the Gate switch next to the method selector applies to runs too
  const sessionGate = useAppStore((s) => s.status?.extraction_gate)
  const methods = config?.extraction_methods ?? []
  const available = (name: string | null | undefined) =>
    methods.length === 0 || methods.find((m) => m.name === name)?.available === true
  const followed = [
    sessionMethod,
    config?.default_method,
    'hybrid',
    'neural_tse',
    'mask_baseline',
  ].find(
    (name) => isExtractionMethod(name) && available(name),
  )
  // (a backend that lists no methods gets no `method`: it uses its own default)
  const processMethod: ExtractionMethod | undefined =
    picked ?? (methods.length > 0 && isExtractionMethod(followed) ? followed : undefined)

  const detect = async (id: number) => {
    if (timelines[id]) {
      setTimelines((prev) => {
        const next = { ...prev }
        delete next[id]
        return next
      })
      return
    }
    setDetecting(id)
    try {
      const timeline = await api.detectRecording(id)
      setTimelines((prev) => ({ ...prev, [id]: timeline }))
    } catch (error) {
      pushToast('error', `Detection failed: ${error instanceof ApiError ? error.message : error}`)
    } finally {
      setDetecting(null)
    }
  }

  /** Fallback when the live channel is down: ask until the run has finished. */
  const poll = (runId: string) => {
    const tick = async () => {
      try {
        const run = await api.run(runId)
        if (!isRun(run)) return
        upsertRun(run)
        if (run.status === 'pending' || run.status === 'running') setTimeout(tick, POLL_MS)
      } catch {
        // the backend went away; the list reloads when it is back
      }
    }
    setTimeout(tick, POLL_MS)
  }

  const process = async (id: number) => {
    try {
      const run = await api.processRecording(id, {
        labels: selected,
        attenuation_db: attenuation,
        ...(processMethod ? { method: processMethod } : {}),
        ...(hasGate(processMethod) && sessionGate != null ? { gate: sessionGate } : {}),
      })
      if (!isRun(run)) return
      upsertRun(run)
      // state changes normally arrive as `run` messages over the WebSocket
      if (useAppStore.getState().ws !== 'connected') poll(run.run_id)
    } catch (error) {
      pushToast('error', `Processing failed: ${error instanceof ApiError ? error.message : error}`)
    }
  }

  useEffect(() => {
    if (backend !== 'up') return
    let cancelled = false
    api
      .recordings()
      .then(async (list) => {
        const rows = Array.isArray(list) ? list : []
        if (cancelled) return
        setRecordings(rows)
        for (const row of rows) {
          try {
            const found = await api.recordingRuns(row.id)
            if (!cancelled && Array.isArray(found)) setRuns(row.id, found.filter(isRun))
          } catch {
            // no runs to show for this one
          }
        }
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [backend, setRecordings, setRuns])

  return (
    <section aria-label="Recordings" className="space-y-2">
      <div className="flex flex-wrap items-center gap-4">
        <h3 className="text-sm font-semibold text-slate-300">Recordings</h3>
        {methods.length > 0 && (
          <label className="flex items-center gap-2 text-xs text-slate-400">
            Process with
            <select
              value={processMethod ?? ''}
              onChange={(e) => setPicked(isExtractionMethod(e.target.value) ? e.target.value : null)}
              aria-label="Processing method"
              className="rounded border border-slate-700 bg-slate-900 px-2 py-0.5"
            >
              {methods.map((m) => (
                <option key={m.name} value={m.name} disabled={!m.available} title={m.reason ?? ''}>
                  {m.display_name}
                  {m.available ? '' : ' (not available)'}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {recordings.length === 0 ? (
        <p className="text-sm text-slate-500">No recordings yet.</p>
      ) : (
        <ul className="divide-y divide-slate-800 rounded-md border border-slate-800">
          {recordings.map((r) => {
            const name = r.path.split('/').pop()
            const missing = r.missing === true
            const list = runs[r.id] ?? []
            return (
              <li
                key={r.id}
                className={`flex flex-wrap items-center gap-3 px-3 py-2 text-sm ${
                  missing ? 'opacity-60' : ''
                }`}
                data-missing={missing}
              >
                <button
                  type="button"
                  onClick={() => setPlaying(playing === r.id ? null : r.id)}
                  disabled={missing}
                  className="rounded border border-slate-700 px-2 py-0.5 disabled:opacity-50"
                  aria-label={`Play ${name}`}
                >
                  {playing === r.id ? 'Close' : 'Play'}
                </button>
                <button
                  type="button"
                  onClick={() => void detect(r.id)}
                  disabled={detecting === r.id || missing}
                  className="rounded border border-slate-700 px-2 py-0.5 disabled:opacity-50"
                  aria-label={`Detect sounds in ${name}`}
                >
                  {detecting === r.id ? 'Detecting…' : timelines[r.id] ? 'Hide' : 'Detect'}
                </button>
                <button
                  type="button"
                  onClick={() => void process(r.id)}
                  disabled={missing || selected.length === 0}
                  title={
                    selected.length === 0
                      ? 'Select one or more sounds above first'
                      : `Keep the selected sounds, turn the rest down by ${attenuation} dB (${methodName(
                          processMethod,
                        )})`
                  }
                  className="rounded border border-slate-700 px-2 py-0.5 disabled:opacity-50"
                  aria-label={`Process ${name}`}
                >
                  Process
                </button>
                <span className="font-mono text-xs text-slate-300">{name}</span>
                {r.label && (
                  <span className="rounded bg-slate-800 px-2 py-0.5 text-xs">{r.label}</span>
                )}
                <span className="tabular-nums text-slate-400">{r.duration_s.toFixed(1)} s</span>
                <span className="text-slate-500">{new Date(r.created_at).toLocaleString()}</span>
                {missing && <span className="text-xs text-amber-300">file missing</span>}
                {playing === r.id && !missing && (
                  <audio
                    controls
                    autoPlay
                    src={recordingAudioUrl(r.id)}
                    className="w-full"
                    data-testid={`audio-${r.id}`}
                  />
                )}
                {timelines[r.id] && config && (
                  <OfflineTimeline timeline={timelines[r.id]!} classes={config.classes} />
                )}
                {list.length > 0 && (
                  <ul className="w-full space-y-1" aria-label={`Processing runs of ${name}`}>
                    {list.map((run) => (
                      <RunRow key={run.run_id} run={run} recording={r} />
                    ))}
                  </ul>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
