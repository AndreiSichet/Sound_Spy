// Record what the session hears (the raw input, every channel) with an optional label and notes.
import { useState } from 'react'
import { ApiError, api } from '../../api/client.ts'
import { useAppStore } from '../../store/appStore.ts'

function formatSeconds(s: number): string {
  const m = Math.floor(s / 60)
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`
}

export function RecordControls() {
  const status = useAppStore((s) => s.status)
  const setStatus = useAppStore((s) => s.setStatus)
  const setRecordings = useAppStore((s) => s.setRecordings)
  const pushToast = useAppStore((s) => s.pushToast)
  const [label, setLabel] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const running = status?.state === 'running'
  const recording = status?.recording ?? false

  const toggle = async () => {
    setBusy(true)
    try {
      if (recording) {
        const saved = await api.recordStop()
        pushToast('info', `Saved ${saved.path.split('/').pop()}`)
        setRecordings(await api.recordings())
        setStatus(await api.session())
      } else {
        setStatus(await api.recordStart({ label: label.trim() || null, notes }))
      }
    } catch (error) {
      pushToast('error', error instanceof ApiError ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section aria-label="Record" className="flex flex-wrap items-center gap-3">
      <button
        type="button"
        onClick={() => void toggle()}
        disabled={!running || busy}
        className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-semibold disabled:opacity-40 ${
          recording ? 'bg-rose-700' : 'border border-slate-700'
        }`}
      >
        <span
          aria-hidden="true"
          className={`h-2.5 w-2.5 rounded-full ${recording ? 'animate-pulse bg-white' : 'bg-rose-500'}`}
        />
        {recording ? `Stop ${formatSeconds(status?.recording_seconds ?? 0)}` : 'Record'}
      </button>
      <input
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        placeholder="label (optional)"
        aria-label="Recording label"
        disabled={recording}
        maxLength={60}
        className="w-40 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm"
      />
      <input
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="notes (optional)"
        aria-label="Recording notes"
        disabled={recording}
        className="w-64 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm"
      />
    </section>
  )
}
