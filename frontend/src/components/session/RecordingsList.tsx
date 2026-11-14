// Saved recordings (GET /api/recordings), newest first, each playable in the browser.
// The browser only ever plays SAVED files; live sound goes to the speakers from the backend.
import { useEffect, useState } from 'react'
import { api, recordingAudioUrl } from '../../api/client.ts'
import { useAppStore } from '../../store/appStore.ts'

export function RecordingsList() {
  const recordings = useAppStore((s) => s.recordings)
  const setRecordings = useAppStore((s) => s.setRecordings)
  const backend = useAppStore((s) => s.backend)
  const [playing, setPlaying] = useState<number | null>(null)

  useEffect(() => {
    if (backend !== 'up') return
    api
      .recordings()
      .then((list) => setRecordings(Array.isArray(list) ? list : []))
      .catch(() => undefined)
  }, [backend, setRecordings])

  return (
    <section aria-label="Recordings" className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-300">Recordings</h3>
      {recordings.length === 0 ? (
        <p className="text-sm text-slate-500">No recordings yet.</p>
      ) : (
        <ul className="divide-y divide-slate-800 rounded-md border border-slate-800">
          {recordings.map((r) => (
            <li key={r.id} className="flex flex-wrap items-center gap-3 px-3 py-2 text-sm">
              <button
                type="button"
                onClick={() => setPlaying(playing === r.id ? null : r.id)}
                className="rounded border border-slate-700 px-2 py-0.5"
                aria-label={`Play ${r.path.split('/').pop()}`}
              >
                {playing === r.id ? 'Close' : 'Play'}
              </button>
              <span className="font-mono text-xs text-slate-300">{r.path.split('/').pop()}</span>
              {r.label && (
                <span className="rounded bg-slate-800 px-2 py-0.5 text-xs">{r.label}</span>
              )}
              <span className="tabular-nums text-slate-400">{r.duration_s.toFixed(1)} s</span>
              <span className="text-slate-500">{new Date(r.created_at).toLocaleString()}</span>
              {playing === r.id && (
                <audio
                  controls
                  autoPlay
                  src={recordingAudioUrl(r.id)}
                  className="w-full"
                  data-testid={`audio-${r.id}`}
                />
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
