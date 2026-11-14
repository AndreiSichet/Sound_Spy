// "Listen": play the processed sound on the laptop's output. Off by default.
// The first time it is switched on, the user must confirm they are using headphones: with laptop
// speakers the sound feeds back into the microphone and howls. The confirmation is remembered in
// localStorage (best effort: storage can be unavailable, so every access is guarded).
import { useState } from 'react'
import { ApiError, api } from '../../api/client.ts'
import { useAppStore } from '../../store/appStore.ts'
import { headphonesConfirmed, rememberHeadphones } from './headphones.ts'

export function MonitorControls() {
  const status = useAppStore((s) => s.status)
  const setStatus = useAppStore((s) => s.setStatus)
  const pushToast = useAppStore((s) => s.pushToast)
  const [asking, setAsking] = useState(false)
  const [usingHeadphones, setUsingHeadphones] = useState(false)
  const enabled = status?.monitoring ?? false
  const gain = status?.gain_db ?? 0
  const muted = status?.muted ?? false
  // A running session without an output device cannot be listened to; say why instead of
  // letting the click fail.
  const noOutput = status?.state === 'running' && !status.output_device
  const noOutputReason = noOutput
    ? status?.message || 'No output device could be opened for this session.'
    : ''

  const apply = async (next: { enabled?: boolean; gain_db?: number; muted?: boolean }) => {
    try {
      setStatus(
        await api.setMonitor({
          enabled: next.enabled ?? enabled,
          gain_db: next.gain_db ?? gain,
          muted: next.muted ?? muted,
        }),
      )
    } catch (error) {
      pushToast('error', error instanceof ApiError ? error.message : String(error))
    }
  }

  const toggle = () => {
    if (enabled) return void apply({ enabled: false })
    if (!headphonesConfirmed()) {
      setAsking(true)
      return
    }
    void apply({ enabled: true })
  }

  const confirm = () => {
    rememberHeadphones()
    setAsking(false)
    void apply({ enabled: true })
  }

  return (
    <section aria-label="Listen" className="flex flex-wrap items-center gap-4">
      <label className="flex items-center gap-2 text-sm" title={noOutputReason || undefined}>
        <input
          type="checkbox"
          checked={enabled}
          onChange={toggle}
          disabled={noOutput}
          aria-label="Listen"
        />
        Listen
      </label>
      {noOutput && (
        <span className="text-xs text-amber-300" data-testid="no-output">
          Listening unavailable – stop and start the session again (devices changed?)
        </span>
      )}
      <label className="flex items-center gap-2 text-sm text-slate-300">
        Volume
        <input
          type="range"
          min={-40}
          max={12}
          step={1}
          value={gain}
          onChange={(e) => void apply({ gain_db: Number(e.target.value) })}
          aria-label="Volume (dB)"
        />
        <span className="w-14 tabular-nums">{gain > 0 ? `+${gain}` : gain} dB</span>
      </label>
      <button
        type="button"
        onClick={() => void apply({ muted: !muted })}
        aria-pressed={muted}
        className="rounded border border-slate-700 px-3 py-1 text-sm"
      >
        {muted ? 'Unmute' : 'Mute'}
      </button>

      {asking && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="headphones-title"
          className="fixed inset-0 z-20 flex items-center justify-center bg-black/60"
        >
          <div className="max-w-sm space-y-4 rounded-lg border border-slate-700 bg-slate-900 p-5">
            <h3 id="headphones-title" className="font-semibold">
              Use headphones
            </h3>
            <p className="text-sm text-slate-300">
              Use headphones. With laptop speakers the sound feeds back into the microphone and
              howls.
            </p>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={usingHeadphones}
                onChange={(e) => setUsingHeadphones(e.target.checked)}
              />
              I&apos;m using headphones
            </label>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setAsking(false)}
                className="rounded px-3 py-1 text-sm"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirm}
                disabled={!usingHeadphones}
                className="rounded bg-emerald-600 px-3 py-1 text-sm font-semibold disabled:opacity-40"
              >
                Start listening
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
