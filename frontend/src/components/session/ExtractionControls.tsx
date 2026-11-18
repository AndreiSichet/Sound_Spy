// Extraction controls next to Listen:
// - Method (step 07): Neural (the trained network) or Baseline (the detector-gated filter of
//   step 06). It belongs to a session, so it can be changed only while stopped. A method that
//   cannot run (no trained model yet) is greyed out, with the reason as its tooltip.
// - Gate (step 07b, Neural only): a soft gate after the network turns its output further down
//   while it holds almost nothing of the input (the selected sound is not there). Like the
//   method it belongs to a session: switched while stopped.
// - Attenuation: how much everything that is NOT selected is turned down (0-40 dB). It is part
//   of the target, so it is sent together with the selected classes (set_target).
// - Bypass (A/B): hear the unprocessed sound instead, with the same delay, to compare at once.
import { ApiError, api } from '../../api/client.ts'
import { MAX_ATTENUATION_DB, isExtractionMethod, targetFor } from '../../api/target.ts'
import type { MethodInfo } from '../../api/types.ts'
import { sendTarget } from '../../hooks/useSessionSocket.ts'
import { useAppStore } from '../../store/appStore.ts'

const FALLBACK_METHODS: MethodInfo[] = [
  { name: 'neural_tse', display_name: 'Neural', available: true, reason: null },
  { name: 'mask_baseline', display_name: 'Baseline', available: true, reason: null },
]

export function ExtractionControls() {
  const status = useAppStore((s) => s.status)
  const setStatus = useAppStore((s) => s.setStatus)
  const selected = useAppStore((s) => s.selected)
  const attenuation = useAppStore((s) => s.attenuation)
  const setAttenuation = useAppStore((s) => s.setAttenuation)
  const pushToast = useAppStore((s) => s.pushToast)
  const config = useAppStore((s) => s.config)
  const bypass = status?.bypass ?? false
  const running = status?.state === 'running'
  const methods = config?.extraction_methods?.length ? config.extraction_methods : FALLBACK_METHODS
  // what sessions are started with; an unavailable choice shows what would really run
  const chosen = status?.extraction_method ?? config?.default_method ?? 'neural_tse'
  const usable = methods.find((m) => m.name === chosen)?.available !== false
  const method = usable ? chosen : (config?.default_method ?? chosen)
  const gate = status?.extraction_gate ?? config?.default_gate ?? true
  const gateUsable = method === 'neural_tse'

  const changeMethod = async (value: string) => {
    if (!isExtractionMethod(value)) return
    try {
      setStatus(await api.setMethod(value))
    } catch (error) {
      pushToast('error', error instanceof ApiError ? error.message : String(error))
    }
  }

  const toggleGate = async () => {
    try {
      setStatus(await api.setGate(!gate))
    } catch (error) {
      pushToast('error', error instanceof ApiError ? error.message : String(error))
    }
  }

  const changeAttenuation = (db: number) => {
    setAttenuation(db)
    const target = targetFor(selected, db)
    if (!sendTarget(target)) void api.setTarget(target).catch(() => undefined)
  }

  const toggleBypass = async () => {
    try {
      setStatus(await api.setBypass(!bypass))
    } catch (error) {
      pushToast('error', error instanceof ApiError ? error.message : String(error))
    }
  }

  return (
    <section aria-label="Extraction" className="flex flex-wrap items-center gap-4">
      <label
        className="flex items-center gap-2 text-sm text-slate-300"
        title={
          running
            ? 'Stop the session to change the method'
            : 'How the selected sounds are extracted'
        }
      >
        Method
        <select
          value={method}
          onChange={(e) => void changeMethod(e.target.value)}
          disabled={running}
          aria-label="Extraction method"
          className="rounded border border-slate-700 bg-slate-900 px-2 py-1 disabled:opacity-60"
        >
          {methods.map((m) => (
            <option key={m.name} value={m.name} disabled={!m.available} title={m.reason ?? ''}>
              {m.display_name}
              {m.available ? '' : ' (not available)'}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        onClick={() => void toggleGate()}
        disabled={running || !gateUsable}
        aria-pressed={gate && gateUsable}
        aria-label="Output gate"
        title={
          !gateUsable
            ? 'The gate belongs to the Neural method'
            : running
              ? 'Stop the session to switch the gate'
              : 'Neural: turn the output further down while the selected sound is not there ' +
                '(switch off to hear the network alone)'
        }
        className={`rounded border px-3 py-1 text-sm disabled:opacity-60 ${
          gate && gateUsable ? 'border-sky-500 bg-sky-900/40 text-sky-100' : 'border-slate-700'
        }`}
      >
        {gate ? 'Gate on' : 'Gate off'}
      </button>
      <label
        className="flex items-center gap-2 text-sm text-slate-300"
        title="How much everything that is not selected is turned down"
      >
        Attenuation
        <input
          type="range"
          min={0}
          max={MAX_ATTENUATION_DB}
          step={1}
          value={attenuation}
          onChange={(e) => changeAttenuation(Number(e.target.value))}
          aria-label="Attenuation (dB)"
        />
        <span className="w-12 tabular-nums">{attenuation} dB</span>
      </label>
      <button
        type="button"
        onClick={() => void toggleBypass()}
        aria-pressed={bypass}
        title="Hear the unprocessed sound (same delay) to compare"
        className={`rounded border px-3 py-1 text-sm ${
          bypass ? 'border-amber-400 bg-amber-900/40 text-amber-100' : 'border-slate-700'
        }`}
      >
        {bypass ? 'Bypass on (unprocessed)' : 'Bypass (A/B)'}
      </button>
    </section>
  )
}
