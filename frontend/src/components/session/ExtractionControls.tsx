// Extraction controls next to Listen:
// - Method: Hybrid (step 07c: the speech enhancer for Speech alone, the Waveformer for other
//   selections, and a gate the detector closes), Neural (the Waveformer alone, step 07) or
//   Baseline (the detector-gated filter of step 06). It belongs to a session, so it can be
//   changed only while stopped. A method that cannot run (no model yet) is greyed out, with
//   the reason as its tooltip.
// - Gate: Hybrid: the gate that silences the pauses (the detector closes it, the extractor's
//   own output reopens it at once); Neural: the step-07b output gate (off by default). Each
//   method starts with its own default; like the method it belongs to a session.
// - Attenuation: how much everything that is NOT selected is turned down (0-40 dB). It is part
//   of the target, so it is sent together with the selected classes (set_target).
// - Bypass (A/B): hear the unprocessed sound instead, with the same delay, to compare at once.
// Step 08: the methods are named "Hybrid (recommended)", "Neural", "Baseline" with a one-line
// explanation under the selector; the chosen method and gate are remembered (localStorage).
import { ApiError, api } from '../../api/client.ts'
import {
  MAX_ATTENUATION_DB,
  METHOD_INFO,
  hasGate,
  isExtractionMethod,
  targetFor,
} from '../../api/target.ts'
import type { MethodInfo } from '../../api/types.ts'
import { sendTarget } from '../../hooks/useSessionSocket.ts'
import { saveSettings } from '../../store/persist.ts'
import { useAppStore } from '../../store/appStore.ts'

const FALLBACK_METHODS: MethodInfo[] = [
  { name: 'hybrid', display_name: 'Hybrid', available: true, reason: null },
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
  const running = status?.state === 'running' || status?.state === 'warming'
  const methods = config?.extraction_methods?.length ? config.extraction_methods : FALLBACK_METHODS
  // what sessions are started with; an unavailable choice shows what would really run
  const chosen = status?.extraction_method ?? config?.default_method ?? 'hybrid'
  const usable = methods.find((m) => m.name === chosen)?.available !== false
  const method = usable ? chosen : (config?.default_method ?? chosen)
  const gate =
    status?.extraction_gate ?? config?.default_gates?.[method] ?? config?.default_gate ?? true
  const gateUsable = hasGate(method)

  const changeMethod = async (value: string) => {
    if (!isExtractionMethod(value)) return
    try {
      const next = await api.setMethod(value)
      setStatus(next)
      saveSettings(
        next.extraction_gate == null
          ? { method: value }
          : { method: value, gate: next.extraction_gate },
      )
    } catch (error) {
      pushToast('error', error instanceof ApiError ? error.message : String(error))
    }
  }

  const toggleGate = async () => {
    try {
      setStatus(await api.setGate(!gate))
      saveSettings({ gate: !gate })
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

  const help = METHOD_INFO[method]?.help
  return (
    <section aria-label="Extraction" className="flex flex-wrap items-center gap-4">
      <div className="flex flex-col gap-1">
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
              {METHOD_INFO[m.name]?.label ?? m.display_name}
              {m.available ? '' : ' (not available)'}
            </option>
          ))}
        </select>
      </label>
      {help && (
        <p className="max-w-sm text-xs text-slate-500" data-testid="method-help">
          {help}
        </p>
      )}
      </div>
      <button
        type="button"
        onClick={() => void toggleGate()}
        disabled={running || !gateUsable}
        aria-pressed={gate && gateUsable}
        aria-label="Output gate"
        title={
          !gateUsable
            ? 'The gate belongs to the Hybrid and Neural methods'
            : running
              ? 'Stop the session to switch the gate'
              : method === 'hybrid'
                ? 'Hybrid: silence while the detector is sure the selected sound is not there; ' +
                  'opens again the moment the extractor hears it (switch off to compare)'
                : 'Neural: turn the output further down while the selected sound is not ' +
                  'there (switch off to hear the network alone)'
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
