// Extraction controls next to Listen (step 06):
// - Attenuation: how much everything that is NOT selected is turned down (0-40 dB). It is part
//   of the target, so it is sent together with the selected classes (set_target).
// - Bypass (A/B): hear the unprocessed sound instead, with the same delay, to compare at once.
import { ApiError, api } from '../../api/client.ts'
import { MAX_ATTENUATION_DB, targetFor } from '../../api/target.ts'
import { sendTarget } from '../../hooks/useSessionSocket.ts'
import { useAppStore } from '../../store/appStore.ts'

export function ExtractionControls() {
  const status = useAppStore((s) => s.status)
  const setStatus = useAppStore((s) => s.setStatus)
  const selected = useAppStore((s) => s.selected)
  const attenuation = useAppStore((s) => s.attenuation)
  const setAttenuation = useAppStore((s) => s.setAttenuation)
  const pushToast = useAppStore((s) => s.pushToast)
  const bypass = status?.bypass ?? false

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
