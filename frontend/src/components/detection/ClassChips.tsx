// The sound-class chips: live probability (smoothed) per class, highlighted while the detector
// says the class is present, and multi-selectable. Selecting sends `set_target` over the
// WebSocket (REST fallback when it is not connected). Extraction acts on it from step 06.
import { api } from '../../api/client.ts'
import { targetFor } from '../../api/target.ts'
import { sendTarget } from '../../hooks/useSessionSocket.ts'
import { useAppStore } from '../../store/appStore.ts'

export function ClassChips() {
  const config = useAppStore((s) => s.config)
  const detection = useAppStore((s) => s.detection)
  const selected = useAppStore((s) => s.selected)
  const setSelected = useAppStore((s) => s.setSelected)
  const status = useAppStore((s) => s.status)
  const detectorOn = status?.state === 'running' && status.detector != null

  if (config === null) return <p className="text-sm text-slate-500">Waiting for the backend…</p>

  const toggle = (name: string) => {
    const next = selected.includes(name) ? selected.filter((n) => n !== name) : [...selected, name]
    // keep class order, not click order
    const ordered = config.classes.map((c) => c.name).filter((n) => next.includes(n))
    setSelected(ordered)
    const target = targetFor(ordered)
    if (!sendTarget(target)) void api.setTarget(target).catch(() => undefined)
  }

  return (
    <div className="space-y-2">
      <ul className="flex flex-wrap gap-2" aria-label="Sound classes">
        {config.classes.map((cls) => {
          const p = detectorOn ? (detection?.probabilities[cls.name] ?? 0) : 0
          const active = detectorOn && (detection?.active.includes(cls.name) ?? false)
          const isSelected = selected.includes(cls.name)
          return (
            <li key={cls.name}>
              <button
                type="button"
                aria-pressed={isSelected}
                title={cls.description}
                onClick={() => toggle(cls.name)}
                data-testid={`chip-${cls.name}`}
                data-active={active}
                className={`relative w-36 overflow-hidden rounded-full border px-3 py-1 text-left text-sm ${
                  isSelected ? 'border-sky-400 text-sky-100' : 'border-slate-700 text-slate-300'
                } ${active ? 'bg-emerald-900/60' : 'bg-slate-900'}`}
              >
                <span
                  aria-hidden
                  className={`absolute inset-y-0 left-0 ${active ? 'bg-emerald-500/40' : 'bg-slate-600/40'}`}
                  style={{ width: `${Math.round(p * 100)}%` }}
                  data-testid={`bar-${cls.name}`}
                />
                <span className="relative flex justify-between gap-2">
                  <span>{cls.display_name}</span>
                  <span className="tabular-nums text-xs text-slate-400">
                    {detectorOn ? `${Math.round(p * 100)}%` : ''}
                  </span>
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      <p className="text-xs text-slate-500">
        {selected.length
          ? `Target: ${selected.length} sound type(s). `
          : 'Click sound types to select them. '}
        Selected sounds will be enhanced from step 06.
        {status?.state === 'running' && !status.detector && ' Detection is off for this session.'}
      </p>
    </div>
  )
}
