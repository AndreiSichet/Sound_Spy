// Tab 1: choose sound types to hear. Lists the classes from GET /api/config (controls in step 03).
import { useAppStore } from '../store/appStore.ts'

export function SemanticTab() {
  const config = useAppStore((s) => s.config)

  return (
    <section className="space-y-4">
      <header>
        <h2 className="text-lg font-semibold">Choose what to hear</h2>
        <p className="text-sm text-slate-400">
          Selected sound types are kept; everything else is attenuated.
        </p>
      </header>

      {config === null ? (
        <p className="text-sm text-slate-500">Waiting for the backend…</p>
      ) : (
        <ul className="flex flex-wrap gap-2" aria-label="Sound classes">
          {config.classes.map((cls) => (
            <li key={cls.name}>
              <button
                type="button"
                disabled
                aria-pressed={false}
                title={cls.description}
                className="cursor-not-allowed rounded-full border border-slate-700 px-3 py-1 text-sm text-slate-400"
              >
                {cls.display_name}
              </button>
            </li>
          ))}
        </ul>
      )}
      <p className="text-xs text-slate-500">Selection available in step 03.</p>
    </section>
  )
}
