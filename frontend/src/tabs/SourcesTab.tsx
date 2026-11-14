// Tab 3: pick one detected source on a spatial map. Placeholder until module 5.
import { useAppStore } from '../store/appStore.ts'

export function SourcesTab() {
  const available = useAppStore((s) => s.config?.modes.find((m) => m.mode === 'sources')?.available)

  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold">Follow a source</h2>
      <p className="text-sm text-slate-400">
        {available === false
          ? 'Requires a microphone array (module 3). Source tracking comes in module 5.'
          : 'Coming in module 5.'}
      </p>
    </section>
  )
}
