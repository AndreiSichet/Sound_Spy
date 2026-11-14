// Tab 2: sound types + a direction (azimuth + beam width). Placeholder until module 3.
import { useAppStore } from '../store/appStore.ts'

export function SpatialTab() {
  const available = useAppStore((s) => s.config?.modes.find((m) => m.mode === 'spatial')?.available)

  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold">Listen in a direction</h2>
      <p className="text-sm text-slate-400">
        {available === false
          ? 'Requires a microphone array (module 3).'
          : 'Direction controls arrive in module 3.'}
      </p>
    </section>
  )
}
