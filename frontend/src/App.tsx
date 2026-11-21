import { useEffect, type ComponentType } from 'react'
import { api } from './api/client.ts'
import type { Mode } from './api/types.ts'
import { ConnectionIndicator } from './components/ConnectionIndicator.tsx'
import { Toasts } from './components/Toasts.tsx'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts.ts'
import { usePersistedSettings } from './hooks/usePersistedSettings.ts'
import { useSessionSocket } from './hooks/useSessionSocket.ts'
import { useAppStore } from './store/appStore.ts'
import { SemanticTab } from './tabs/SemanticTab.tsx'
import { SourcesTab } from './tabs/SourcesTab.tsx'
import { SpatialTab } from './tabs/SpatialTab.tsx'

const TABS: { id: Mode; label: string; Component: ComponentType }[] = [
  { id: 'semantic', label: 'Semantic', Component: SemanticTab },
  { id: 'spatial', label: 'Spatial', Component: SpatialTab },
  { id: 'sources', label: 'Sources', Component: SourcesTab },
]

export default function App() {
  const activeTab = useAppStore((s) => s.activeTab)
  const setActiveTab = useAppStore((s) => s.setActiveTab)
  const backend = useAppStore((s) => s.backend)
  const hasConfig = useAppStore((s) => s.config !== null)
  const setConfig = useAppStore((s) => s.setConfig)
  const setMeasuredLatency = useAppStore((s) => s.setMeasuredLatency)

  useSessionSocket()
  usePersistedSettings()
  useKeyboardShortcuts(activeTab === 'semantic')

  // The latest acoustic latency measurement (step 08), shown next to the session's estimate.
  useEffect(() => {
    if (backend !== 'up') return
    api
      .latency()
      .then(setMeasuredLatency)
      .catch(() => undefined)
  }, [backend, setMeasuredLatency])

  // Load the app configuration once the backend is reachable.
  useEffect(() => {
    if (backend !== 'up' || hasConfig) return
    let cancelled = false
    api
      .config()
      .then((config) => {
        if (!cancelled) setConfig(config)
      })
      .catch((error: unknown) => console.warn('[api] failed to load config', error))
    return () => {
      cancelled = true
    }
  }, [backend, hasConfig, setConfig])

  const active = TABS.find((tab) => tab.id === activeTab) ?? TABS[0]!

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4">
      <header className="flex items-center justify-between border-b border-slate-800 py-4">
        <h1 className="text-xl font-bold tracking-tight">Sound Spy</h1>
        <ConnectionIndicator />
      </header>

      <nav role="tablist" aria-label="Modes" className="flex gap-1 pt-4">
        {TABS.map((tab) => {
          const selected = tab.id === active.id
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`tab-${tab.id}`}
              aria-selected={selected}
              aria-controls={`panel-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-t-md px-4 py-2 text-sm font-medium ${
                selected
                  ? 'bg-slate-800 text-white'
                  : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          )
        })}
      </nav>

      {backend === 'down' && (
        <div
          role="alert"
          className="mt-3 rounded-md border border-rose-700 bg-rose-950/60 p-3 text-sm text-rose-100"
        >
          The backend is not reachable. Start the app with <code>start.ps1</code> (or{' '}
          <code>soundspy serve</code> in development); this page reconnects by itself.
        </div>
      )}
      {backend === 'up' && !hasConfig && (
        <p className="mt-3 text-sm text-slate-400" role="status">
          Loading the configuration…
        </p>
      )}

      <main
        role="tabpanel"
        id={`panel-${active.id}`}
        aria-labelledby={`tab-${active.id}`}
        className="flex-1 rounded-b-md rounded-tr-md bg-slate-900/60 p-6"
      >
        <active.Component />
      </main>
      <Toasts />
    </div>
  )
}
