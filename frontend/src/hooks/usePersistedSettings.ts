// Restores the remembered Semantic-tab settings once the backend's configuration is known, and
// keeps remembering the selection and the attenuation as they change (step 08). The method and
// the gate are restored only while no session runs (they belong to a session); they are saved
// where the user changes them (ExtractionControls).
import { useEffect, useRef } from 'react'
import { api } from '../api/client.ts'
import { isExtractionMethod, targetFor } from '../api/target.ts'
import { loadSettings, saveSettings } from '../store/persist.ts'
import { useAppStore } from '../store/appStore.ts'

export function usePersistedSettings(): void {
  const config = useAppStore((s) => s.config)
  const restored = useRef(false)

  // restore once the classes are known
  useEffect(() => {
    if (config === null || restored.current) return
    restored.current = true
    const saved = loadSettings()
    const store = useAppStore.getState()
    const known = new Set(config.classes.map((c) => c.name))
    const selected = (saved.selected ?? []).filter((name) => known.has(name))
    if (selected.length) store.setSelected(selected)
    if (saved.attenuation != null) store.setAttenuation(saved.attenuation)
    if (selected.length || saved.attenuation != null) {
      const state = useAppStore.getState()
      void api.setTarget(targetFor(state.selected, state.attenuation)).catch(() => undefined)
    }
    const idle = store.status?.state !== 'running' && store.status?.state !== 'warming'
    const available = config.extraction_methods?.find((m) => m.name === saved.method)?.available
    if (idle && isExtractionMethod(saved.method) && available !== false) {
      const method = saved.method
      void api
        .setMethod(method)
        .then((status) => {
          useAppStore.getState().setStatus(status)
          if (saved.gate != null && status.extraction_gate !== saved.gate) {
            return api.setGate(saved.gate).then(useAppStore.getState().setStatus)
          }
          return undefined
        })
        .catch(() => undefined)
    }
  }, [config])

  // remember the selection and the attenuation
  useEffect(
    () =>
      useAppStore.subscribe((state, previous) => {
        if (!restored.current) return
        if (state.selected !== previous.selected) saveSettings({ selected: state.selected })
        if (state.attenuation !== previous.attenuation) {
          saveSettings({ attenuation: state.attenuation })
        }
      }),
    [],
  )
}
