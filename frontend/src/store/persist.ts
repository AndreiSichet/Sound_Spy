// Settings remembered between visits (step 08): the selected sounds, the attenuation, the method,
// the gate and the spectrogram view, in localStorage. Best effort: storage can be unavailable
// (private window, blocked site data), so every access is wrapped; the page then simply starts
// with the backend's defaults.

export type SpectrogramView = 'input' | 'output' | 'both'

export interface SavedSettings {
  selected?: string[]
  attenuation?: number
  method?: string
  gate?: boolean
  spectrogram?: SpectrogramView
}

export const SETTINGS_KEY = 'soundspy.semantic.v1'

function valid(raw: unknown): SavedSettings {
  if (typeof raw !== 'object' || raw === null) return {}
  const r = raw as Record<string, unknown>
  const out: SavedSettings = {}
  if (Array.isArray(r.selected) && r.selected.every((s) => typeof s === 'string')) {
    out.selected = r.selected as string[]
  }
  if (typeof r.attenuation === 'number' && Number.isFinite(r.attenuation)) {
    out.attenuation = r.attenuation
  }
  if (typeof r.method === 'string') out.method = r.method
  if (typeof r.gate === 'boolean') out.gate = r.gate
  if (r.spectrogram === 'input' || r.spectrogram === 'output' || r.spectrogram === 'both') {
    out.spectrogram = r.spectrogram
  }
  return out
}

export function loadSettings(): SavedSettings {
  try {
    const text = window.localStorage.getItem(SETTINGS_KEY)
    return text ? valid(JSON.parse(text)) : {}
  } catch {
    return {}
  }
}

export function saveSettings(patch: SavedSettings): void {
  try {
    const next = { ...loadSettings(), ...patch }
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(next))
  } catch {
    // storage unavailable: the setting just is not remembered
  }
}
