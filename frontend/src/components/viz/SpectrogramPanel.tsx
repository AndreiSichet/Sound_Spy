// The spectrogram with an Input / Output / Both switch (step 06). "Input" is what the microphone
// hears, "Output" what is left after extraction; "Both" stacks them so the removed sound can be
// SEEN: it is in the upper picture and missing from the lower one.
// Step 08: the chosen view is remembered (localStorage).
import { useState } from 'react'
import { loadSettings, saveSettings, type SpectrogramView } from '../../store/persist.ts'
import { Spectrogram } from './Spectrogram.tsx'

export type { SpectrogramView }

const VIEWS: { view: SpectrogramView; label: string }[] = [
  { view: 'input', label: 'Input' },
  { view: 'output', label: 'Output' },
  { view: 'both', label: 'Both' },
]

export function SpectrogramPanel({ initial }: { initial?: SpectrogramView }) {
  const [view, setViewState] = useState<SpectrogramView>(
    () => initial ?? loadSettings().spectrogram ?? 'both',
  )
  const setView = (next: SpectrogramView) => {
    setViewState(next)
    saveSettings({ spectrogram: next })
  }
  const half = view === 'both'
  return (
    <figure className="space-y-1">
      {view !== 'output' && <Spectrogram source="input" className={half ? 'h-40' : 'h-64'} />}
      {view !== 'input' && <Spectrogram source="output" className={half ? 'h-40' : 'h-64'} />}
      <figcaption className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
        <span>
          Spectrogram (mel scale)
          {view === 'both' ? ': input above, output below' : view === 'output' ? ': output' : ': input'}
        </span>
        <span role="group" aria-label="Spectrogram view" className="flex overflow-hidden rounded border border-slate-700">
          {VIEWS.map(({ view: v, label }) => (
            <button
              key={v}
              type="button"
              onClick={() => setView(v)}
              aria-pressed={view === v}
              className={`px-2 py-0.5 ${view === v ? 'bg-sky-700 text-white' : 'text-slate-300'}`}
            >
              {label}
            </button>
          ))}
        </span>
      </figcaption>
    </figure>
  )
}
