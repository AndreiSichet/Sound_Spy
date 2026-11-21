// Tab 1: the live session (step 08 layout). The session bar on top (Start/Stop, source, state,
// route / gate / delay), then three areas:
//   Listen   - Listen / volume / mute / output device, the method with a one-line explanation,
//              the gate, the attenuation, Bypass, recording, the keyboard shortcuts;
//   Sounds   - the class chips: live probability, selected state, what will happen;
//   Visuals  - input / output level meters, the spectrogram (input, output or both), the
//              detection timeline and the input waveform;
// and the recordings with their processing runs in a collapsible panel. Waiting, warming and
// error states are shown in words where the content would otherwise be empty.
import type { ReactNode } from 'react'
import { ClassChips } from '../components/detection/ClassChips.tsx'
import { DetectionTimeline } from '../components/detection/DetectionTimeline.tsx'
import { ExtractionControls } from '../components/session/ExtractionControls.tsx'
import { MonitorControls } from '../components/session/MonitorControls.tsx'
import { OutputDeviceControls } from '../components/session/OutputDeviceControls.tsx'
import { RecordControls } from '../components/session/RecordControls.tsx'
import { RecordingsList } from '../components/session/RecordingsList.tsx'
import { SessionBar } from '../components/session/SessionBar.tsx'
import { LevelMeters } from '../components/viz/LevelMeters.tsx'
import { SpectrogramPanel } from '../components/viz/SpectrogramPanel.tsx'
import { Waveform } from '../components/viz/Waveform.tsx'
import { SHORTCUTS } from '../hooks/useKeyboardShortcuts.ts'
import { useAppStore } from '../store/appStore.ts'

function Area({ title, id, children }: { title: string; id: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="space-y-3 rounded-md border border-slate-800 p-4">
      <h3 id={id} className="text-sm font-semibold uppercase tracking-wide text-slate-400">
        {title}
      </h3>
      {children}
    </section>
  )
}

function VisualsHint() {
  const state = useAppStore((s) => s.status?.state ?? 'idle')
  if (state === 'running') return null
  const text =
    state === 'warming'
      ? 'Preparing models… the pictures start moving as soon as the sound flows.'
      : state === 'error'
        ? 'The session stopped (see above). Press Restart to try again.'
        : 'Press Start to see the sound live.'
  return (
    <p className="text-xs text-slate-500" data-testid="visuals-hint">
      {text}
    </p>
  )
}

export function SemanticTab() {
  return (
    <section className="space-y-5">
      <header>
        <h2 className="text-lg font-semibold">Choose what to hear</h2>
        <p className="text-sm text-slate-400">
          Selected sound types are kept; everything else is attenuated.
        </p>
      </header>

      <SessionBar />

      <div className="grid gap-5 lg:grid-cols-2">
        <Area title="Listen" id="area-listen">
          <MonitorControls />
          <OutputDeviceControls />
          <ExtractionControls />
          <RecordControls />
          <p className="text-xs text-slate-500" data-testid="shortcuts">
            Keys:{' '}
            {SHORTCUTS.map((s, i) => (
              <span key={s.key}>
                {i > 0 && ' · '}
                <kbd className="rounded border border-slate-700 px-1">{s.key}</kbd> {s.action}
              </span>
            ))}
          </p>
        </Area>

        <Area title="Sounds" id="area-sounds">
          <ClassChips />
        </Area>
      </div>

      <Area title="Visuals" id="area-visuals">
        <VisualsHint />
        <div className="grid gap-4 md:grid-cols-[12rem_1fr]">
          <div className="grid grid-cols-[2fr_1fr] gap-2">
            <figure className="space-y-1">
              <LevelMeters />
              <figcaption className="text-center text-xs text-slate-500">Input (dBFS)</figcaption>
            </figure>
            <figure className="space-y-1">
              <LevelMeters source="output" />
              <figcaption className="text-center text-xs text-slate-500">Output</figcaption>
            </figure>
          </div>
          <div className="space-y-3">
            <SpectrogramPanel />
            <figure className="space-y-1">
              <DetectionTimeline />
              <figcaption className="text-xs text-slate-500">
                Detected sounds (last 30 s; brighter = more certain)
              </figcaption>
            </figure>
            <figure className="space-y-1">
              <Waveform />
              <figcaption className="text-xs text-slate-500">Waveform (input, mono mix)</figcaption>
            </figure>
          </div>
        </div>
      </Area>

      <details className="rounded-md border border-slate-800 p-4" data-testid="recordings-panel">
        <summary className="cursor-pointer text-sm font-semibold uppercase tracking-wide text-slate-400">
          Recordings and processing
        </summary>
        <div className="mt-3">
          <RecordingsList />
        </div>
      </details>
    </section>
  )
}
