// Tab 1: the live session. Session bar, listening + extraction controls (method, attenuation,
// bypass), recording, the class chips (live detection + target selection), the detection
// timeline, the live visuals (input and output level meters side by side; the spectrogram can
// show input, output or both) and the recordings list with processing runs.
import { ClassChips } from '../components/detection/ClassChips.tsx'
import { DetectionTimeline } from '../components/detection/DetectionTimeline.tsx'
import { ExtractionControls } from '../components/session/ExtractionControls.tsx'
import { MonitorControls } from '../components/session/MonitorControls.tsx'
import { RecordControls } from '../components/session/RecordControls.tsx'
import { RecordingsList } from '../components/session/RecordingsList.tsx'
import { SessionBar } from '../components/session/SessionBar.tsx'
import { LevelMeters } from '../components/viz/LevelMeters.tsx'
import { SpectrogramPanel } from '../components/viz/SpectrogramPanel.tsx'
import { Waveform } from '../components/viz/Waveform.tsx'

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

      <div className="flex flex-col gap-3 rounded-md border border-slate-800 p-4">
        <MonitorControls />
        <ExtractionControls />
        <RecordControls />
      </div>

      <ClassChips />

      <figure className="space-y-1">
        <DetectionTimeline />
        <figcaption className="text-xs text-slate-500">
          Detected sounds (last 30 s; brighter = more certain)
        </figcaption>
      </figure>

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
            <Waveform />
            <figcaption className="text-xs text-slate-500">Waveform (input, mono mix)</figcaption>
          </figure>
        </div>
      </div>

      <RecordingsList />
    </section>
  )
}
