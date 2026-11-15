// Tab 1: the live session. Session bar, listening, recording, the class chips (live detection
// + target selection), the detection timeline, the three live visuals and the recordings list.
import { ClassChips } from '../components/detection/ClassChips.tsx'
import { DetectionTimeline } from '../components/detection/DetectionTimeline.tsx'
import { MonitorControls } from '../components/session/MonitorControls.tsx'
import { RecordControls } from '../components/session/RecordControls.tsx'
import { RecordingsList } from '../components/session/RecordingsList.tsx'
import { SessionBar } from '../components/session/SessionBar.tsx'
import { LevelMeters } from '../components/viz/LevelMeters.tsx'
import { Spectrogram } from '../components/viz/Spectrogram.tsx'
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
        <RecordControls />
      </div>

      <ClassChips />

      <figure className="space-y-1">
        <DetectionTimeline />
        <figcaption className="text-xs text-slate-500">
          Detected sounds (last 30 s; brighter = more certain)
        </figcaption>
      </figure>

      <div className="grid gap-4 md:grid-cols-[8rem_1fr]">
        <figure className="space-y-1">
          <LevelMeters />
          <figcaption className="text-center text-xs text-slate-500">Input level (dBFS)</figcaption>
        </figure>
        <div className="space-y-3">
          <figure className="space-y-1">
            <Spectrogram />
            <figcaption className="text-xs text-slate-500">Spectrogram (mel scale)</figcaption>
          </figure>
          <figure className="space-y-1">
            <Waveform />
            <figcaption className="text-xs text-slate-500">Waveform (mono mix)</figcaption>
          </figure>
        </div>
      </div>

      <RecordingsList />
    </section>
  )
}
