// Tab 1: the live session. Session bar, listening, recording, the three live visuals, the class
// chips (selection arrives with detection in step 05) and the recordings list.
import { MonitorControls } from '../components/session/MonitorControls.tsx'
import { RecordControls } from '../components/session/RecordControls.tsx'
import { RecordingsList } from '../components/session/RecordingsList.tsx'
import { SessionBar } from '../components/session/SessionBar.tsx'
import { LevelMeters } from '../components/viz/LevelMeters.tsx'
import { Spectrogram } from '../components/viz/Spectrogram.tsx'
import { Waveform } from '../components/viz/Waveform.tsx'
import { useAppStore } from '../store/appStore.ts'

export function SemanticTab() {
  const config = useAppStore((s) => s.config)

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

      <div className="space-y-2">
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
        <p className="text-xs text-slate-500">Sound selection arrives with detection (step 05).</p>
      </div>

      <RecordingsList />
    </section>
  )
}
