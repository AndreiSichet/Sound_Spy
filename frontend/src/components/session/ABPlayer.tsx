// A/B player for a finished processing run: the original recording and the processed result
// behind ONE play position. Two <audio> elements are kept in step: switching copies the current
// time to the other one and carries on playing there, so you hear the same moment with and
// without processing. The processed file is time-aligned with the original and equally long
// (the backend compensates the extractor's delay), which is what makes this comparison fair.
import { useRef, useState } from 'react'

export type ABSide = 'original' | 'processed'

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return '0:00'
  const whole = Math.max(0, Math.floor(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`
}

export function ABPlayer({
  id,
  originalUrl,
  processedUrl,
  duration,
  downloadName,
}: {
  id: string
  /** null when the original recording's file is gone: only the processed side can play. */
  originalUrl: string | null
  processedUrl: string
  duration: number
  downloadName: string
}) {
  const original = useRef<HTMLAudioElement | null>(null)
  const processed = useRef<HTMLAudioElement | null>(null)
  const [side, setSide] = useState<ABSide>('processed')
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)

  const element = (which: ABSide) => (which === 'original' ? original.current : processed.current)

  const play = (audio: HTMLAudioElement | null) => {
    // play() returns a promise that rejects if the browser refuses; never leave it unhandled
    void audio?.play()?.catch(() => setPlaying(false))
  }

  const toggle = () => {
    const audio = element(side)
    if (!audio) return
    if (playing) {
      audio.pause()
      setPlaying(false)
    } else {
      play(audio)
      setPlaying(true)
    }
  }

  const switchTo = (next: ABSide) => {
    if (next === side) return
    const from = element(side)
    const to = element(next)
    if (!from || !to) return
    to.currentTime = from.currentTime // same moment on the other side
    from.pause()
    if (playing) play(to)
    setSide(next)
  }

  const seek = (seconds: number) => {
    const audio = element(side)
    if (audio) audio.currentTime = seconds
    setTime(seconds)
  }

  const handlers = (which: ABSide) => ({
    onTimeUpdate: (event: React.SyntheticEvent<HTMLAudioElement>) => {
      if (which === side) setTime(event.currentTarget.currentTime)
    },
    onEnded: () => {
      if (which === side) setPlaying(false)
    },
  })

  return (
    <div className="flex w-full flex-wrap items-center gap-3" data-testid={`ab-player-${id}`}>
      {originalUrl && (
        <audio
          ref={original}
          src={originalUrl}
          preload="metadata"
          data-testid={`ab-original-${id}`}
          {...handlers('original')}
        />
      )}
      <audio
        ref={processed}
        src={processedUrl}
        preload="metadata"
        data-testid={`ab-processed-${id}`}
        {...handlers('processed')}
      />
      <button
        type="button"
        onClick={toggle}
        className="w-16 rounded border border-slate-700 px-2 py-0.5"
        aria-label={playing ? 'Pause' : 'Play'}
      >
        {playing ? 'Pause' : 'Play'}
      </button>
      <div role="group" aria-label="Version" className="flex overflow-hidden rounded border border-slate-700">
        {(['original', 'processed'] as const).map((which) => (
          <button
            key={which}
            type="button"
            onClick={() => switchTo(which)}
            aria-pressed={side === which}
            disabled={which === 'original' && !originalUrl}
            title={which === 'original' && !originalUrl ? 'The original file is gone' : undefined}
            className={`px-2 py-0.5 text-xs disabled:opacity-40 ${
              side === which ? 'bg-sky-700 text-white' : 'text-slate-300'
            }`}
          >
            {which === 'original' ? 'Original' : 'Processed'}
          </button>
        ))}
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(duration, 0.01)}
        step={0.01}
        value={Math.min(time, duration)}
        onChange={(e) => seek(Number(e.target.value))}
        aria-label="Position"
        className="min-w-32 flex-1"
      />
      <span className="tabular-nums text-xs text-slate-400">
        {formatTime(time)} / {formatTime(duration)}
      </span>
      <a href={processedUrl} download={downloadName} className="text-xs text-sky-300 underline">
        Download
      </a>
    </div>
  )
}
