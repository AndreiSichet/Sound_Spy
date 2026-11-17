// One vertical meter per channel: RMS bar, peak tick, peak hold (~1 s), red clip LED.
// Scale -60..+6 dBFS (the laptop mic can exceed 0 dBFS: Windows adds +7.4 dB after the converter).
// `source="input"` shows what the microphone hears (one meter per microphone); `source="output"`
// shows the sound after extraction (one channel), so the two can be compared at a glance: with a
// class selected the output meter drops whenever only other sounds are present (step 07).
import { useCallback } from 'react'
import type { FrameSource } from '../../api/types.ts'
import { liveFrames } from './liveFrames.ts'
import { METER_MIN_DB, dbToFraction } from './scales.ts'
import { useCanvasLoop } from './useCanvasLoop.ts'

const TICKS = [6, 0, -6, -12, -20, -30, -40, -60]

function barColour(db: number): string {
  if (db >= 0) return '#f87171' // above full scale
  if (db >= -12) return '#fbbf24'
  return '#34d399'
}

export function LevelMeters({ source = 'input' }: { source?: FrameSource }) {
  const output = source === 'output'
  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, now: number) => {
    ctx.clearRect(0, 0, w, h)
    const levels = liveFrames.levelsOf(source)
    const channels = levels?.rms_dbfs.length ?? (output ? 1 : 2)
    const top = 14
    const bottom = h - 18
    const scaleW = 28
    const gap = 8
    const meterW = Math.max(8, (w - scaleW - gap * channels) / channels)
    const y = (db: number) => bottom - dbToFraction(db) * (bottom - top)

    ctx.font = '10px ui-sans-serif, system-ui, sans-serif'
    ctx.fillStyle = '#64748b'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (const tick of TICKS) {
      ctx.fillText(tick > 0 ? `+${tick}` : `${tick}`, scaleW - 4, y(tick))
      ctx.fillRect(scaleW - 2, y(tick), 2, 1)
    }

    for (let ch = 0; ch < channels; ch++) {
      const x = scaleW + gap + ch * (meterW + gap)
      ctx.fillStyle = '#1e293b'
      ctx.fillRect(x, top, meterW, bottom - top)
      ctx.fillStyle = '#334155'
      ctx.fillRect(x, y(0), meterW, 1) // 0 dBFS line
      const rms = levels?.rms_dbfs[ch]
      if (rms !== undefined && rms > METER_MIN_DB) {
        ctx.fillStyle = barColour(rms)
        ctx.fillRect(x, y(rms), meterW, bottom - y(rms))
      }
      const peak = levels?.peak_dbfs[ch]
      if (peak !== undefined && peak > METER_MIN_DB) {
        ctx.fillStyle = '#e2e8f0'
        ctx.fillRect(x, y(peak), meterW, 2)
      }
      const hold = liveFrames.peakHoldOf(source)[ch]
      if (hold !== undefined && hold > METER_MIN_DB) {
        ctx.fillStyle = barColour(hold)
        ctx.fillRect(x, y(hold) - 1, meterW, 2)
      }
      // clip LED above the meter
      ctx.fillStyle = liveFrames.clipping(ch, now, source) ? '#ef4444' : '#3f1d1d'
      ctx.beginPath()
      ctx.arc(x + meterW / 2, 6, 4, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#94a3b8'
      ctx.textAlign = 'center'
      ctx.fillText(output ? 'out' : `ch${ch}`, x + meterW / 2, h - 7)
      ctx.textAlign = 'right'
    }
  }, [source, output])
  const ref = useCanvasLoop(draw)
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={output ? 'Output level meter' : 'Input level meters'}
      className="h-48 w-full"
      data-testid={output ? 'level-meters-output' : 'level-meters'}
    />
  )
}
