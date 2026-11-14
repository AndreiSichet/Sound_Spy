// One vertical meter per input channel: RMS bar, peak tick, peak hold (~1 s), red clip LED.
// Scale -60..+6 dBFS (the laptop mic can exceed 0 dBFS: Windows adds +7.4 dB after the converter).
import { useCallback } from 'react'
import { liveFrames } from './liveFrames.ts'
import { METER_MIN_DB, dbToFraction } from './scales.ts'
import { useCanvasLoop } from './useCanvasLoop.ts'

const TICKS = [6, 0, -6, -12, -20, -30, -40, -60]

function barColour(db: number): string {
  if (db >= 0) return '#f87171' // above full scale
  if (db >= -12) return '#fbbf24'
  return '#34d399'
}

export function LevelMeters() {
  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number, now: number) => {
    ctx.clearRect(0, 0, w, h)
    const levels = liveFrames.levels
    const channels = levels?.rms_dbfs.length ?? 2
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
      const hold = liveFrames.peakHold[ch]
      if (hold !== undefined && hold > METER_MIN_DB) {
        ctx.fillStyle = barColour(hold)
        ctx.fillRect(x, y(hold) - 1, meterW, 2)
      }
      // clip LED above the meter
      ctx.fillStyle = liveFrames.clipping(ch, now) ? '#ef4444' : '#3f1d1d'
      ctx.beginPath()
      ctx.arc(x + meterW / 2, 6, 4, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#94a3b8'
      ctx.textAlign = 'center'
      ctx.fillText(`ch${ch}`, x + meterW / 2, h - 7)
      ctx.textAlign = 'right'
    }
  }, [])
  const ref = useCanvasLoop(draw)
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="Input level meters"
      className="h-48 w-full"
      data-testid="level-meters"
    />
  )
}
