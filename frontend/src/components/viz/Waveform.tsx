// Scrolling min/max envelope of the mono mix over the last WAVEFORM_SECONDS.
// Each WaveformFrame covers the interval since the previous frame; its min/max pairs are spread
// across that interval, newest at the right edge.
import { useCallback } from 'react'
import { WAVEFORM_SECONDS, liveFrames } from './liveFrames.ts'
import { useCanvasLoop } from './useCanvasLoop.ts'

export function Waveform() {
  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number) => {
    ctx.clearRect(0, 0, w, h)
    const mid = h / 2
    ctx.fillStyle = '#334155'
    ctx.fillRect(0, mid, w, 1)
    const frames = liveFrames.waveform
    const latest = frames[frames.length - 1]
    if (!latest) return
    const start = latest.t - WAVEFORM_SECONDS
    const x = (t: number) => ((t - start) / WAVEFORM_SECONDS) * w
    ctx.strokeStyle = '#38bdf8'
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i]
      if (!frame) continue
      const prevT = frames[i - 1]?.t ?? frame.t - 0.04
      const n = frame.mins.length
      for (let k = 0; k < n; k++) {
        const t = prevT + ((k + 0.5) / n) * (frame.t - prevT)
        const px = Math.round(x(t)) + 0.5
        const lo = Math.max(-1, Math.min(1, frame.mins[k] ?? 0))
        const hi = Math.max(-1, Math.min(1, frame.maxs[k] ?? 0))
        ctx.moveTo(px, mid - hi * mid)
        ctx.lineTo(px, mid - lo * mid + 1)
      }
    }
    ctx.stroke()
    ctx.fillStyle = '#64748b'
    ctx.font = '10px ui-sans-serif, system-ui, sans-serif'
    ctx.textBaseline = 'top'
    ctx.textAlign = 'left'
    ctx.fillText(`-${WAVEFORM_SECONDS} s`, 4, 4)
    ctx.textAlign = 'right'
    ctx.fillText('now', w - 4, 4)
  }, [])
  const ref = useCanvasLoop(draw)
  return (
    <canvas ref={ref} role="img" aria-label="Waveform" className="h-28 w-full" data-testid="waveform" />
  )
}
