// Live detection timeline: one lane per class over the last TIMELINE_SECONDS, newest at the
// right like the spectrogram (same left margin, so "now" lines up). A lane is filled where the
// class was active; the opacity follows its probability. Read from liveFrames, drawn in rAF.
import { useCallback, useMemo } from 'react'
import { useAppStore } from '../../store/appStore.ts'
import { TIMELINE_SECONDS, liveFrames } from '../viz/liveFrames.ts'
import { useCanvasLoop } from '../viz/useCanvasLoop.ts'

const AXIS_W = 42 // same as the spectrogram's frequency axis
const AXIS_H = 16

export function DetectionTimeline() {
  const config = useAppStore((s) => s.config)
  const classes = useMemo(() => config?.classes ?? [], [config])

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      ctx.fillStyle = '#020617'
      ctx.fillRect(0, 0, w, h)
      if (!classes.length) return
      const plotW = w - AXIS_W
      const laneH = (h - AXIS_H) / classes.length
      const frames = liveFrames.detections
      const now = frames[frames.length - 1]?.t ?? liveFrames.latestT
      const x = (t: number) => AXIS_W + plotW - ((now - t) / TIMELINE_SECONDS) * plotW
      ctx.font = '10px ui-sans-serif, system-ui, sans-serif'
      classes.forEach((cls, i) => {
        const y = i * laneH
        ctx.fillStyle = i % 2 ? '#0b1220' : '#0f172a'
        ctx.fillRect(AXIS_W, y, plotW, laneH)
        for (let k = 0; k < frames.length; k++) {
          const f = frames[k]
          if (!f) continue
          const next = frames[k + 1]
          const x0 = x(f.t)
          const x1 = next ? x(next.t) : x0 + 2
          const p = f.probabilities[cls.name] ?? 0
          const on = f.active.includes(cls.name)
          ctx.fillStyle = on ? `rgba(16,185,129,${0.35 + 0.65 * p})` : `rgba(148,163,184,${0.5 * p})`
          ctx.fillRect(x0, y + 1, Math.max(1, x1 - x0), laneH - 2)
        }
        ctx.fillStyle = '#94a3b8'
        ctx.textAlign = 'right'
        ctx.textBaseline = 'middle'
        ctx.fillText(cls.display_name.slice(0, 7), AXIS_W - 4, y + laneH / 2)
      })
      ctx.textBaseline = 'top'
      ctx.textAlign = 'left'
      ctx.fillStyle = '#94a3b8'
      ctx.fillText(`-${TIMELINE_SECONDS} s`, AXIS_W + 2, h - AXIS_H + 3)
      ctx.textAlign = 'right'
      ctx.fillText('now', w - 2, h - AXIS_H + 3)
    },
    [classes],
  )
  const ref = useCanvasLoop(draw)
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="Detection timeline"
      className="h-40 w-full rounded"
      data-testid="detection-timeline"
    />
  )
}
