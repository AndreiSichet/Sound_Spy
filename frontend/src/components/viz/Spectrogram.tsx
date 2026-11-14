// Scrolling mel spectrogram of the last SPECTROGRAM_SECONDS.
// Time runs left to right (newest at the right), frequency bottom to top on the mel scale (the
// bands are equally spaced in mel, like hearing), brightness = level from the floor up to 0 dB.
// Columns are painted once into an offscreen canvas that scrolls; each frame only blits it.
import { useCallback, useRef } from 'react'
import { SPECTROGRAM_SECONDS, liveFrames } from './liveFrames.ts'
import { COLOUR_LUT, SPECTROGRAM_DB_FLOOR, dbToColourIndex, melFraction } from './scales.ts'
import { useCanvasLoop } from './useCanvasLoop.ts'

const AXIS_W = 42
const AXIS_H = 16
const FREQ_TICKS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

function formatHz(hz: number): string {
  return hz >= 1000 ? `${hz / 1000} kHz` : `${hz} Hz`
}

export function Spectrogram() {
  const offscreen = useRef<HTMLCanvasElement | null>(null)
  const drawnSeq = useRef(0)

  const draw = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const columns = liveFrames.spectrogram
    const latest = columns[columns.length - 1]
    const bands = latest?.magnitudes.length ?? 128
    // One offscreen pixel column per spectrogram column of the visible history (~25 Hz * 10 s).
    const capacity = 250
    let off = offscreen.current
    if (!off || off.height !== bands) {
      off = document.createElement('canvas')
      off.width = capacity
      off.height = bands
      offscreen.current = off
      drawnSeq.current = 0
    }
    const octx = off.getContext('2d')
    if (octx && latest) {
      const fresh = columns.filter((c) => c.seq > drawnSeq.current)
      if (latest.seq < drawnSeq.current) drawnSeq.current = 0 // history was reset
      if (fresh.length) {
        const shift = Math.min(fresh.length, capacity)
        octx.drawImage(off, -shift, 0)
        const image = octx.createImageData(1, bands)
        fresh.slice(-shift).forEach((col, i) => {
          for (let b = 0; b < bands; b++) {
            const level = dbToColourIndex(col.magnitudes[b] ?? SPECTROGRAM_DB_FLOOR)
            const row = bands - 1 - b // low frequencies at the bottom
            image.data[row * 4] = COLOUR_LUT[level * 3] ?? 0
            image.data[row * 4 + 1] = COLOUR_LUT[level * 3 + 1] ?? 0
            image.data[row * 4 + 2] = COLOUR_LUT[level * 3 + 2] ?? 0
            image.data[row * 4 + 3] = 255
          }
          octx.putImageData(image, capacity - shift + i, 0)
        })
        drawnSeq.current = latest.seq
      }
    }

    ctx.fillStyle = '#020617'
    ctx.fillRect(0, 0, w, h)
    const plotW = w - AXIS_W
    const plotH = h - AXIS_H
    if (latest) {
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(off, 0, 0, capacity, bands, AXIS_W, 0, plotW, plotH)
    }
    ctx.font = '10px ui-sans-serif, system-ui, sans-serif'
    ctx.fillStyle = '#94a3b8'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    const fmin = liveFrames.freqMinHz
    const fmax = liveFrames.freqMaxHz
    for (const hz of FREQ_TICKS) {
      if (hz <= fmin || hz >= fmax) continue
      const y = plotH - melFraction(hz, fmin, fmax) * plotH
      ctx.fillText(formatHz(hz), AXIS_W - 4, y)
      ctx.fillRect(AXIS_W - 2, y, 2, 1)
    }
    ctx.textBaseline = 'top'
    ctx.textAlign = 'left'
    ctx.fillText(`-${SPECTROGRAM_SECONDS} s`, AXIS_W + 2, plotH + 3)
    ctx.textAlign = 'center'
    ctx.fillText(`-${SPECTROGRAM_SECONDS / 2} s`, AXIS_W + plotW / 2, plotH + 3)
    ctx.textAlign = 'right'
    ctx.fillText('now', w - 2, plotH + 3)
  }, [])
  const ref = useCanvasLoop(draw)
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="Spectrogram"
      className="h-64 w-full rounded"
      data-testid="spectrogram"
    />
  )
}
