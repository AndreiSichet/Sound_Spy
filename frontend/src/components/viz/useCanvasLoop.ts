// Shared requestAnimationFrame loop for the live canvases.
// Draws only when liveFrames changed (or every `idleMs` for time-based effects such as peak hold),
// sizes the backing store for the device pixel ratio, and stops when the component unmounts.
import { useEffect, useRef } from 'react'
import { liveFrames } from './liveFrames.ts'

export type DrawFn = (ctx: CanvasRenderingContext2D, width: number, height: number, now: number) => void

export function useCanvasLoop(draw: DrawFn, idleMs = 100) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const drawRef = useRef(draw)

  useEffect(() => {
    drawRef.current = draw
  }, [draw])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return // e.g. in tests without a canvas implementation
    const raf = window.requestAnimationFrame?.bind(window) ?? ((cb: FrameRequestCallback) =>
      window.setTimeout(() => cb(performance.now()), 16))
    const cancel = window.cancelAnimationFrame?.bind(window) ?? window.clearTimeout.bind(window)
    let handle = 0
    let lastVersion = -1
    let lastDraw = 0

    const frame = (now: number) => {
      const ratio = window.devicePixelRatio || 1
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      const resized =
        canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)
      if (resized) {
        canvas.width = Math.round(width * ratio)
        canvas.height = Math.round(height * ratio)
      }
      if (resized || liveFrames.version !== lastVersion || now - lastDraw > idleMs) {
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
        drawRef.current(ctx, width, height, now)
        lastVersion = liveFrames.version
        lastDraw = now
      }
      handle = raf(frame)
    }
    handle = raf(frame)
    return () => cancel(handle)
  }, [idleMs])

  return canvasRef
}
