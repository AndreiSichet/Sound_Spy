// Live visual data, deliberately OUTSIDE React state.
//
// Levels, waveform and spectrogram frames arrive 20-25 times per second each. Putting them in
// React state would re-render the page ~70 times a second. Instead the WebSocket handler writes
// them here, and the canvas components read this module inside requestAnimationFrame. A `version`
// counter tells a component whether anything changed since it last drew.
//
// History is trimmed by time (from each message's `t`, seconds since the session started), so the
// buffers do not depend on the server's frame rates.
import type {
  DetectionFrame,
  LevelFrame,
  SpectrogramFrame,
  WaveformFrame,
} from '../../api/types.ts'

export const WAVEFORM_SECONDS = 5
export const SPECTROGRAM_SECONDS = 10
export const TIMELINE_SECONDS = 30
export const PEAK_HOLD_MS = 1000
export const CLIP_LED_MS = 1000

export interface TimedWaveform extends WaveformFrame {
  t: number
}

export interface SpectrogramColumn {
  seq: number
  t: number
  magnitudes: Float32Array
}

export interface TimedDetection extends DetectionFrame {
  t: number
}

export class LiveFrames {
  version = 0
  levels: (LevelFrame & { t: number }) | null = null
  peakHold: number[] = []
  peakHoldAt: number[] = []
  clipAt: number[] = []
  waveform: TimedWaveform[] = []
  spectrogram: SpectrogramColumn[] = []
  spectrogramSeq = 0
  detections: TimedDetection[] = []
  freqMinHz = 50
  freqMaxHz = 24000
  latestT = 0

  reset(): void {
    this.levels = null
    this.peakHold = []
    this.peakHoldAt = []
    this.clipAt = []
    this.waveform = []
    this.spectrogram = []
    this.detections = []
    this.latestT = 0
    this.version += 1
  }

  /** A session restarted (time went backwards): drop the old history. */
  private checkTime(t: number): void {
    if (t + 0.5 < this.latestT) this.reset()
    this.latestT = Math.max(this.latestT, t)
  }

  pushLevels(frame: LevelFrame, t: number, now: number = performance.now()): void {
    this.checkTime(t)
    this.levels = { ...frame, t }
    frame.peak_dbfs.forEach((peak, ch) => {
      const held = this.peakHold[ch]
      const heldAt = this.peakHoldAt[ch] ?? 0
      if (held === undefined || peak >= held || now - heldAt > PEAK_HOLD_MS) {
        this.peakHold[ch] = peak
        this.peakHoldAt[ch] = now
      }
      if (frame.clipping[ch]) this.clipAt[ch] = now
    })
    this.version += 1
  }

  pushWaveform(frame: WaveformFrame, t: number): void {
    this.checkTime(t)
    this.waveform.push({ ...frame, t })
    const oldest = t - WAVEFORM_SECONDS
    while (this.waveform.length && (this.waveform[0]?.t ?? t) < oldest) this.waveform.shift()
    this.version += 1
  }

  pushSpectrogram(frame: SpectrogramFrame, t: number): void {
    this.checkTime(t)
    this.spectrogramSeq += 1
    this.spectrogram.push({
      seq: this.spectrogramSeq,
      t,
      magnitudes: Float32Array.from(frame.magnitudes_db),
    })
    this.freqMaxHz = frame.freq_max_hz
    if (frame.freq_min_hz !== undefined) this.freqMinHz = frame.freq_min_hz
    const oldest = t - SPECTROGRAM_SECONDS
    while (this.spectrogram.length && (this.spectrogram[0]?.t ?? t) < oldest) {
      this.spectrogram.shift()
    }
    this.version += 1
  }

  pushDetections(frame: DetectionFrame, t: number): void {
    this.checkTime(t)
    this.detections.push({ ...frame, t })
    const oldest = t - TIMELINE_SECONDS
    while (this.detections.length && (this.detections[0]?.t ?? t) < oldest) {
      this.detections.shift()
    }
    this.version += 1
  }

  clipping(ch: number, now: number = performance.now()): boolean {
    const at = this.clipAt[ch]
    return at !== undefined && now - at < CLIP_LED_MS
  }
}

export const liveFrames = new LiveFrames()
