// Scales and colours shared by the live visuals (kept out of the component files so Vite's fast
// refresh works).

export const METER_MIN_DB = -60
export const METER_MAX_DB = 6
export const SPECTROGRAM_DB_FLOOR = -100

/** Position of a dBFS value on the meter scale, 0 (bottom) .. 1 (top), clamped. */
export function dbToFraction(db: number): number {
  return Math.min(1, Math.max(0, (db - METER_MIN_DB) / (METER_MAX_DB - METER_MIN_DB)))
}

export const hzToMel = (hz: number) => 2595 * Math.log10(1 + hz / 700)

/** Fraction (0 = bottom, 1 = top) of a frequency on a mel axis from fmin to fmax. */
export function melFraction(hz: number, fmin: number, fmax: number): number {
  return (hzToMel(hz) - hzToMel(fmin)) / (hzToMel(fmax) - hzToMel(fmin))
}

// Spectrogram colour map: dark navy -> violet -> orange -> pale yellow (dark = quiet).
const STOPS: [number, number, number][] = [
  [8, 10, 30],
  [60, 15, 110],
  [170, 40, 120],
  [240, 110, 50],
  [252, 240, 170],
]

export const COLOUR_LUT: Uint8ClampedArray = (() => {
  const lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const pos = (i / 255) * (STOPS.length - 1)
    const lo = Math.floor(pos)
    const hi = Math.min(STOPS.length - 1, lo + 1)
    const f = pos - lo
    for (let c = 0; c < 3; c++) {
      const a = STOPS[lo]?.[c] ?? 0
      const b = STOPS[hi]?.[c] ?? 0
      lut[i * 3 + c] = a + (b - a) * f
    }
  }
  return lut
})()

/** Colour-map index 0..255 for a dB value between the floor and 0 dB. */
export function dbToColourIndex(db: number, floor: number = SPECTROGRAM_DB_FLOOR): number {
  return Math.round(Math.min(1, Math.max(0, (db - floor) / -floor)) * 255)
}
