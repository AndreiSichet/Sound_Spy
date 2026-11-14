import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EngineStatus, Recording } from '../src/api/types.ts'
import { HEADPHONES_KEY } from '../src/components/session/headphones.ts'
import { MonitorControls } from '../src/components/session/MonitorControls.tsx'
import { RecordingsList } from '../src/components/session/RecordingsList.tsx'
import { SessionBar } from '../src/components/session/SessionBar.tsx'
import { LevelMeters } from '../src/components/viz/LevelMeters.tsx'
import { LiveFrames, liveFrames } from '../src/components/viz/liveFrames.ts'
import { dbToFraction, melFraction } from '../src/components/viz/scales.ts'
import { Spectrogram } from '../src/components/viz/Spectrogram.tsx'
import { Waveform } from '../src/components/viz/Waveform.tsx'
import { handleServerMessage } from '../src/hooks/useSessionSocket.ts'
import { initialAppState, useAppStore } from '../src/store/appStore.ts'

const RUNNING: EngineStatus = {
  state: 'running',
  mode: 'semantic',
  device: 'Microphone Array (Realtek(R) Audio)',
  sample_rate: 48000,
  channels: 2,
  message: '',
  source: 'mic',
  output_device: 'Speakers (Realtek(R) Audio)',
  monitoring: false,
  muted: false,
  gain_db: 0,
  recording: false,
  latency_ms: 23.4,
  rtf_p95: 0.012,
  input_overflows: 0,
  output_underruns: 0,
  dropped_chunks: 0,
}

const RECORDING: Recording = {
  id: 3,
  created_at: '2026-11-14T14:32:56.131Z',
  path: 'data/raw/speech_20261114-163256-131.wav',
  duration_s: 5,
  sample_rate: 48000,
  channels: 2,
  device: 'Microphone Array (Realtek(R) Audio)',
  profile: 'laptop',
  label: 'speech',
  notes: '',
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 })
}

beforeEach(() => {
  useAppStore.setState({ ...initialAppState, backend: 'up' })
  liveFrames.reset()
  try {
    window.localStorage.clear()
  } catch {
    // ignore
  }
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('WebSocket message routing', () => {
  it('sends status to the store and visual frames to liveFrames (not React state)', () => {
    const before = liveFrames.version
    handleServerMessage({ type: 'status', t: 1, payload: RUNNING })
    handleServerMessage({
      type: 'levels', t: 1.05, payload: { rms_dbfs: [-30, -31], peak_dbfs: [-10, -12], clipping: [false, true] },
    })
    handleServerMessage({ type: 'waveform', t: 1.04, payload: { mins: [-0.1], maxs: [0.2] } })
    handleServerMessage({
      type: 'spectrogram', t: 1.04, payload: { magnitudes_db: [-50, -60], freq_max_hz: 24000, freq_min_hz: 50 },
    })
    expect(useAppStore.getState().status?.latency_ms).toBe(23.4)
    expect(liveFrames.levels?.rms_dbfs).toEqual([-30, -31])
    expect(liveFrames.peakHold).toEqual([-10, -12])
    expect(liveFrames.clipping(1)).toBe(true)
    expect(liveFrames.clipping(0)).toBe(false)
    expect(liveFrames.waveform).toHaveLength(1)
    expect(Array.from(liveFrames.spectrogram[0]?.magnitudes ?? [])).toEqual([-50, -60])
    expect(liveFrames.freqMinHz).toBe(50)
    expect(liveFrames.version).toBeGreaterThan(before)
  })

  it('turns warnings and errors into toasts, without duplicates', () => {
    const warning = { type: 'warning' as const, t: 2, payload: { code: 'rtf_high', message: 'slow' } }
    handleServerMessage(warning)
    handleServerMessage(warning)
    handleServerMessage({ type: 'error', t: 0, payload: { message: 'bad' } })
    handleServerMessage({ type: 'status', t: 3, payload: { ...RUNNING, state: 'error', message: 'mic lost' } })
    const toasts = useAppStore.getState().toasts
    expect(toasts.map((t) => [t.kind, t.message])).toEqual([
      ['warning', 'slow'],
      ['error', 'bad'],
      ['error', 'Session stopped: mic lost'],
    ])
  })
})

describe('liveFrames history', () => {
  it('keeps 5 s of waveform and 10 s of spectrogram, and resets when time goes back', () => {
    const frames = new LiveFrames()
    for (let i = 1; i <= 400; i++) {
      frames.pushWaveform({ mins: [0], maxs: [0] }, i * 0.04)
      frames.pushSpectrogram({ magnitudes_db: [0], freq_max_hz: 24000 }, i * 0.04)
    }
    const wave = frames.waveform
    expect((wave[wave.length - 1]?.t ?? 0) - (wave[0]?.t ?? 0)).toBeLessThanOrEqual(5)
    const spec = frames.spectrogram
    expect((spec[spec.length - 1]?.t ?? 0) - (spec[0]?.t ?? 0)).toBeLessThanOrEqual(10)
    expect(spec.length).toBeGreaterThan(240)
    frames.pushLevels({ rms_dbfs: [-20], peak_dbfs: [-3], clipping: [false] }, 0.05) // new session
    expect(frames.waveform).toHaveLength(0)
    expect(frames.spectrogram).toHaveLength(0)
  })

  it('holds a peak for about a second', () => {
    const frames = new LiveFrames()
    frames.pushLevels({ rms_dbfs: [-20], peak_dbfs: [-3], clipping: [false] }, 0.05, 1000)
    frames.pushLevels({ rms_dbfs: [-20], peak_dbfs: [-30], clipping: [false] }, 0.1, 1500)
    expect(frames.peakHold[0]).toBe(-3)
    frames.pushLevels({ rms_dbfs: [-20], peak_dbfs: [-30], clipping: [false] }, 0.15, 2100)
    expect(frames.peakHold[0]).toBe(-30)
  })
})

describe('MonitorControls', () => {
  it('asks for headphones before listening the first time, then remembers', async () => {
    useAppStore.setState({ status: RUNNING })
    const fetchMock = vi.fn(async (_: RequestInfo | URL, init?: RequestInit) =>
      jsonResponse({ ...RUNNING, monitoring: JSON.parse(String(init?.body)).enabled }),
    )
    vi.stubGlobal('fetch', fetchMock)
    render(<MonitorControls />)

    fireEvent.click(screen.getByRole('checkbox', { name: 'Listen' }))
    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveTextContent('feeds back into the microphone and howls')
    const go = screen.getByRole('button', { name: 'Start listening' })
    expect(go).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('checkbox', { name: "I'm using headphones" }))
    fireEvent.click(go)
    await waitFor(() => expect(useAppStore.getState().status?.monitoring).toBe(true))
    const [url, init] = fetchMock.mock.calls[0] ?? []
    expect(String(url)).toBe('/api/session/monitor')
    expect(JSON.parse(String(init?.body))).toEqual({ enabled: true, gain_db: 0, muted: false })
    expect(window.localStorage.getItem(HEADPHONES_KEY)).toBe('yes')

    // Off, then on again: no second confirmation.
    fireEvent.click(screen.getByRole('checkbox', { name: 'Listen' }))
    await waitFor(() => expect(useAppStore.getState().status?.monitoring).toBe(false))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Listen' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    await waitFor(() => expect(useAppStore.getState().status?.monitoring).toBe(true))
  })

  it('disables Listen and says why when the session has no output device', () => {
    useAppStore.setState({
      status: {
        ...RUNNING,
        output_device: null,
        message: 'Listening is unavailable for this session: device busy',
      },
    })
    render(<MonitorControls />)
    expect(screen.getByRole('checkbox', { name: 'Listen' })).toBeDisabled()
    expect(screen.getByTestId('no-output')).toHaveTextContent('Listening unavailable')
  })

  it('cancel keeps monitoring off', () => {
    useAppStore.setState({ status: RUNNING })
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<MonitorControls />)
    fireEvent.click(screen.getByRole('checkbox', { name: 'Listen' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('SessionBar', () => {
  it('shows the live figures of a running session', () => {
    useAppStore.setState({ status: { ...RUNNING, output_underruns: 2 } })
    render(<SessionBar />)
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled()
    expect(screen.getByTestId('session-state')).toHaveTextContent('running')
    const stats = screen.getByTestId('session-stats')
    expect(stats).toHaveTextContent('Realtek')
    expect(stats).toHaveTextContent('48000 Hz · 2 ch')
    expect(stats).toHaveTextContent('23.4 ms')
    expect(stats).toHaveTextContent('0.012')
    expect(stats).toHaveTextContent('0 / 2 / 0')
  })

  it('starts a microphone session, or a looped file session from a recording', async () => {
    useAppStore.setState({ status: null, recordings: [RECORDING] })
    const fetchMock = vi.fn(async () => jsonResponse(RUNNING))
    vi.stubGlobal('fetch', fetchMock)
    render(<SessionBar />)
    fireEvent.click(screen.getByRole('button', { name: 'Start' }))
    await waitFor(() => expect(useAppStore.getState().status?.state).toBe('running'))
    const calls = fetchMock.mock.calls as unknown as [string, RequestInit][]
    expect(calls[0]?.[0]).toBe('/api/session/start')
    expect(JSON.parse(String(calls[0]?.[1].body))).toEqual({ mode: 'semantic', source: { type: 'mic' } })

    act(() => useAppStore.setState({ status: null }))
    fireEvent.change(screen.getByRole('combobox', { name: 'Source' }), { target: { value: 'file:3' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(JSON.parse(String(calls[1]?.[1].body)).source).toEqual({
      type: 'file', path: RECORDING.path, loop: true,
    })
  })

  it('reports a failed start as a toast', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail: 'mic busy' }), { status: 409 })))
    render(<SessionBar />)
    fireEvent.click(screen.getByRole('button', { name: 'Start' }))
    await waitFor(() => expect(useAppStore.getState().toasts[0]?.message).toBe('Could not start: mic busy'))
  })
})

describe('RecordingsList', () => {
  it('lists recordings and plays one from the audio route', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse([RECORDING])))
    render(<RecordingsList />)
    const play = await screen.findByRole('button', { name: `Play ${RECORDING.path.split('/').pop()}` })
    expect(screen.getByText('speech')).toBeInTheDocument()
    expect(screen.getByText('5.0 s')).toBeInTheDocument()
    fireEvent.click(play)
    expect(screen.getByTestId('audio-3')).toHaveAttribute('src', '/api/recordings/3/audio')
  })
})

describe('scales', () => {
  it('maps dBFS onto the meter and Hz onto the mel axis', () => {
    expect(dbToFraction(-60)).toBe(0)
    expect(dbToFraction(6)).toBe(1)
    expect(dbToFraction(-100)).toBe(0)
    expect(dbToFraction(-27)).toBeCloseTo(0.5)
    expect(melFraction(50, 50, 24000)).toBe(0)
    expect(melFraction(24000, 50, 24000)).toBeCloseTo(1)
    expect(melFraction(1000, 50, 24000)).toBeGreaterThan(melFraction(500, 50, 24000))
  })
})

describe('live visuals', () => {
  it('mount and draw without errors (canvas mocked)', async () => {
    const calls: string[] = []
    const ctx = new Proxy(
      {
        createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      } as Record<string, unknown>,
      {
        get(target, prop: string) {
          if (prop in target) return target[prop]
          return () => {
            calls.push(prop)
          }
        },
        set(target, prop: string, value) {
          target[prop] = value
          return true
        },
      },
    )
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
      ctx as unknown as CanvasRenderingContext2D,
    )
    liveFrames.pushLevels({ rms_dbfs: [-20, -22], peak_dbfs: [-6, -8], clipping: [false, false] }, 0.05)
    liveFrames.pushWaveform({ mins: [-0.5, -0.2], maxs: [0.5, 0.3] }, 0.04)
    liveFrames.pushSpectrogram({ magnitudes_db: [-40, -60, -80], freq_max_hz: 24000, freq_min_hz: 50 }, 0.04)
    render(
      <>
        <LevelMeters />
        <Waveform />
        <Spectrogram />
      </>,
    )
    expect(screen.getByTestId('level-meters')).toBeInTheDocument()
    expect(screen.getByTestId('waveform')).toBeInTheDocument()
    expect(screen.getByTestId('spectrogram')).toBeInTheDocument()
    await waitFor(() => expect(calls).toContain('fillRect'))
    expect(calls).toContain('putImageData') // a spectrogram column was painted
  })
})
