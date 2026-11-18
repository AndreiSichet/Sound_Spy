import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ConfigResponse, EngineStatus, Recording, Run } from '../src/api/types.ts'
import { LevelMeters } from '../src/components/viz/LevelMeters.tsx'
import { ClassChips } from '../src/components/detection/ClassChips.tsx'
import { ABPlayer } from '../src/components/session/ABPlayer.tsx'
import { ExtractionControls } from '../src/components/session/ExtractionControls.tsx'
import { RecordingsList } from '../src/components/session/RecordingsList.tsx'
import { SessionBar } from '../src/components/session/SessionBar.tsx'
import { liveFrames } from '../src/components/viz/liveFrames.ts'
import { SpectrogramPanel } from '../src/components/viz/SpectrogramPanel.tsx'
import { handleServerMessage, sendTarget } from '../src/hooks/useSessionSocket.ts'
import { initialAppState, useAppStore } from '../src/store/appStore.ts'

vi.mock('../src/hooks/useSessionSocket.ts', async (importOriginal) => {
  const original = await importOriginal<typeof import('../src/hooks/useSessionSocket.ts')>()
  return { ...original, sendTarget: vi.fn(() => true) }
})

const CONFIG: ConfigResponse = {
  classes: [
    { name: 'speech', display_name: 'Speech', description: 'People talking' },
    { name: 'music', display_name: 'Music', description: 'Songs' },
  ],
  modes: [{ mode: 'semantic', available: true, reason: null }],
  device_profile: 'laptop',
}

const RUNNING: EngineStatus = {
  state: 'running',
  mode: 'semantic',
  device: 'Microphone Array',
  sample_rate: 48000,
  channels: 2,
  message: '',
  latency_ms: 105.3,
  bypass: false,
  detector: { model: 'det_mn10_v1', device: 'cuda', hop_s: 0.25, latency_ms: 28, skipped: 0 },
  extraction: { method: 'mask_baseline', latency_ms: 21.33, gating: true, target_active: false },
}

const RECORDING: Recording = {
  id: 3,
  created_at: '2026-11-14T14:32:56.131Z',
  path: 'data/raw/speech_1.wav',
  duration_s: 5,
  sample_rate: 48000,
  channels: 2,
  device: 'mic',
  profile: 'laptop',
  label: 'speech',
  notes: '',
}

const RUN: Run = {
  run_id: 'run_20261116-101500-123',
  created_at: '2026-11-16T08:15:00.123Z',
  recording_id: 3,
  recording_path: 'data/raw/speech_1.wav',
  labels: ['speech'],
  attenuation_db: 24,
  method: 'mask_baseline',
  status: 'pending',
  error: null,
  output_path: null,
  duration_s: null,
  processing_seconds: null,
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status })
}

beforeEach(() => {
  useAppStore.setState({ ...initialAppState, backend: 'up', config: CONFIG, status: RUNNING })
  liveFrames.reset()
  vi.mocked(sendTarget).mockClear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('extraction controls', () => {
  it('the attenuation slider sends the target with the selected classes', () => {
    useAppStore.setState({ selected: ['speech'] })
    render(<ExtractionControls />)
    const slider = screen.getByLabelText('Attenuation (dB)')
    expect(slider).toHaveAttribute('min', '0')
    expect(slider).toHaveAttribute('max', '40')
    expect(slider).toHaveValue('15') // the live default (step 07)
    fireEvent.change(slider, { target: { value: '12' } })
    expect(useAppStore.getState().attenuation).toBe(12)
    expect(vi.mocked(sendTarget).mock.calls.at(-1)?.[0]).toEqual({
      labels: ['speech'], direction: null, source_id: null, attenuation_db: 12,
    })
    expect(screen.getByText('12 dB')).toBeInTheDocument()
  })

  it('falls back to REST for the target when the live channel is down', () => {
    vi.mocked(sendTarget).mockReturnValueOnce(false)
    const fetchMock = vi.fn(async () => json({}))
    vi.stubGlobal('fetch', fetchMock)
    render(<ExtractionControls />)
    fireEvent.change(screen.getByLabelText('Attenuation (dB)'), { target: { value: '20' } })
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/session/target')
    expect(init.method).toBe('PUT')
    expect(JSON.parse(String(init.body)).attenuation_db).toBe(20)
  })

  it('bypass sends the request and shows the new state', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const enabled = JSON.parse(String(init?.body)).enabled as boolean
      return json({ ...RUNNING, bypass: enabled })
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<ExtractionControls />)
    const button = screen.getByRole('button', { name: 'Bypass (A/B)' })
    expect(button).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(button)
    await waitFor(() => expect(useAppStore.getState().status?.bypass).toBe(true))
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/session/bypass')
    expect(init.method).toBe('PUT')
    expect(JSON.parse(String(init.body))).toEqual({ enabled: true })
    const on = screen.getByRole('button', { name: 'Bypass on (unprocessed)' })
    expect(on).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(on)
    await waitFor(() => expect(useAppStore.getState().status?.bypass).toBe(false))
    expect(JSON.parse(String((fetchMock.mock.calls[1] as unknown as [string, RequestInit])[1].body)))
      .toEqual({ enabled: false })
  })

  it('chips send the current attenuation and explain an empty selection', () => {
    useAppStore.setState({ attenuation: 18 })
    render(<ClassChips />)
    expect(screen.getByTestId('selection-hint')).toHaveTextContent(
      'Select one or more sounds to enhance them.',
    )
    fireEvent.click(screen.getByTestId('chip-music'))
    expect(vi.mocked(sendTarget).mock.calls.at(-1)?.[0]).toEqual({
      labels: ['music'], direction: null, source_id: null, attenuation_db: 18,
    })
    expect(screen.getByTestId('selection-hint')).toHaveTextContent('turned down by 18 dB')
  })

  it('says when selection cannot gate because detection is off', () => {
    useAppStore.setState({
      selected: ['speech'],
      status: {
        ...RUNNING,
        detector: null,
        extraction: { method: 'mask_baseline', latency_ms: 21.33, gating: false, target_active: false },
      },
    })
    render(<ClassChips />)
    expect(screen.getByTestId('selection-hint')).toHaveTextContent('only steady noise is reduced')
  })

  it('the session bar shows the extraction method and its latency', () => {
    render(<SessionBar />)
    const stats = screen.getByTestId('session-stats')
    expect(stats).toHaveTextContent('Baseline · +21.33 ms')
    expect(stats).not.toHaveTextContent('Model time')
    expect(stats).toHaveTextContent('105.3 ms')
    expect(stats).not.toHaveTextContent('no gating')
    act(() => useAppStore.setState({ status: { ...RUNNING, bypass: true } }))
    expect(stats).toHaveTextContent('bypassed')
  })
})

describe('extraction method (step 07)', () => {
  const METHODS: ConfigResponse = {
    ...CONFIG,
    extraction_methods: [
      { name: 'neural_tse', display_name: 'Neural', available: true, reason: null },
      { name: 'mask_baseline', display_name: 'Baseline', available: true, reason: null },
    ],
    default_method: 'neural_tse',
    default_attenuation_db: 15,
  }
  const IDLE: EngineStatus = {
    state: 'idle', mode: null, device: null, sample_rate: null, channels: null, message: '',
    extraction_method: 'neural_tse',
  }

  it('is chosen while stopped and sent to the backend', async () => {
    const fetchMock = vi.fn(async () =>
      json({ ...IDLE, extraction_method: 'mask_baseline' }),
    )
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({ config: METHODS, status: IDLE })
    render(<ExtractionControls />)
    const select = screen.getByLabelText('Extraction method')
    expect(select).toHaveValue('neural_tse')
    expect(select).toBeEnabled()
    expect(screen.getByRole('option', { name: 'Neural' })).toBeInTheDocument()
    fireEvent.change(select, { target: { value: 'mask_baseline' } })
    await waitFor(() => expect(select).toHaveValue('mask_baseline'))
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/session/method')
    expect(init.method).toBe('PUT')
    expect(JSON.parse(String(init.body))).toEqual({ method: 'mask_baseline' })
    expect(useAppStore.getState().status?.extraction_method).toBe('mask_baseline')
  })

  it('cannot be changed during a session', () => {
    useAppStore.setState({ config: METHODS, status: { ...RUNNING, extraction_method: 'neural_tse' } })
    render(<ExtractionControls />)
    expect(screen.getByLabelText('Extraction method')).toBeDisabled()
  })

  it('greys out a method that cannot run and shows the one that will', () => {
    const reason = 'its model file does not exist yet; train it with `soundspy train extraction`'
    useAppStore.setState({
      status: IDLE, // the configured choice is still "neural_tse"
      config: {
        ...METHODS,
        default_method: 'mask_baseline',
        extraction_methods: [
          { name: 'neural_tse', display_name: 'Neural', available: false, reason },
          { name: 'mask_baseline', display_name: 'Baseline', available: true, reason: null },
        ],
      },
    })
    render(<ExtractionControls />)
    const option = screen.getByRole('option', { name: 'Neural (not available)' })
    expect(option).toBeDisabled()
    expect(option).toHaveAttribute('title', reason)
    expect(screen.getByLabelText('Extraction method')).toHaveValue('mask_baseline')
  })

  it('shows a failed change as a toast', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => json({ detail: 'stop the session to change the extraction method' }, 409)),
    )
    useAppStore.setState({ config: METHODS, status: IDLE })
    render(<ExtractionControls />)
    fireEvent.change(screen.getByLabelText('Extraction method'), {
      target: { value: 'mask_baseline' },
    })
    await waitFor(() => expect(useAppStore.getState().toasts).toHaveLength(1))
    expect(useAppStore.getState().toasts[0]?.message).toContain('stop the session')
  })

  it('the session bar shows the neural method, its device and the model time', () => {
    useAppStore.setState({
      status: {
        ...RUNNING,
        rtf_p95: 0.41,
        extraction_method: 'neural_tse',
        extraction: {
          method: 'neural_tse', latency_ms: 10.6, gating: false, target_active: true,
          device: 'cuda', model: 'tse_waveformer_v1', model_ms: 6.1, model_ms_p95: 7.9,
        },
      },
    })
    render(<SessionBar />)
    const stats = screen.getByTestId('session-stats')
    expect(stats).toHaveTextContent('Neural · cuda · +10.6 ms')
    expect(stats).toHaveTextContent('Model time')
    expect(stats).toHaveTextContent('6.1 ms (p95 7.9)')
    expect(stats).toHaveTextContent('0.410')
    expect(stats).not.toHaveTextContent('no gating') // the neural method needs no detector
  })

  it('the attenuation slider starts at the backend default until it is moved', () => {
    expect(useAppStore.getState().attenuation).toBe(15)
    act(() => useAppStore.getState().setConfig({ ...METHODS, default_attenuation_db: 12 }))
    expect(useAppStore.getState().attenuation).toBe(12)
    render(<ExtractionControls />)
    fireEvent.change(screen.getByLabelText('Attenuation (dB)'), { target: { value: '25' } })
    act(() => useAppStore.getState().setConfig({ ...METHODS, default_attenuation_db: 12 }))
    expect(useAppStore.getState().attenuation).toBe(25) // a reconnect does not undo the choice
  })
})

describe('output level meter (step 07)', () => {
  it('keeps its own level, peak hold and clip memory', () => {
    const frame = (peak: number, clip = false) => ({
      rms_dbfs: [peak - 10], peak_dbfs: [peak], clipping: [clip],
    })
    liveFrames.pushLevels({ ...frame(-6), rms_dbfs: [-16, -18], peak_dbfs: [-6, -8],
      clipping: [false, false] }, 1.0, 1000)
    liveFrames.pushLevels({ ...frame(-30, true), source: 'output' }, 1.0, 1000)
    expect(liveFrames.levelsOf('input')?.peak_dbfs).toEqual([-6, -8])
    expect(liveFrames.levelsOf('output')?.peak_dbfs).toEqual([-30])
    expect(liveFrames.peakHoldOf('input')).toEqual([-6, -8])
    expect(liveFrames.peakHoldOf('output')).toEqual([-30])
    expect(liveFrames.clipping(0, 1100, 'output')).toBe(true)
    expect(liveFrames.clipping(0, 1100)).toBe(false) // the input did not clip
    // a lower peak is held for about a second, then the hold follows
    liveFrames.pushLevels({ ...frame(-40), source: 'output' }, 1.2, 1200)
    expect(liveFrames.peakHoldOf('output')).toEqual([-30])
    liveFrames.pushLevels({ ...frame(-40), source: 'output' }, 2.3, 2300)
    expect(liveFrames.peakHoldOf('output')).toEqual([-40])
    expect(liveFrames.peakHoldOf('input')).toEqual([-6, -8])
    liveFrames.reset()
    expect(liveFrames.peakHoldOf('output')).toEqual([])
    expect(liveFrames.levelsOf('output')).toBeNull()
  })

  it('is drawn next to the input meters', () => {
    render(
      <>
        <LevelMeters />
        <LevelMeters source="output" />
      </>,
    )
    expect(screen.getByTestId('level-meters')).toHaveAttribute('aria-label', 'Input level meters')
    expect(screen.getByTestId('level-meters-output')).toHaveAttribute(
      'aria-label', 'Output level meter',
    )
  })
})

describe('input / output spectrogram', () => {
  it('routes frames by source and keeps the two histories apart', () => {
    const column = { magnitudes_db: [-50, -40], freq_max_hz: 24000, freq_min_hz: 50 }
    handleServerMessage({ type: 'spectrogram', t: 1, payload: column }) // no source = input
    handleServerMessage({ type: 'spectrogram', t: 1, payload: { ...column, source: 'input' } })
    handleServerMessage({ type: 'spectrogram', t: 1, payload: { ...column, source: 'output' } })
    expect(liveFrames.spectrogram).toHaveLength(2)
    expect(liveFrames.spectrogramOut).toHaveLength(1)
    expect(liveFrames.columns('output')).toBe(liveFrames.spectrogramOut)
    expect(liveFrames.columns('input')).toBe(liveFrames.spectrogram)
    const levels = { rms_dbfs: [-30], peak_dbfs: [-10], clipping: [false] }
    handleServerMessage({ type: 'levels', t: 1.1, payload: { ...levels, source: 'output' } })
    expect(liveFrames.levels).toBeNull() // the input meters ignore output levels
    expect(liveFrames.levelsOut?.rms_dbfs).toEqual([-30])
    handleServerMessage({ type: 'levels', t: 1.2, payload: levels })
    expect(liveFrames.levels?.peak_dbfs).toEqual([-10])
    handleServerMessage({
      type: 'waveform', t: 1.2, payload: { mins: [-0.1], maxs: [0.1], source: 'output' },
    })
    // step 07: the output is shown as a level meter; an output waveform is not kept
    expect(liveFrames.waveform).toHaveLength(0)
    expect('waveformOut' in liveFrames).toBe(false)
    liveFrames.reset()
    expect(liveFrames.spectrogramOut).toHaveLength(0)
    expect(liveFrames.levelsOut).toBeNull()
  })

  it('toggles between input, output and both', () => {
    render(<SpectrogramPanel />)
    // default: both, so the suppression can be seen
    expect(screen.getByTestId('spectrogram')).toBeInTheDocument()
    expect(screen.getByTestId('spectrogram-output')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Both' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Output' }))
    expect(screen.queryByTestId('spectrogram')).toBeNull()
    expect(screen.getByTestId('spectrogram-output')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Output' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Input' }))
    expect(screen.getByTestId('spectrogram')).toBeInTheDocument()
    expect(screen.queryByTestId('spectrogram-output')).toBeNull()
  })
})

/** jsdom has no media playback: give two <audio> elements a working clock and play state. */
function fakeMedia(...elements: HTMLAudioElement[]) {
  for (const el of elements) {
    let paused = true
    let time = 0
    Object.defineProperty(el, 'paused', { get: () => paused, configurable: true })
    Object.defineProperty(el, 'currentTime', {
      get: () => time,
      set: (value: number) => {
        time = value
      },
      configurable: true,
    })
    el.play = vi.fn(async () => {
      paused = false
    })
    el.pause = vi.fn(() => {
      paused = true
    })
  }
}

describe('A/B player', () => {
  it('keeps the playback time when switching between original and processed', () => {
    render(
      <ABPlayer id="r1" originalUrl="/api/recordings/3/audio" processedUrl="/api/runs/r1/audio"
        duration={5} downloadName="r1.wav" />,
    )
    const original = screen.getByTestId('ab-original-r1') as HTMLAudioElement
    const processed = screen.getByTestId('ab-processed-r1') as HTMLAudioElement
    fakeMedia(original, processed)
    expect(original).toHaveAttribute('src', '/api/recordings/3/audio')
    expect(processed).toHaveAttribute('src', '/api/runs/r1/audio')
    // starts on the processed side
    expect(screen.getByRole('button', { name: 'Processed' })).toHaveAttribute('aria-pressed', 'true')

    fireEvent.click(screen.getByRole('button', { name: 'Play' }))
    expect(processed.play).toHaveBeenCalledTimes(1)
    expect(original.play).not.toHaveBeenCalled()
    processed.currentTime = 2.4
    fireEvent.timeUpdate(processed)
    expect(screen.getByLabelText('Position')).toHaveValue('2.4')

    fireEvent.click(screen.getByRole('button', { name: 'Original' }))
    expect(original.currentTime).toBe(2.4) // same moment on the other side
    expect(processed.pause).toHaveBeenCalled()
    expect(original.play).toHaveBeenCalledTimes(1) // and it keeps playing
    expect(screen.getByRole('button', { name: 'Original' })).toHaveAttribute('aria-pressed', 'true')

    original.currentTime = 3.1
    fireEvent.timeUpdate(original)
    fireEvent.click(screen.getByRole('button', { name: 'Pause' }))
    expect(original.pause).toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Processed' }))
    expect(processed.currentTime).toBe(3.1)
    expect(processed.play).toHaveBeenCalledTimes(1) // paused: switching does not start playback

    fireEvent.change(screen.getByLabelText('Position'), { target: { value: '1.25' } })
    expect(processed.currentTime).toBe(1.25)
    expect(original.currentTime).toBe(3.1) // only the active side seeks; it syncs on the switch
    const link = screen.getByRole('link', { name: 'Download' })
    expect(link).toHaveAttribute('href', '/api/runs/r1/audio')
    expect(link).toHaveAttribute('download', 'r1.wav')
  })

  it('plays only the processed side when the original file is gone', () => {
    render(
      <ABPlayer id="r2" originalUrl={null} processedUrl="/api/runs/r2/audio" duration={5}
        downloadName="r2.wav" />,
    )
    expect(screen.queryByTestId('ab-original-r2')).toBeNull()
    expect(screen.getByRole('button', { name: 'Original' })).toBeDisabled()
  })
})

describe('processing a recording with a chosen method (step 07)', () => {
  const METHODS: ConfigResponse = {
    ...CONFIG,
    extraction_methods: [
      { name: 'neural_tse', display_name: 'Neural', available: true, reason: null },
      { name: 'mask_baseline', display_name: 'Baseline', available: true, reason: null },
    ],
    default_method: 'neural_tse',
    default_attenuation_db: 15,
  }

  function mockApi() {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/recordings') return json([RECORDING])
      if (url === '/api/recordings/3/runs') return json([])
      if (url === '/api/recordings/3/process' && init?.method === 'POST') {
        const body = JSON.parse(String(init.body)) as { method?: string }
        const n = fetchMock.mock.calls.filter(([u]) => u === url).length
        return json({ ...RUN, run_id: `run_${n}`, method: body.method ?? 'neural_tse' }, 202)
      }
      return json({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }

  function bodies(fetchMock: ReturnType<typeof mockApi>) {
    return fetchMock.mock.calls
      .filter(([url]) => url === '/api/recordings/3/process')
      .map((call) => JSON.parse(String((call as unknown as [string, RequestInit])[1].body)))
  }

  it('follows the session method and can be set per run', async () => {
    const fetchMock = mockApi()
    useAppStore.setState({
      ws: 'connected', config: METHODS, selected: ['speech'], attenuation: 15,
      status: { ...RUNNING, extraction_method: 'mask_baseline' },
    })
    render(<RecordingsList />)
    const button = await screen.findByRole('button', { name: 'Process speech_1.wav' })
    const select = screen.getByLabelText('Processing method')
    expect(select).toHaveValue('mask_baseline') // the session's method
    expect(button).toHaveAttribute(
      'title', 'Keep the selected sounds, turn the rest down by 15 dB (Baseline)',
    )
    fireEvent.click(button)
    await waitFor(() => expect(screen.getByTestId('run-run_1')).toBeInTheDocument())
    fireEvent.change(select, { target: { value: 'neural_tse' } })
    fireEvent.click(button)
    await waitFor(() => expect(screen.getByTestId('run-run_2')).toBeInTheDocument())
    expect(bodies(fetchMock)).toEqual([
      { labels: ['speech'], attenuation_db: 15, method: 'mask_baseline' },
      { labels: ['speech'], attenuation_db: 15, method: 'neural_tse' },
    ])
    // both runs are listed with their method, ready to be compared
    expect(screen.getByTestId('run-run_1')).toHaveTextContent('Baseline')
    expect(screen.getByTestId('run-run_2')).toHaveTextContent('Neural')
  })

  it('skips a method that is not available', async () => {
    const fetchMock = mockApi()
    useAppStore.setState({
      ws: 'connected', selected: ['speech'],
      status: { ...RUNNING, extraction_method: 'neural_tse' },
      config: {
        ...METHODS,
        default_method: 'mask_baseline',
        extraction_methods: [
          { name: 'neural_tse', display_name: 'Neural', available: false, reason: 'not trained' },
          { name: 'mask_baseline', display_name: 'Baseline', available: true, reason: null },
        ],
      },
    })
    render(<RecordingsList />)
    const button = await screen.findByRole('button', { name: 'Process speech_1.wav' })
    expect(screen.getByLabelText('Processing method')).toHaveValue('mask_baseline')
    fireEvent.click(button)
    await waitFor(() => expect(bodies(fetchMock)).toHaveLength(1))
    expect(bodies(fetchMock)[0].method).toBe('mask_baseline')
  })
})

describe('processing a recording', () => {
  function mockApi(runs: Run[] = []) {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/recordings') return json([RECORDING])
      if (url === '/api/recordings/3/runs') return json(runs)
      if (url === '/api/recordings/3/process' && init?.method === 'POST') return json(RUN, 202)
      return json({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }

  it('needs a selection, then sends the selected classes and attenuation', async () => {
    const fetchMock = mockApi()
    useAppStore.setState({ ws: 'connected' })
    render(<RecordingsList />)
    const button = await screen.findByRole('button', { name: 'Process speech_1.wav' })
    expect(button).toBeDisabled() // nothing selected yet
    expect(button).toHaveAttribute('title', 'Select one or more sounds above first')
    act(() => useAppStore.setState({ selected: ['speech'], attenuation: 24 }))
    expect(button).toBeEnabled()
    fireEvent.click(button)
    await waitFor(() => expect(screen.getByTestId(`run-status-${RUN.run_id}`)).toBeInTheDocument())
    const call = fetchMock.mock.calls.find(([url]) => url === '/api/recordings/3/process')
    expect(JSON.parse(String((call as unknown as [string, RequestInit])[1].body))).toEqual({
      labels: ['speech'], attenuation_db: 24,
    })
    expect(screen.getByTestId(`run-status-${RUN.run_id}`)).toHaveTextContent('waiting…')
    expect(screen.getByTestId(`run-${RUN.run_id}`)).toHaveTextContent('Speech')
    expect(screen.getByTestId(`run-${RUN.run_id}`)).toHaveTextContent('rest −24 dB')
    expect(screen.getByTestId(`run-${RUN.run_id}`)).toHaveTextContent('Baseline')
    expect(screen.queryByLabelText('Processing method')).toBeNull() // no methods from the backend

    // state changes arrive over the WebSocket: no polling
    act(() => handleServerMessage({ type: 'run', t: 0, payload: { ...RUN, status: 'running' } }))
    expect(screen.getByTestId(`run-status-${RUN.run_id}`)).toHaveTextContent('processing…')
    expect(screen.queryByTestId(`ab-player-${RUN.run_id}`)).toBeNull()
    act(() =>
      handleServerMessage({
        type: 'run', t: 0,
        payload: { ...RUN, status: 'done', duration_s: 5, processing_seconds: 0.42,
          output_path: 'outputs/processed/run_20261116-101500-123.wav' },
      }),
    )
    expect(screen.getByTestId(`run-status-${RUN.run_id}`)).toHaveTextContent('done')
    expect(screen.getByTestId(`ab-player-${RUN.run_id}`)).toBeInTheDocument()
    expect(screen.getByTestId(`ab-processed-${RUN.run_id}`)).toHaveAttribute(
      'src', `/api/runs/${RUN.run_id}/audio`,
    )
    expect(screen.getByTestId(`ab-original-${RUN.run_id}`)).toHaveAttribute(
      'src', '/api/recordings/3/audio',
    )
    expect(screen.getByTestId(`run-${RUN.run_id}`)).toHaveTextContent('0.4 s to process')
    expect(useAppStore.getState().runs[3]).toHaveLength(1) // updated in place, not duplicated
  })

  it('lists earlier runs and shows a failure with its reason', async () => {
    mockApi([{ ...RUN, status: 'failed', error: 'the recording file is gone: data/raw/x.wav' }])
    render(<RecordingsList />)
    await waitFor(() => expect(screen.getByTestId(`run-status-${RUN.run_id}`)).toHaveTextContent('failed'))
    expect(screen.getByText(/the recording file is gone/)).toBeInTheDocument()
    act(() =>
      handleServerMessage({
        type: 'run', t: 0, payload: { ...RUN, run_id: 'run_other', status: 'failed', error: 'boom' },
      }),
    )
    expect(useAppStore.getState().toasts.at(-1)?.message).toBe('Processing failed: boom')
    expect(useAppStore.getState().runs[3]?.map((r) => r.run_id)).toEqual(['run_other', RUN.run_id])
  })

  it('shows a recording whose file is gone, without play or process', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url === '/api/recordings') return json([{ ...RECORDING, missing: true }])
      if (url === '/api/recordings/3/runs') {
        return json([{ ...RUN, status: 'done', duration_s: 5 }])
      }
      return json({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({ selected: ['speech'] })
    render(<RecordingsList />)
    expect(await screen.findByText('file missing')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Play speech_1.wav' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Process speech_1.wav' })).toBeDisabled()
    // its run is still there and its processed audio still plays
    await waitFor(() => expect(screen.getByTestId(`ab-player-${RUN.run_id}`)).toBeInTheDocument())
    expect(screen.queryByTestId(`ab-original-${RUN.run_id}`)).toBeNull()
  })
})

describe('the neural output gate (step 07b)', () => {
  const METHODS: ConfigResponse = {
    ...CONFIG,
    extraction_methods: [
      { name: 'neural_tse', display_name: 'Neural', available: true, reason: null },
      { name: 'mask_baseline', display_name: 'Baseline', available: true, reason: null },
    ],
    default_method: 'neural_tse',
    default_attenuation_db: 15,
    default_gate: true,
  }
  const IDLE: EngineStatus = {
    state: 'idle', mode: null, device: null, sample_rate: null, channels: null, message: '',
    extraction_method: 'neural_tse', extraction_gate: true,
  }

  it('is switched while stopped and sent to the backend', async () => {
    const fetchMock = vi.fn(async () => json({ ...IDLE, extraction_gate: false }))
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({ config: METHODS, status: IDLE })
    render(<ExtractionControls />)
    const button = screen.getByLabelText('Output gate')
    expect(button).toHaveTextContent('Gate on')
    expect(button).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(button)
    await waitFor(() => expect(button).toHaveTextContent('Gate off'))
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/session/gate')
    expect(init.method).toBe('PUT')
    expect(JSON.parse(String(init.body))).toEqual({ enabled: false })
  })

  it('cannot be switched during a session or for the baseline', () => {
    useAppStore.setState({ config: METHODS, status: { ...RUNNING, extraction_method: 'neural_tse' } })
    const { unmount } = render(<ExtractionControls />)
    expect(screen.getByLabelText('Output gate')).toBeDisabled()
    unmount()
    useAppStore.setState({ status: { ...IDLE, extraction_method: 'mask_baseline' } })
    render(<ExtractionControls />)
    expect(screen.getByLabelText('Output gate')).toBeDisabled()
    expect(screen.getByLabelText('Output gate')).toHaveAttribute(
      'title', 'The gate belongs to the Neural method',
    )
  })

  it('the session bar says whether the gate is on', () => {
    useAppStore.setState({
      status: {
        ...RUNNING,
        extraction_method: 'neural_tse',
        extraction: {
          method: 'neural_tse', latency_ms: 10.6, gating: false, target_active: true,
          device: 'cuda', model: 'tse_waveformer_v2', model_ms: 6.1, gate: true, gate_closed: 0.4,
        },
      },
    })
    render(<SessionBar />)
    expect(screen.getByTestId('session-stats')).toHaveTextContent('Neural · cuda · gate · +10.6 ms')
  })

  it('applies to neural processing runs too, which show it', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/recordings') return json([RECORDING])
      if (url === '/api/recordings/3/runs') return json([])
      if (url === '/api/recordings/3/process' && init?.method === 'POST') {
        return json({ ...RUN, run_id: 'run_g', method: 'neural_tse', status: 'done',
                      info: { gate: false } }, 202)
      }
      return json({}, 404)
    })
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({
      ws: 'connected', config: METHODS, selected: ['speech'], attenuation: 40,
      status: { ...IDLE, extraction_gate: false },
    })
    render(<RecordingsList />)
    fireEvent.click(await screen.findByRole('button', { name: 'Process speech_1.wav' }))
    await waitFor(() => expect(screen.getByTestId('run-run_g')).toBeInTheDocument())
    const call = fetchMock.mock.calls.find(([u]) => u === '/api/recordings/3/process')
    const body = JSON.parse(String((call as unknown as [string, RequestInit])[1].body))
    expect(body).toEqual({ labels: ['speech'], attenuation_db: 40, method: 'neural_tse',
                           gate: false })
    expect(screen.getByTestId('run-run_g')).toHaveTextContent('Neural · no gate')
  })
})
