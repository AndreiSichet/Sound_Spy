import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ConfigResponse, EngineStatus, Recording, Run } from '../src/api/types.ts'
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
    expect(slider).toHaveValue('30')
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
    expect(stats).toHaveTextContent('mask_baseline · +21.33 ms')
    expect(stats).toHaveTextContent('105.3 ms')
    expect(stats).not.toHaveTextContent('no gating')
    act(() => useAppStore.setState({ status: { ...RUNNING, bypass: true } }))
    expect(stats).toHaveTextContent('bypassed')
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
    expect(liveFrames.waveform).toHaveLength(0)
    expect(liveFrames.waveformOut).toHaveLength(1)
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
