import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ConfigResponse, EngineStatus, Recording } from '../src/api/types.ts'
import { ClassChips } from '../src/components/detection/ClassChips.tsx'
import { DetectionTimeline } from '../src/components/detection/DetectionTimeline.tsx'
import { RecordingsList } from '../src/components/session/RecordingsList.tsx'
import { liveFrames } from '../src/components/viz/liveFrames.ts'
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
    { name: 'dishes', display_name: 'Dishes & cutlery', description: 'Plates' },
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
  detector: { model: 'det_mn10_v1', device: 'cuda', hop_s: 0.25, latency_ms: 6.1, skipped: 0 },
}

beforeEach(() => {
  useAppStore.setState({ ...initialAppState, backend: 'up', config: CONFIG, status: RUNNING })
  liveFrames.reset()
  vi.mocked(sendTarget).mockClear()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('class chips', () => {
  it('show the live probability and highlight active classes', () => {
    handleServerMessage({
      type: 'detections',
      t: 4.25,
      payload: { probabilities: { speech: 0.92, music: 0.1, dishes: 0.4 }, active: ['speech'] },
    })
    render(<ClassChips />)
    expect(screen.getByTestId('chip-speech')).toHaveTextContent('92%')
    expect(screen.getByTestId('chip-speech')).toHaveAttribute('data-active', 'true')
    expect(screen.getByTestId('chip-music')).toHaveAttribute('data-active', 'false')
    expect(screen.getByTestId('bar-dishes')).toHaveStyle({ width: '40%' })
    expect(liveFrames.detections).toHaveLength(1)
    expect(useAppStore.getState().detection?.t).toBe(4.25)
  })

  it('are selectable and send set_target in class order', () => {
    render(<ClassChips />)
    fireEvent.click(screen.getByTestId('chip-dishes'))
    fireEvent.click(screen.getByTestId('chip-speech'))
    expect(screen.getByTestId('chip-speech')).toHaveAttribute('aria-pressed', 'true')
    expect(useAppStore.getState().selected).toEqual(['speech', 'dishes'])
    const last = vi.mocked(sendTarget).mock.calls.at(-1)?.[0]
    expect(last).toEqual({ labels: ['speech', 'dishes'], direction: null, source_id: null,
      attenuation_db: 15 })
    fireEvent.click(screen.getByTestId('chip-speech'))
    expect(vi.mocked(sendTarget).mock.calls.at(-1)?.[0].labels).toEqual(['dishes'])
    expect(screen.getByTestId('selection-hint')).toHaveTextContent(
      'Keeping 1 sound type(s); everything else is turned down by 15 dB.',
    )
  })

  it('say when detection is off for the session', () => {
    useAppStore.setState({ status: { ...RUNNING, detector: null } })
    render(<ClassChips />)
    expect(screen.getByText(/Detection is off for this session/)).toBeInTheDocument()
    expect(screen.getByTestId('chip-speech')).not.toHaveTextContent('%')
  })
})

describe('detection timeline', () => {
  it('mounts as a canvas and keeps 30 s of history', () => {
    render(<DetectionTimeline />)
    expect(screen.getByTestId('detection-timeline').tagName).toBe('CANVAS')
    for (let i = 0; i < 200; i++) {
      liveFrames.pushDetections({ probabilities: { speech: 0.5 }, active: [] }, i * 0.25)
    }
    const ts = liveFrames.detections.map((d) => d.t)
    expect(ts[ts.length - 1]).toBe(49.75)
    expect(ts[0]).toBeGreaterThanOrEqual(49.75 - 30)
  })
})

describe('offline detection of a recording', () => {
  it('shows the timeline under the recording after Detect', async () => {
    const recording: Recording = {
      id: 3, created_at: '2026-11-14T14:32:56.131Z', path: 'data/raw/speech_1.wav',
      duration_s: 5, sample_rate: 48000, channels: 2, device: 'mic', profile: 'laptop',
      label: 'speech', notes: '',
    }
    const timeline = {
      duration_s: 5, hop_s: 0.25, window_s: 1, labels: ['speech', 'music', 'dishes'],
      times: [1, 1.25], probabilities: { speech: [0.9, 0.9], music: [0, 0], dishes: [0, 0] },
      segments: [{ label: 'speech', probability: 0.86, t_start: 0.5, t_end: 5 }],
    }
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/recordings') return new Response(JSON.stringify([recording]))
      if (url === '/api/recordings/3/detect' && init?.method === 'POST') {
        return new Response(JSON.stringify(timeline))
      }
      return new Response('{}', { status: 404 })
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<RecordingsList />)
    fireEvent.click(await screen.findByRole('button', { name: /Detect sounds in speech_1.wav/ }))
    await waitFor(() => expect(screen.getByTestId('offline-timeline')).toBeInTheDocument())
    const segment = screen.getByTestId('segment-speech')
    expect(segment).toHaveStyle({ left: '10%', width: '90%' })
    expect(screen.queryByTestId('segment-music')).toBeNull()
    expect(screen.getByText('1 segment(s) over 5.0 s.')).toBeInTheDocument()
  })
})
