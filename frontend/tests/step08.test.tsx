// Step 08: keyboard shortcuts (Space = Bypass, M = Mute, not while a control has the focus),
// settings remembered in localStorage (also when storage is unavailable), the warming / error
// states of the session bar with Restart, the route / gate / delay summary (measured vs
// estimated), the method names with their explanation, the remembered spectrogram view.
import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ConfigResponse, EngineStatus } from '../src/api/types.ts'
import { ExtractionControls } from '../src/components/session/ExtractionControls.tsx'
import { SessionBar } from '../src/components/session/SessionBar.tsx'
import { latencyLabel } from '../src/api/target.ts'
import { SpectrogramPanel } from '../src/components/viz/SpectrogramPanel.tsx'
import { useKeyboardShortcuts } from '../src/hooks/useKeyboardShortcuts.ts'
import { usePersistedSettings } from '../src/hooks/usePersistedSettings.ts'
import { SETTINGS_KEY, loadSettings, saveSettings } from '../src/store/persist.ts'
import { initialAppState, useAppStore } from '../src/store/appStore.ts'

const RUNNING: EngineStatus = {
  state: 'running',
  mode: 'semantic',
  device: 'Microphone Array (Realtek(R) Audio)',
  sample_rate: 48000,
  channels: 2,
  message: '',
  monitoring: true,
  muted: false,
  gain_db: 0,
  bypass: false,
  latency_ms: 124,
  extraction: {
    method: 'hybrid',
    latency_ms: 39.98,
    gating: true,
    target_active: true,
    route: 'speech_enhancer',
    gate_state: 'closed',
  },
}

const CONFIG: ConfigResponse = {
  classes: [
    { name: 'speech', display_name: 'Speech', description: '' },
    { name: 'music', display_name: 'Music', description: '' },
  ],
  modes: [],
  device_profile: 'laptop',
  extraction_methods: [
    { name: 'hybrid', display_name: 'Hybrid', available: true, reason: null },
    { name: 'neural_tse', display_name: 'Neural', available: true, reason: null },
    { name: 'mask_baseline', display_name: 'Baseline', available: true, reason: null },
  ],
  default_method: 'hybrid',
  default_attenuation_db: 15,
}

type Call = { url: string; method: string; body: unknown }

function mockFetch(reply: (call: Call) => unknown = () => RUNNING): Call[] {
  const calls: Call[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const call = {
        url,
        method: init?.method ?? 'GET',
        body: init?.body ? JSON.parse(String(init.body)) : null,
      }
      calls.push(call)
      return new Response(JSON.stringify(reply(call)), { status: 200 })
    }),
  )
  return calls
}

beforeEach(() => {
  useAppStore.setState({ ...initialAppState, backend: 'up' })
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

function ShortcutHost() {
  useKeyboardShortcuts(true)
  return <input aria-label="A text field" />
}

describe('keyboard shortcuts', () => {
  it('Space toggles Bypass and M toggles Mute, unless a control has the focus', async () => {
    useAppStore.setState({ status: RUNNING })
    const calls = mockFetch((call) => ({ ...RUNNING, bypass: call.url.endsWith('/bypass') }))
    render(<ShortcutHost />)
    await act(async () => {
      fireEvent.keyDown(window, { key: ' ', code: 'Space' })
    })
    expect(calls.at(-1)).toMatchObject({ url: '/api/session/bypass', method: 'PUT',
                                         body: { enabled: true } })
    await act(async () => {
      fireEvent.keyDown(window, { key: 'm' })
    })
    expect(calls.at(-1)).toMatchObject({ url: '/api/session/monitor',
                                         body: { enabled: true, gain_db: 0, muted: true } })
    const count = calls.length
    const field = screen.getByLabelText('A text field')
    field.focus()
    await act(async () => {
      fireEvent.keyDown(field, { key: ' ', code: 'Space' })
      fireEvent.keyDown(field, { key: 'm' })
    })
    expect(calls.length).toBe(count) // typing in a field is not a shortcut
  })
})

describe('remembered settings', () => {
  it('round-trips and survives unavailable storage', () => {
    saveSettings({ selected: ['speech'], attenuation: 30, spectrogram: 'output' })
    expect(loadSettings()).toEqual({ selected: ['speech'], attenuation: 30, spectrogram: 'output' })
    window.localStorage.setItem(SETTINGS_KEY, '{"selected": [1], "attenuation": "x"}')
    expect(loadSettings()).toEqual({}) // invalid values are dropped
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    expect(loadSettings()).toEqual({})
    expect(() => saveSettings({ attenuation: 10 })).not.toThrow()
  })

  it('restores the selection and the attenuation once the classes are known', async () => {
    saveSettings({ selected: ['music', 'gone_class'], attenuation: 32 })
    const calls = mockFetch((call) => call.body)
    function Host() {
      usePersistedSettings()
      return null
    }
    render(<Host />)
    await act(async () => {
      useAppStore.getState().setConfig(CONFIG)
    })
    const state = useAppStore.getState()
    expect(state.selected).toEqual(['music']) // unknown classes are dropped
    expect(state.attenuation).toBe(32)
    expect(calls.find((c) => c.url === '/api/session/target')?.body).toMatchObject({
      labels: ['music'],
      attenuation_db: 32,
    })
    act(() => useAppStore.getState().setSelected(['speech']))
    expect(loadSettings().selected).toEqual(['speech']) // changes are remembered
  })

  it('remembers the spectrogram view', () => {
    render(<SpectrogramPanel />)
    fireEvent.click(screen.getByRole('button', { name: 'Output' }))
    expect(loadSettings().spectrogram).toBe('output')
  })
})

describe('session bar states', () => {
  it('shows warming with what is being prepared, and Stop already works', () => {
    useAppStore.setState({
      status: { ...RUNNING, state: 'warming', message: 'Preparing models: loading the speech enhancer (2/3)' },
    })
    render(<SessionBar />)
    expect(screen.getByTestId('session-state').textContent).toContain('Preparing models')
    expect(screen.getByTestId('session-state').textContent).toContain('speech enhancer')
    expect(screen.getByRole('button', { name: 'Stop' })).toBeTruthy()
  })

  it('shows an error with Restart', async () => {
    useAppStore.setState({
      status: { ...RUNNING, state: 'error', message: "OutputLostError: the output device stopped playing" },
    })
    const calls = mockFetch(() => ({ ...RUNNING, state: 'warming' }))
    render(<SessionBar />)
    expect(screen.getByTestId('session-state').textContent).toContain('stopped playing')
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Restart' }))
    })
    expect(calls.at(-1)).toMatchObject({ url: '/api/session/start', method: 'POST' })
  })

  it('summarises route, gate (in words) and the delay, measured or estimated', () => {
    useAppStore.setState({ status: RUNNING })
    const { rerender } = render(<SessionBar />)
    const summary = screen.getByTestId('session-summary').textContent ?? ''
    expect(summary).toContain('speech enhancer')
    expect(screen.getByTestId('gate-indicator').textContent).toBe('Gate closed')
    expect(screen.getByTestId('latency-label').textContent).toBe('Delay 124 ms estimated')
    act(() =>
      useAppStore.getState().setMeasuredLatency({
        time: '2026-11-20T10:00:00',
        modes: { speech: { title: 'Hybrid, speech route', median_ms: 96.4, min_ms: 95,
                           max_ms: 98, found: 10 } },
      }),
    )
    rerender(<SessionBar />)
    expect(screen.getByTestId('latency-label').textContent).toBe('Delay 96 ms measured')
    expect(latencyLabel({ ...RUNNING, extraction: { ...RUNNING.extraction!, route: 'waveformer' } },
                        useAppStore.getState().measuredLatency)?.measured).toBe(false)
  })
})

describe('method names', () => {
  it('recommends Hybrid and explains the chosen method in one line', () => {
    useAppStore.setState({ config: CONFIG, status: { ...RUNNING, state: 'idle',
                                                     extraction_method: 'hybrid' } })
    render(<ExtractionControls />)
    const select = screen.getByLabelText('Extraction method') as HTMLSelectElement
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      'Hybrid (recommended)',
      'Neural',
      'Baseline',
    ])
    expect(screen.getByTestId('method-help').textContent).toContain('speech enhancer')
  })
})
