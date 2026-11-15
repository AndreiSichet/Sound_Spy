import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App.tsx'
import type { ConfigResponse } from '../src/api/types.ts'
import { parseServerMessage } from '../src/api/ws.ts'
import { initialAppState, useAppStore } from '../src/store/appStore.ts'

const CONFIG: ConfigResponse = {
  classes: [
    { name: 'speech', display_name: 'Speech', description: 'People talking' },
    { name: 'chair_scrape', display_name: 'Chair scraping', description: 'Chairs dragged' },
  ],
  modes: [
    { mode: 'semantic', available: true, reason: null },
    { mode: 'spatial', available: false, reason: 'requires a microphone array' },
    { mode: 'sources', available: false, reason: 'requires a microphone array' },
  ],
  device_profile: 'laptop',
}

/** Minimal WebSocket stand-in so tests never open real sockets. */
class FakeWebSocket {
  static readonly OPEN = 1
  static instances: FakeWebSocket[] = []
  readyState = 0
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  readonly url: string

  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }
  send() {}
  close() {
    this.readyState = 3
  }
  open() {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.()
  }
  receive(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }
  drop() {
    this.readyState = 3
    this.onclose?.()
  }
}

function mockFetch(healthy: boolean) {
  return vi.fn(async (input: RequestInfo | URL) => {
    if (!healthy) throw new TypeError('Failed to fetch')
    const url = String(input)
    const body = url.endsWith('/api/health') ? { status: 'ok', version: '0.1.0' } : CONFIG
    return new Response(JSON.stringify(body), { status: 200 })
  })
}

beforeEach(() => {
  useAppStore.setState(initialAppState)
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('App shell', () => {
  it('renders the three tabs and the connection indicator', async () => {
    vi.stubGlobal('fetch', mockFetch(true))
    render(<App />)
    const tabs = screen.getAllByRole('tab')
    expect(tabs.map((t) => t.textContent)).toEqual(['Semantic', 'Spatial', 'Sources'])
    expect(screen.getByRole('status', { name: 'Backend connection' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('Speech')).toBeInTheDocument())
  })

  it('lists classes as selectable chips and shows array requirement on other tabs', async () => {
    vi.stubGlobal('fetch', mockFetch(true))
    render(<App />)
    const chip = await screen.findByRole('button', { name: /Chair scraping/ })
    expect(chip).toBeEnabled() // selectable since step 05
    expect(chip).toHaveAttribute('aria-pressed', 'false')
    expect(chip).toHaveAttribute('title', 'Chairs dragged')
    expect(screen.queryByText('chair_scrape')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Spatial' }))
    expect(screen.getByText(/requires a microphone array/i)).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Spatial' })).toHaveAttribute('aria-selected', 'true')
  })

  it('turns green when backend and websocket are up, red when down', async () => {
    vi.stubGlobal('fetch', mockFetch(true))
    render(<App />)
    const indicator = screen.getByRole('status', { name: 'Backend connection' })
    act(() => FakeWebSocket.instances.at(-1)!.open())
    await waitFor(() => expect(indicator).toHaveAttribute('data-state', 'connected'))
    act(() => FakeWebSocket.instances.at(-1)!.receive({
      type: 'status', t: 0, payload: { state: 'idle', mode: null, device: null,
        sample_rate: null, channels: null, message: '' },
    }))
    expect(useAppStore.getState().status?.state).toBe('idle')
  })

  it('shows disconnected/reconnecting when the backend is down', async () => {
    vi.stubGlobal('fetch', mockFetch(false))
    render(<App />)
    act(() => FakeWebSocket.instances.at(-1)!.drop())
    const indicator = screen.getByRole('status', { name: 'Backend connection' })
    await waitFor(() =>
      expect(indicator).toHaveAttribute('data-state', 'disconnected · reconnecting'),
    )
  })
})

describe('parseServerMessage', () => {
  it('accepts known envelopes and ignores unknown types', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    expect(parseServerMessage('{"type":"error","t":0,"payload":{"message":"x"}}')?.type).toBe(
      'error',
    )
    expect(parseServerMessage('{"type":"bogus","t":0,"payload":{}}')).toBeNull()
    expect(parseServerMessage('not json')).toBeNull()
    expect(warn).toHaveBeenCalledTimes(2)
    warn.mockRestore()
  })
})
