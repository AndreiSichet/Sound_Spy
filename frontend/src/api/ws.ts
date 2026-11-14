// Reconnecting WebSocket client for /ws/session with typed message parsing.
import { SERVER_MESSAGE_TYPES, type ClientMessage, type ServerMessage } from './types.ts'

export type SocketState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected'

export interface SessionSocketOptions {
  onMessage: (message: ServerMessage) => void
  onStateChange?: (state: SocketState) => void
  initialDelayMs?: number
  maxDelayMs?: number
}

/** Parse one raw frame; returns null (with a console warning) for anything unexpected. */
export function parseServerMessage(raw: unknown): ServerMessage | null {
  if (typeof raw !== 'string') {
    console.warn('[ws] ignoring non-text frame')
    return null
  }
  let data: unknown
  try {
    data = JSON.parse(raw)
  } catch {
    console.warn('[ws] ignoring invalid JSON', raw)
    return null
  }
  if (typeof data !== 'object' || data === null) {
    console.warn('[ws] ignoring non-object message', data)
    return null
  }
  const { type, t, payload } = data as Record<string, unknown>
  if (!SERVER_MESSAGE_TYPES.includes(type as ServerMessage['type'])) {
    console.warn('[ws] ignoring unknown message type', type)
    return null
  }
  if (typeof t !== 'number' || typeof payload !== 'object' || payload === null) {
    console.warn('[ws] ignoring malformed message', data)
    return null
  }
  return data as ServerMessage
}

export function sessionSocketUrl(loc: Location = window.location): string {
  const scheme = loc.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${loc.host}/ws/session`
}

/** Connects, and reconnects with exponential backoff until close() is called. */
export class SessionSocket {
  private socket: WebSocket | null = null
  private timer: ReturnType<typeof setTimeout> | null = null
  private delayMs: number
  private closed = false
  private readonly url: string
  private readonly options: Required<SessionSocketOptions>

  constructor(url: string, options: SessionSocketOptions) {
    this.url = url
    this.options = {
      onStateChange: () => {},
      initialDelayMs: 500,
      maxDelayMs: 10_000,
      ...options,
    }
    this.delayMs = this.options.initialDelayMs
  }

  connect(): void {
    this.closed = false
    this.open('connecting')
  }

  send(message: ClientMessage): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) return false
    this.socket.send(JSON.stringify(message))
    return true
  }

  close(): void {
    this.closed = true
    if (this.timer !== null) clearTimeout(this.timer)
    this.timer = null
    const socket = this.socket
    this.socket = null
    socket?.close()
    this.options.onStateChange('disconnected')
  }

  private open(state: SocketState): void {
    this.options.onStateChange(state)
    const socket = new WebSocket(this.url)
    this.socket = socket
    socket.onopen = () => {
      if (this.socket !== socket) return
      this.delayMs = this.options.initialDelayMs
      this.options.onStateChange('connected')
    }
    socket.onmessage = (event: MessageEvent) => {
      if (this.socket !== socket) return
      const message = parseServerMessage(event.data)
      if (message) this.options.onMessage(message)
    }
    socket.onclose = () => {
      if (this.socket !== socket || this.closed) return
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect(): void {
    this.options.onStateChange('reconnecting')
    const delay = this.delayMs
    this.delayMs = Math.min(this.delayMs * 2, this.options.maxDelayMs)
    this.timer = setTimeout(() => {
      this.timer = null
      if (!this.closed) this.open('reconnecting')
    }, delay)
  }
}
