// Subscribes to /ws/session for the lifetime of the component and routes every message:
// status -> store; levels/waveform/spectrogram -> liveFrames (no React re-render);
// warning/error -> toasts. Also exposes sendTarget() for the class chips (wired in step 05).
import { useEffect } from 'react'
import type { ServerMessage, TargetSpec } from '../api/types.ts'
import { SessionSocket, sessionSocketUrl } from '../api/ws.ts'
import { liveFrames } from '../components/viz/liveFrames.ts'
import { useAppStore } from '../store/appStore.ts'

let activeSocket: SessionSocket | null = null

/** Send a set_target message over the live channel; false if it is not connected. */
export function sendTarget(target: TargetSpec): boolean {
  return activeSocket?.send({ type: 'set_target', t: 0, payload: target }) ?? false
}

export function handleServerMessage(message: ServerMessage): void {
  const store = useAppStore.getState()
  switch (message.type) {
    case 'status':
      store.setStatus(message.payload)
      if (message.payload.state === 'error' && message.payload.message) {
        store.pushToast('error', `Session stopped: ${message.payload.message}`)
      }
      break
    case 'levels':
      liveFrames.pushLevels(message.payload, message.t)
      break
    case 'waveform':
      liveFrames.pushWaveform(message.payload, message.t)
      break
    case 'spectrogram':
      liveFrames.pushSpectrogram(message.payload, message.t)
      break
    case 'warning':
      store.pushToast('warning', message.payload.message)
      break
    case 'error':
      store.pushToast('error', message.payload.message)
      break
    default:
      // detections / sources arrive from step 05 / module 5 on.
      break
  }
}

export function useSessionSocket(url: string = sessionSocketUrl()): void {
  const setWsState = useAppStore((s) => s.setWsState)

  useEffect(() => {
    const socket = new SessionSocket(url, {
      onStateChange: setWsState,
      onMessage: handleServerMessage,
    })
    activeSocket = socket
    socket.connect()
    return () => {
      if (activeSocket === socket) activeSocket = null
      socket.close()
    }
  }, [url, setWsState])
}
