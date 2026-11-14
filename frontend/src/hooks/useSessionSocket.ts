// Subscribes to /ws/session for the lifetime of the component and feeds the store.
import { useEffect } from 'react'
import { SessionSocket, sessionSocketUrl } from '../api/ws.ts'
import { useAppStore } from '../store/appStore.ts'

export function useSessionSocket(url: string = sessionSocketUrl()): void {
  const setWsState = useAppStore((s) => s.setWsState)
  const setStatus = useAppStore((s) => s.setStatus)

  useEffect(() => {
    const socket = new SessionSocket(url, {
      onStateChange: setWsState,
      onMessage: (message) => {
        switch (message.type) {
          case 'status':
            setStatus(message.payload)
            break
          case 'error':
            console.warn('[ws] server error:', message.payload.message)
            break
          default:
            // levels / spectrogram / detections / sources are consumed from step 03 on.
            break
        }
      },
    })
    socket.connect()
    return () => socket.close()
  }, [url, setWsState, setStatus])
}
