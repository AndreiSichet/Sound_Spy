// Keyboard shortcuts of the Semantic tab (step 08):
//   Space = Bypass on / off (a toggle: press once to hear the unprocessed sound, once more to go
//           back; a toggle rather than "while held" so a long comparison needs no finger on
//           the key)
//   M     = Mute on / off
// Ignored while a text field, a selector, a slider or a button has the focus: there the key
// keeps its own meaning (Space presses a focused button, types in a field).
import { useEffect } from 'react'
import { api } from '../api/client.ts'
import { useAppStore } from '../store/appStore.ts'

export const SHORTCUTS = [
  { key: 'Space', action: 'Bypass on / off (hear the unprocessed sound)' },
  { key: 'M', action: 'Mute on / off' },
] as const

function ownsTheKey(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  return (
    tag === 'INPUT' ||
    tag === 'SELECT' ||
    tag === 'TEXTAREA' ||
    tag === 'BUTTON' ||
    tag === 'SUMMARY' ||
    target.isContentEditable
  )
}

export function toggleBypass(): Promise<void> {
  const { status, setStatus, pushToast } = useAppStore.getState()
  return api
    .setBypass(!(status?.bypass ?? false))
    .then(setStatus)
    .catch((error: unknown) => pushToast('error', String(error)))
}

export function toggleMute(): Promise<void> {
  const { status, setStatus, pushToast } = useAppStore.getState()
  return api
    .setMonitor({
      enabled: status?.monitoring ?? false,
      gain_db: status?.gain_db ?? 0,
      muted: !(status?.muted ?? false),
    })
    .then(setStatus)
    .catch((error: unknown) => pushToast('error', String(error)))
}

export function useKeyboardShortcuts(enabled = true): void {
  useEffect(() => {
    if (!enabled) return
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.altKey || event.metaKey || event.repeat) return
      if (ownsTheKey(event.target)) return
      if (event.key === ' ' || event.code === 'Space') {
        event.preventDefault() // no page scroll
        void toggleBypass()
      } else if (event.key === 'm' || event.key === 'M') {
        void toggleMute()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enabled])
}
