// Dismissible warnings and errors (from the WebSocket and from failed requests).
import { useAppStore } from '../store/appStore.ts'

const STYLE = {
  error: 'border-rose-700 bg-rose-950 text-rose-100',
  warning: 'border-amber-700 bg-amber-950 text-amber-100',
  info: 'border-slate-700 bg-slate-900 text-slate-100',
} as const

export function Toasts() {
  const toasts = useAppStore((s) => s.toasts)
  const dismiss = useAppStore((s) => s.dismissToast)
  return (
    <div aria-live="polite" className="fixed bottom-4 right-4 z-30 flex w-80 flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.kind === 'error' ? 'alert' : 'status'}
          className={`flex items-start gap-2 rounded-md border px-3 py-2 text-sm ${STYLE[t.kind]}`}
        >
          <span className="flex-1">{t.message}</span>
          <button type="button" onClick={() => dismiss(t.id)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ))}
    </div>
  )
}
