// Where the sound is played (step 08): the output devices from /api/devices, the choice stored
// by the backend in data/user_settings.json (it survives a restart), and "Rescan devices" for
// headphones or a USB device plugged in after the backend started. Both only while no session
// runs (a session keeps the device it opened). The input stays the device profile's microphone.
import { useEffect, useState } from 'react'
import { ApiError, api } from '../../api/client.ts'
import type { DevicesResponse } from '../../api/types.ts'
import { useAppStore } from '../../store/appStore.ts'

function message(error: unknown): string {
  return error instanceof ApiError ? error.message : String(error)
}

export function OutputDeviceControls() {
  const status = useAppStore((s) => s.status)
  const backend = useAppStore((s) => s.backend)
  const pushToast = useAppStore((s) => s.pushToast)
  const [devices, setDevices] = useState<DevicesResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const active = status?.state === 'running' || status?.state === 'warming'

  useEffect(() => {
    if (backend !== 'up') return
    let cancelled = false
    api
      .devices()
      .then((d) => {
        if (!cancelled) setDevices(d)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [backend])

  const rescan = async () => {
    setBusy(true)
    try {
      setDevices(await api.rescanDevices())
      pushToast('info', 'Audio devices scanned again.')
    } catch (error) {
      pushToast('error', `Rescan failed: ${message(error)}`)
    } finally {
      setBusy(false)
    }
  }

  const choose = async (value: string) => {
    const device = value === 'default' ? null : Number(value)
    setBusy(true)
    try {
      await api.setOutputDevice(device)
      setDevices(await api.devices())
    } catch (error) {
      pushToast('error', `Could not change the output: ${message(error)}`)
    } finally {
      setBusy(false)
    }
  }

  // WASAPI twins first: the backend prefers that host API for the same physical device
  const outputs = (devices?.outputs ?? []).filter((d) => d.hostapi === 'Windows WASAPI')
  const list = outputs.length ? outputs : (devices?.outputs ?? [])
  const setting = devices?.output_setting
  const selectedIndex =
    setting == null ? 'default' : String(devices?.selected_output?.index ?? 'default')
  const title = active ? 'Stop the session to change the output or rescan' : undefined

  return (
    <div className="flex flex-wrap items-center gap-2 text-sm text-slate-300" title={title}>
      <label className="flex items-center gap-2">
        Output
        <select
          value={selectedIndex}
          onChange={(e) => void choose(e.target.value)}
          disabled={active || busy || devices === null}
          aria-label="Output device"
          className="max-w-56 rounded border border-slate-700 bg-slate-900 px-2 py-1 disabled:opacity-60"
        >
          <option value="default">
            System default{devices?.selected_output && setting == null
              ? ` (${devices.selected_output.name})`
              : ''}
          </option>
          {list.map((d) => (
            <option key={d.index} value={String(d.index)}>
              {d.name}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        onClick={() => void rescan()}
        disabled={active || busy || backend !== 'up'}
        className="rounded border border-slate-700 px-2 py-1 disabled:opacity-60"
      >
        Rescan devices
      </button>
      {devices?.error && (
        <span className="text-xs text-amber-300" role="note">
          Microphone: {devices.error}
        </span>
      )}
    </div>
  )
}
