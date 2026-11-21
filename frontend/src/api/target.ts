// Building TargetSpec messages for the class chips and the attenuation slider.
import type { EngineStatus, ExtractionMethod, MeasuredLatency, TargetSpec } from './types.ts'

/** Where the slider starts until /api/config says otherwise (the backend's live default:
 *  15 dB keeps some of the surroundings and hides the extractor's mistakes better than 30). */
export const DEFAULT_ATTENUATION_DB = 15
export const MAX_ATTENUATION_DB = 40

/** Names for the extraction methods when the backend did not send any. */
export const METHOD_NAMES: Record<string, string> = {
  hybrid: 'Hybrid',
  neural_tse: 'Neural',
  mask_baseline: 'Baseline',
  passthrough: 'Off',
}

export function methodName(method: string | null | undefined): string {
  return method ? (METHOD_NAMES[method] ?? method) : '–'
}

export function isExtractionMethod(value: unknown): value is ExtractionMethod {
  return value === 'hybrid' || value === 'neural_tse' || value === 'mask_baseline'
}

/** Methods with a gate the page's "Gate" switch controls: the hybrid gate (step 07c) and the
 *  neural output gate (step 07b). */
export function hasGate(method: string | null | undefined): boolean {
  return method === 'hybrid' || method === 'neural_tse'
}

/** One-line explanations shown under the method selector (step 08). */
export const METHOD_INFO: Record<string, { label: string; help: string }> = {
  hybrid: {
    label: 'Hybrid (recommended)',
    help:
      'A speech enhancer for Speech, a sound-separation network for everything else; ' +
      'pauses go silent.',
  },
  neural_tse: {
    label: 'Neural',
    help: 'The sound-separation network (Waveformer) for every selection.',
  },
  mask_baseline: {
    label: 'Baseline',
    help:
      'The detector switches a noise filter on and off: simple, but sounds that overlap stay ' +
      'mixed.',
  },
}

/** Which measured mode (soundspy measure-latency) belongs to a hybrid route. */
export function measuredModeFor(method: string | null | undefined, route: string | null | undefined):
  string | null {
  if (method !== 'hybrid') return null
  if (route === 'speech_enhancer' || route === 'cascade') return 'speech'
  if (route === 'waveformer') return 'waveformer'
  if (route === 'off') return 'passthrough'
  return null
}

/** Names for the hybrid extractor's routes (step 07c). */
export const ROUTE_NAMES: Record<string, string> = {
  speech_enhancer: 'speech enhancer',
  waveformer: 'Waveformer',
  cascade: 'Waveformer + speech enhancer',
  off: 'nothing selected',
}

/** The Semantic tab's target: the selected class IDs and how much everything else is turned
 *  down; no direction. */
export function targetFor(
  labels: string[],
  attenuationDb: number = DEFAULT_ATTENUATION_DB,
): TargetSpec {
  return { labels, direction: null, source_id: null, attenuation_db: attenuationDb }
}

/** The session bar's one-line description of the running extractor: method, route (hybrid),
 *  device, gate, its own delay. */
export function extractionSummary(
  extraction: {
    method: string
    latency_ms: number
    gating: boolean
    device?: string | null
    gate?: boolean | null
    route?: string | null
    gate_state?: string | null
  },
  bypass = false,
): string {
  const parts = [methodName(extraction.method)]
  if (extraction.method === 'hybrid') {
    if (extraction.route) parts.push(ROUTE_NAMES[extraction.route] ?? extraction.route)
    if (extraction.device) parts.push(extraction.device)
    parts.push(
      extraction.gate === false || extraction.gate_state === 'off'
        ? 'no gate'
        : extraction.gate_state === 'closed'
          ? 'gate closed'
          : 'gate open',
    )
    if (!extraction.gating) parts.push('no detector')
  } else {
    if (extraction.device) parts.push(extraction.device)
    if (extraction.method === 'neural_tse' && extraction.gate != null) {
      parts.push(extraction.gate ? 'gate' : 'no gate')
    }
  }
  parts.push(`+${extraction.latency_ms} ms`)
  if (extraction.method === 'mask_baseline' && !extraction.gating) parts.push('no gating')
  if (bypass) parts.push('bypassed')
  return parts.join(' · ')
}

/** "96 ms measured" when the latest measurement covers this route, else the estimate. */
export function latencyLabel(
  status: EngineStatus | null,
  measured: MeasuredLatency | null,
): { text: string; title: string; measured: boolean } | null {
  if (status?.latency_ms == null) return null
  const mode = measuredModeFor(status.extraction?.method, status.extraction?.route)
  const m = mode ? measured?.modes[mode] : undefined
  if (m?.median_ms != null && m.found > 0) {
    return {
      text: `${Math.round(m.median_ms)} ms measured`,
      title:
        `Measured with soundspy measure-latency${measured?.time ? ` on ${measured.time}` : ''}` +
        `${measured?.output ? ` through ${measured.output}` : ''}` +
        ` (${m.min_ms}-${m.max_ms} ms over ${m.found} clicks); the session's own estimate is ` +
        `${status.latency_ms} ms.`,
      measured: true,
    }
  }
  return {
    text: `${Math.round(status.latency_ms)} ms estimated`,
    title:
      "The session's own estimate: input + chunk + processing + output queue + output. " +
      'Run `soundspy measure-latency --all` (laptop speakers, headphones off) to measure it.',
    measured: false,
  }
}
