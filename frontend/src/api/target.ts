// Building TargetSpec messages for the class chips and the attenuation slider.
import type { ExtractionMethod, TargetSpec } from './types.ts'

/** Where the slider starts until /api/config says otherwise (the backend's live default:
 *  15 dB keeps some of the surroundings and hides the extractor's mistakes better than 30). */
export const DEFAULT_ATTENUATION_DB = 15
export const MAX_ATTENUATION_DB = 40

/** Names for the extraction methods when the backend did not send any. */
export const METHOD_NAMES: Record<string, string> = {
  neural_tse: 'Neural',
  mask_baseline: 'Baseline',
  passthrough: 'Off',
}

export function methodName(method: string | null | undefined): string {
  return method ? (METHOD_NAMES[method] ?? method) : '–'
}

export function isExtractionMethod(value: unknown): value is ExtractionMethod {
  return value === 'neural_tse' || value === 'mask_baseline'
}

/** The Semantic tab's target: the selected class IDs and how much everything else is turned
 *  down; no direction. */
export function targetFor(
  labels: string[],
  attenuationDb: number = DEFAULT_ATTENUATION_DB,
): TargetSpec {
  return { labels, direction: null, source_id: null, attenuation_db: attenuationDb }
}
