// Building TargetSpec messages for the class chips and the attenuation slider.
import type { TargetSpec } from './types.ts'

export const DEFAULT_ATTENUATION_DB = 30
export const MAX_ATTENUATION_DB = 40

/** The Semantic tab's target: the selected class IDs and how much everything else is turned
 *  down; no direction. */
export function targetFor(
  labels: string[],
  attenuationDb: number = DEFAULT_ATTENUATION_DB,
): TargetSpec {
  return { labels, direction: null, source_id: null, attenuation_db: attenuationDb }
}
