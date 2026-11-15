// Building TargetSpec messages for the class chips.
import type { TargetSpec } from './types.ts'

/** The Semantic tab's target: the selected class IDs, no direction, default attenuation. */
export function targetFor(labels: string[]): TargetSpec {
  return { labels, direction: null, source_id: null, attenuation_db: 30 }
}
