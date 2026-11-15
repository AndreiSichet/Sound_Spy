// Offline detection of a saved recording: one lane per class across the whole file, with the
// active segments drawn as bars (title = time span and mean probability).
import type { ClassInfo, DetectionTimeline } from '../../api/types.ts'

export function OfflineTimeline({
  timeline,
  classes,
}: {
  timeline: DetectionTimeline
  classes: ClassInfo[]
}) {
  const duration = Math.max(timeline.duration_s, 1e-3)
  const found = new Set(timeline.segments.map((s) => s.label))
  return (
    <div className="w-full space-y-1" data-testid="offline-timeline">
      {classes.map((cls) => (
        <div key={cls.name} className="flex items-center gap-2 text-xs">
          <span className="w-28 shrink-0 truncate text-slate-400">{cls.display_name}</span>
          <div className="relative h-3 flex-1 rounded bg-slate-900">
            {timeline.segments
              .filter((s) => s.label === cls.name)
              .map((s) => (
                <span
                  key={`${s.t_start}-${s.t_end}`}
                  className="absolute inset-y-0 rounded bg-emerald-500"
                  style={{
                    left: `${(s.t_start / duration) * 100}%`,
                    width: `${Math.max(0.5, ((s.t_end - s.t_start) / duration) * 100)}%`,
                    opacity: 0.4 + 0.6 * s.probability,
                  }}
                  title={`${s.t_start.toFixed(2)}–${s.t_end.toFixed(2)} s, p ${s.probability.toFixed(2)}`}
                  data-testid={`segment-${cls.name}`}
                />
              ))}
          </div>
        </div>
      ))}
      <p className="text-xs text-slate-500">
        {found.size
          ? `${timeline.segments.length} segment(s) over ${timeline.duration_s.toFixed(1)} s.`
          : `Nothing detected in ${timeline.duration_s.toFixed(1)} s.`}
      </p>
    </div>
  )
}
