import { useEffect, useRef } from 'react'
import { STAGES, stageIndex } from '../lib/format'
import type { Stream } from '../lib/events'
import type { Job, JobStatus } from '../types'
import { CheckIcon } from './ChecksList'

export function ProgressPanel({ job, stream }: { job: Job; stream: Stream }) {
  const status: JobStatus = stream.status ?? job.status
  const current = stageIndex(status)
  const progress = Math.max(job.progress ?? 0, stream.progress)
  const message = stream.message || job.message || 'Waiting to start'
  const interpretation = stream.interpretation || job.plan?.interpretation
  const list = useRef<HTMLUListElement>(null)
  // keep the newest line in view by scrolling the log itself, never the page
  useEffect(() => { if (list.current) list.current.scrollTop = list.current.scrollHeight }, [stream.log.length])

  return (
    <div className="space-y-4" role="status" aria-live="polite" aria-label="Progress">
      <ol className="grid grid-cols-4 gap-2">
        {STAGES.map((s, i) => {
          const done = i < current
          const active = i === current
          return (
            <li key={s.key} className="min-w-0">
              <div className={`h-1.5 rounded-full ${done ? 'bg-accent' : active ? 'bg-accent/60 pulse-dot' : 'bg-line'}`} aria-hidden="true" />
              <p className={`mt-1.5 truncate text-xs font-semibold ${active ? 'text-accent' : done ? 'text-ink' : 'text-ink-soft'}`}>
                <span className="sr-only">{done ? 'Done: ' : active ? 'Now: ' : 'Next: '}</span>{s.label}
              </p>
            </li>
          )
        })}
      </ol>

      <div>
        <div className="mb-1 flex items-baseline justify-between gap-3">
          <p className="min-w-0 truncate text-sm font-medium">{message}</p>
          <span className="num text-xs text-ink-soft">{Math.round(progress * 100)}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-paper-2" aria-hidden="true">
          <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${Math.max(3, Math.round(progress * 100))}%` }} />
        </div>
      </div>

      {interpretation && (
        <p className="rounded-lg border border-line bg-paper px-3 py-2 text-sm"><span className="font-semibold">Gauge understood: </span>{interpretation}</p>
      )}

      {stream.log.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-ink-soft">How Gauge is checking its work</p>
          <ul ref={list} className="max-h-44 space-y-1.5 overflow-y-auto pr-1">
            {stream.log.map((l) => (
              <li key={l.id} className="rise flex items-start gap-2 text-sm leading-snug"><CheckIcon ok={l.ok} /><span>{l.message}</span></li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
