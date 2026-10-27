import type { JobEvent, JobStatus } from '../types'

/** What the progress panel shows, built up from the server's events. Pure, so it is easy to test. */
export interface Stream {
  events: JobEvent[]
  lastId: number
  status: JobStatus | null
  progress: number
  message: string
  frames: { done: number; total: number } | null
  log: { id: number; message: string; ok: boolean | null; step: string }[]
  interpretation: string
  tool: string | null
  ended: boolean
  error: string | null
}

export const emptyStream = (): Stream => ({
  events: [], lastId: 0, status: null, progress: 0, message: '', frames: null, log: [],
  interpretation: '', tool: null, ended: false, error: null,
})

const STATUS_FROM_MESSAGE: Record<string, JobStatus> = {
  'Reading your question and looking at the video': 'planning',
  'Finding and following the object in every frame': 'tracking',
  Measuring: 'measuring',
  'Writing the explanation': 'explaining',
}

export function applyEvent(s: Stream, e: JobEvent): Stream {
  if (e.id <= s.lastId) return s // replayed after a reconnect: already seen
  const next: Stream = { ...s, events: [...s.events, e], lastId: e.id }
  if (e.kind === 'status') {
    next.message = e.message
    next.status = STATUS_FROM_MESSAGE[e.message] ?? s.status
  } else if (e.kind === 'plan') {
    next.interpretation = e.message
    const tool = (e.data as { plan?: { tool?: unknown } } | null)?.plan?.tool
    next.tool = typeof tool === 'string' ? tool : s.tool
  } else if (e.kind === 'progress') {
    next.message = e.message
    const d = e.data as { done?: number; total?: number } | null
    if (d && typeof d.done === 'number' && typeof d.total === 'number') {
      next.frames = { done: Math.min(d.done, d.total), total: d.total }
      next.progress = Math.max(s.progress, 0.1 + 0.55 * Math.min(1, d.done / Math.max(d.total, 1)))
    }
  } else if (e.kind === 'log') {
    const d = e.data as { step?: string; ok?: boolean | null } | null
    next.log = [...s.log, { id: e.id, message: e.message, ok: d?.ok ?? null, step: d?.step ?? 'check' }]
  } else if (e.kind === 'error') {
    next.error = e.message
  }
  return next
}

/** Server-sent event text -> the events in it (used by tests and by the fallback reader). */
export function parseSse(text: string): { event: string; data: string; id?: string }[] {
  const out: { event: string; data: string; id?: string }[] = []
  for (const block of text.split('\n\n')) {
    const fields: Record<string, string> = {}
    for (const line of block.split('\n')) {
      if (!line || line.startsWith(':')) continue
      const i = line.indexOf(': ')
      if (i > 0) fields[line.slice(0, i)] = line.slice(i + 2)
    }
    if (fields.event) out.push({ event: fields.event, data: fields.data ?? '', id: fields.id })
  }
  return out
}
