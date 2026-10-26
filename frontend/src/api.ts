import type { Job, JobEvent, Tracks, VideoInfo } from './types'

const BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function check(r: Response): Promise<Response> {
  if (r.ok) return r
  let detail = `The server answered ${r.status}.`
  try {
    const d = await r.json()
    if (typeof d.detail === 'string') detail = d.detail
    else if (Array.isArray(d.detail) && d.detail[0]?.msg) detail = String(d.detail[0].msg)
  } catch { /* not JSON */ }
  throw new ApiError(detail, r.status)
}

async function getJson<T>(path: string): Promise<T> {
  return (await (await check(await fetch(BASE + path))).json()) as T
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  return (await (await check(r)).json()) as T
}

/** Upload with progress (fetch cannot report upload progress, XMLHttpRequest can). */
export function uploadVideo(file: File, onProgress: (fraction: number) => void): { promise: Promise<VideoInfo>; cancel: () => void } {
  const xhr = new XMLHttpRequest()
  const promise = new Promise<VideoInfo>((resolve, reject) => {
    xhr.open('POST', BASE + '/api/videos')
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total) }
    xhr.onload = () => {
      let body: any = null
      try { body = JSON.parse(xhr.responseText) } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300 && body) resolve(body as VideoInfo)
      else reject(new ApiError(typeof body?.detail === 'string' ? body.detail : `The upload failed (${xhr.status}).`, xhr.status))
    }
    xhr.onerror = () => reject(new ApiError('Could not reach Gauge. Check your connection and try again.', 0))
    xhr.onabort = () => reject(new ApiError('Upload cancelled.', 0))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
  return { promise, cancel: () => xhr.abort() }
}

export const getVideo = (id: string) => getJson<VideoInfo>(`/api/videos/${id}`)
export const getJob = (id: string) => getJson<Job>(`/api/jobs/${id}`)
export const getTracks = (url: string) => getJson<Tracks>(url)
export const videoUrl = (id: string) => `${BASE}/api/videos/${id}/video`
export const csvUrl = (jobId: string) => `${BASE}/api/jobs/${jobId}/data.csv`
export const rawSegmentationUrl = (videoId: string, phrase: string) =>
  `${BASE}/api/videos/${videoId}/segmentation?phrase=${encodeURIComponent(phrase)}&format=raw`

export const ask = (videoId: string, question: string, knownValues?: Record<string, number | string>) =>
  postJson<{ job_id: string }>(`/api/videos/${videoId}/questions`, { question, ...(knownValues ? { known_values: knownValues } : {}) })

export type AnswerBody = Record<string, { value?: number; unit?: string; text?: string }>
export const answer = (jobId: string, answers: AnswerBody) => postJson<{ job_id: string }>(`/api/jobs/${jobId}/answers`, { answers })

/**
 * Follow a job's events live. Uses server-sent events (the browser reconnects by itself and resumes from the last
 * event id); if they are unavailable it polls. Returns a function that stops listening.
 */
export function followJob(jobId: string, onEvent: (e: JobEvent) => void, onEnd: (status: string) => void): () => void {
  let closed = false
  let es: EventSource | null = null
  let timer: ReturnType<typeof setTimeout> | null = null
  let last = 0

  const poll = async () => {
    if (closed) return
    try {
      const job = await getJob(jobId)
      if (['done', 'failed', 'needs_input'].includes(job.status)) { onEnd(job.status); return }
    } catch { /* try again */ }
    timer = setTimeout(poll, 1500)
  }

  if (typeof EventSource === 'undefined') {
    poll()
  } else {
    es = new EventSource(`${BASE}/api/jobs/${jobId}/events`)
    const handle = (m: MessageEvent) => {
      try {
        const e = JSON.parse(m.data) as JobEvent
        if (e.id > last) { last = e.id; onEvent(e) }
      } catch { /* ignore a malformed line */ }
    }
    for (const kind of ['status', 'plan', 'progress', 'log', 'needs', 'result', 'error']) es.addEventListener(kind, handle as EventListener)
    es.addEventListener('end', (m) => {
      es?.close()
      let status = 'done'
      try { status = JSON.parse((m as MessageEvent).data).status } catch { /* default */ }
      if (!closed) onEnd(status)
    })
  }
  return () => { closed = true; es?.close(); if (timer) clearTimeout(timer) }
}
