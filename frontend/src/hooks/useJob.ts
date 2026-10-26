import { useEffect, useState } from 'react'
import { followJob, getJob } from '../api'
import { applyEvent, emptyStream, type Stream } from '../lib/events'
import { ACTIVE_STATUSES, type Job } from '../types'

/** One question: its stored state, plus the live events while it is being worked on. */
export function useJob(jobId: string) {
  const [job, setJob] = useState<Job | null>(null)
  const [stream, setStream] = useState<Stream>(emptyStream())
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let stop = () => {}
    setJob(null)
    setStream(emptyStream())
    setError(null)
    ;(async () => {
      try {
        const first = await getJob(jobId)
        if (cancelled) return
        setJob(first)
        if (ACTIVE_STATUSES.includes(first.status)) {
          stop = followJob(
            jobId,
            (e) => setStream((s) => applyEvent(s, e)),
            async () => {
              try {
                const last = await getJob(jobId)
                if (!cancelled) setJob(last)
              } catch (e) {
                if (!cancelled) setError(e instanceof Error ? e.message : 'Could not load the answer.')
              }
            },
          )
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Could not load this question.')
      }
    })()
    return () => { cancelled = true; stop() }
  }, [jobId])

  return { job, stream, error }
}
