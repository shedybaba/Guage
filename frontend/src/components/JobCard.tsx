import { useEffect } from 'react'
import { answer, type AnswerBody } from '../api'
import { useJob } from '../hooks/useJob'
import { ACTIVE_STATUSES, type Job } from '../types'
import { NeedsForm } from './NeedsForm'
import { ProgressPanel } from './ProgressPanel'
import { NoAnswerCard, ResultCard } from './ResultCard'

interface Props {
  jobId: string
  superseded: boolean // a later question answered this one's request for input
  shown: boolean // its tracks are the ones drawn on the video
  onShow: (job: Job) => void
  onFinished: (job: Job) => void
  onAnswered: (newJobId: string) => void
  onRetry: (question: string) => void
}

/** One question and its answer, whatever state it is in. */
export function JobCard({ jobId, superseded, shown, onShow, onFinished, onAnswered, onRetry }: Props) {
  const { job, stream, error } = useJob(jobId)
  const status = job ? (stream.status && ACTIVE_STATUSES.includes(job.status) ? stream.status : job.status) : null

  useEffect(() => {
    if (job && !ACTIVE_STATUSES.includes(job.status)) onFinished(job)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status, job?.job_id])

  const send = async (a: AnswerBody) => {
    if (!job) return
    const { job_id } = await answer(job.job_id, a)
    onAnswered(job_id)
  }

  return (
    <section className="rounded-2xl border border-line bg-card p-4 shadow-sm sm:p-5" aria-label={job ? `Question: ${job.question}` : 'Question'}>
      <p className="text-sm text-ink-soft">You asked</p>
      <h2 className="mb-4 text-lg font-semibold leading-snug">{job?.question ?? '…'}</h2>

      {error && <p role="alert" className="rounded-lg border border-bad/30 bg-bad-soft px-3 py-2 text-sm text-bad">{error}</p>}
      {!job && !error && <p className="text-sm text-ink-soft" role="status">Loading…</p>}

      {job && status && ACTIVE_STATUSES.includes(status) && <ProgressPanel job={job} stream={stream} />}

      {job?.status === 'needs_input' && job.needs && (
        superseded
          ? <p className="text-sm text-ink-soft">Answered below.</p>
          : <NeedsForm needs={job.needs} onSubmit={send} />
      )}

      {job?.status === 'failed' && (
        <div className="space-y-3" role="alert">
          <p className="rounded-lg border border-bad/30 bg-bad-soft px-3 py-2 text-sm text-bad">{job.message ?? 'Something went wrong.'}</p>
          <button type="button" onClick={() => onRetry(job.question)} className="rounded-lg border border-line bg-card px-4 py-2 text-sm font-semibold hover:border-ink/40">Ask again</button>
        </div>
      )}

      {job?.status === 'done' && job.result && (
        job.result.kind === 'measurement'
          ? <ResultCard job={job} result={job.result} shown={shown} onShow={() => onShow(job)} />
          : <NoAnswerCard result={job.result} />
      )}
    </section>
  )
}
