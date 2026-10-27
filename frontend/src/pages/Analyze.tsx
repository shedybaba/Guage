import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ask, getJob, getTracks, getVideo, rawSegmentationUrl, videoUrl } from '../api'
import { JobCard } from '../components/JobCard'
import { QuestionBox } from '../components/QuestionBox'
import { TrackedVideo } from '../components/TrackedVideo'
import { UploadCard } from '../components/UploadCard'
import { formatDuration } from '../lib/format'
import { ACTIVE_STATUSES, type Job, type Tracks, type VideoInfo } from '../types'

interface Overlay { jobId: string; tracksUrl: string; phrase: string }

export default function Analyze() {
  const { videoId } = useParams()
  const navigate = useNavigate()
  if (!videoId) return <UploadCard onUploaded={(v) => navigate(`/analyze/${v.video_id}`)} />
  return <Workspace key={videoId} videoId={videoId} onNew={() => navigate('/analyze')} />
}

function Workspace({ videoId, onNew }: { videoId: string; onNew: () => void }) {
  const [info, setInfo] = useState<VideoInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [jobIds, setJobIds] = useState<string[]>([])
  const [answered, setAnswered] = useState<Set<string>>(new Set()) // paused questions that later got their answer
  const [busy, setBusy] = useState<Set<string>>(new Set())
  const [overlay, setOverlay] = useState<Overlay | null>(null)
  const [tracks, setTracks] = useState<Tracks | null>(null)
  const lastCard = useRef<HTMLDivElement>(null)
  const firstRender = useRef(true)

  // load the video and its earlier questions (so a reload or a shared link shows the whole conversation)
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const v = await getVideo(videoId)
        if (cancelled) return
        setInfo(v)
        const qs = v.questions ?? []
        setJobIds(qs.map((q) => q.job_id))
        const jobs = await Promise.all(qs.map((q) => getJob(q.job_id).catch(() => null)))
        if (cancelled) return
        const loaded = jobs.filter((j): j is Job => !!j)
        setAnswered(new Set(loaded.map((j) => j.parent_id).filter((p): p is string => !!p)))
        setBusy(new Set(loaded.filter((j) => ACTIVE_STATUSES.includes(j.status)).map((j) => j.job_id)))
        const lastMeasured = [...loaded].reverse().find((j) => j.result?.kind === 'measurement')
        if (lastMeasured && lastMeasured.result?.kind === 'measurement') {
          setOverlay({ jobId: lastMeasured.job_id, tracksUrl: lastMeasured.result.tracks.url, phrase: lastMeasured.result.tracks.phrase })
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Could not open this video.')
      }
    })()
    return () => { cancelled = true }
  }, [videoId])

  useEffect(() => {
    let cancelled = false
    setTracks(null)
    if (!overlay) return
    getTracks(overlay.tracksUrl).then((t) => { if (!cancelled) setTracks(t) }).catch(() => { if (!cancelled) setTracks(null) })
    return () => { cancelled = true }
  }, [overlay])

  // bring a new question into view
  useEffect(() => {
    if (firstRender.current) { firstRender.current = false; return }
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    lastCard.current?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })
  }, [jobIds.length])

  const start = useCallback((id: string) => {
    setJobIds((ids) => [...ids, id])
    setBusy((b) => new Set(b).add(id))
  }, [])

  const onAsk = useCallback(async (q: string) => { start((await ask(videoId, q)).job_id) }, [videoId, start])

  const onFinished = useCallback((job: Job) => {
    setBusy((b) => { if (!b.has(job.job_id)) return b; const n = new Set(b); n.delete(job.job_id); return n })
    if (job.status === 'done' && job.result?.kind === 'measurement') {
      const r = job.result
      setOverlay((o) => (o?.jobId === job.job_id ? o : { jobId: job.job_id, tracksUrl: r.tracks.url, phrase: r.tracks.phrase }))
    }
  }, [])

  const onShow = useCallback((job: Job) => {
    if (job.result?.kind === 'measurement') setOverlay({ jobId: job.job_id, tracksUrl: job.result.tracks.url, phrase: job.result.tracks.phrase })
  }, [])

  const onAnswered = useCallback((parent: string) => (child: string) => {
    setAnswered((a) => new Set(a).add(parent))
    start(child)
  }, [start])

  if (error) {
    return (
      <div className="mx-auto max-w-xl px-4 py-16 text-center">
        <p role="alert" className="rounded-lg border border-bad/30 bg-bad-soft px-4 py-3 text-bad">{error}</p>
        <button type="button" onClick={onNew} className="mt-6 rounded-lg bg-accent px-5 py-2.5 font-semibold text-white hover:bg-accent-strong">Upload a video</button>
      </div>
    )
  }
  if (!info) return <p className="p-10 text-center text-ink-soft" role="status">Opening your video…</p>

  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <p className="min-w-0 truncate text-sm text-ink-soft">
          <span className="font-semibold text-ink">{info.filename ?? 'Your video'}</span>
          <span className="num"> · {formatDuration(info.duration)} · {info.fps.toFixed(0)} fps · {info.width}×{info.height}</span>
        </p>
        <button type="button" onClick={onNew} className="text-sm font-medium text-ink-soft underline underline-offset-4 hover:text-ink">Use another video</button>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        <div className="sticky top-[3.7rem] z-20 -mx-4 bg-paper/95 px-4 pb-2 pt-1 backdrop-blur lg:top-20 lg:mx-0 lg:self-start lg:bg-transparent lg:p-0 lg:backdrop-blur-none">
          <TrackedVideo
            src={videoUrl(videoId)} info={info} tracks={tracks}
            rawUrl={overlay ? rawSegmentationUrl(videoId, overlay.phrase) : null}
          />
        </div>

        <div className="min-w-0 space-y-4">
          {jobIds.length === 0 && (
            <div className="graph-paper rounded-2xl border border-dashed border-line bg-card p-6">
              <p className="text-lg font-semibold">Ask your first question</p>
              <p className="mt-1 text-sm leading-relaxed text-ink-soft">
                Describe what you want measured. Gauge looks at the video, finds the object by name, follows it, and works out the answer with code.
                If it needs something (a length, a mass) it will ask.
              </p>
            </div>
          )}
          {jobIds.map((id, i) => (
            <div key={id} ref={i === jobIds.length - 1 ? lastCard : undefined} className="scroll-mt-20">
              <JobCard
                jobId={id} superseded={answered.has(id)} shown={overlay?.jobId === id}
                onShow={onShow} onFinished={onFinished} onAnswered={onAnswered(id)} onRetry={(q) => { void onAsk(q) }}
              />
            </div>
          ))}
          <div className="lg:sticky lg:bottom-3 lg:z-10">
            <QuestionBox onAsk={onAsk} busy={busy.size > 0} hasHistory={jobIds.length > 0}
              disabledReason={busy.size > 0 ? 'Gauge is working on your last question.' : undefined} />
          </div>
        </div>
      </div>
    </div>
  )
}
