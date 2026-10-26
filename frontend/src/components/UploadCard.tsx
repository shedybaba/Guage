import { useRef, useState } from 'react'
import { uploadVideo } from '../api'
import { formatBytes } from '../lib/format'
import type { VideoInfo } from '../types'

const TIPS = [
  ['Keep the phone completely still', 'A tripod, or propped against something solid. Camera shake is the biggest enemy of a good measurement.'],
  ['Keep the object fully in view', 'Film the whole motion, from before it starts to after it ends, in good even light.'],
  ['Put a ruler in the same plane', 'A ruler, a metre stick or a sheet of A4 paper next to the motion lets Gauge turn pixels into metres.'],
  ['Use slow motion for fast things', 'Falling objects and collisions are over in a blink. 120 or 240 frames per second makes them measurable.'],
]

export function UploadCard({ onUploaded }: { onUploaded: (v: VideoInfo) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const cancel = useRef<(() => void) | null>(null)
  const [drag, setDrag] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [progress, setProgress] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const start = async (f: File | undefined | null) => {
    if (!f) return
    setError(null)
    if (!(f.type.startsWith('video/') || /\.(mp4|mov|m4v|webm|mkv|avi|3gp)$/i.test(f.name))) {
      setError('That does not look like a video. Use an MP4 or MOV file.')
      return
    }
    setFile(f)
    setBusy(true)
    setProgress(0)
    const { promise, cancel: c } = uploadVideo(f, setProgress)
    cancel.current = c
    try {
      onUploaded(await promise)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The upload failed.')
      setFile(null)
    } finally {
      setBusy(false)
      cancel.current = null
    }
  }

  return (
    <div className="mx-auto grid max-w-5xl gap-6 px-4 py-10 md:grid-cols-[1.2fr_1fr]">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Upload a video</h1>
        <p className="mt-2 max-w-lg text-ink-soft">Then ask a question about it in plain English. Gauge answers only what you ask.</p>
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); void start(e.dataTransfer.files[0]) }}
          className={`graph-paper mt-6 flex min-h-64 flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${drag ? 'border-accent bg-accent-soft' : 'border-line bg-card'}`}
        >
          {busy && file ? (
            <div className="w-full max-w-sm" role="status" aria-live="polite">
              <p className="truncate font-medium">{file.name}</p>
              <p className="num text-sm text-ink-soft">{formatBytes(file.size)}</p>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-paper-2" aria-hidden="true">
                <div className="h-full bg-accent transition-[width]" style={{ width: `${Math.round(progress * 100)}%` }} />
              </div>
              <p className="mt-2 text-sm text-ink-soft">{progress < 1 ? `Uploading ${Math.round(progress * 100)}%` : 'Checking the video and preparing it…'}</p>
              <button type="button" onClick={() => cancel.current?.()} className="mt-3 text-sm font-medium text-ink-soft underline underline-offset-4 hover:text-ink">Cancel</button>
            </div>
          ) : (
            <>
              <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#0e6a60" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="3" y="5" width="13" height="14" rx="2" /><path d="M16 10l5-3v10l-5-3" />
              </svg>
              <p className="text-lg font-medium">Drop a video here</p>
              <button type="button" onClick={() => input.current?.click()} className="rounded-lg bg-accent px-5 py-2.5 font-semibold text-white transition-colors hover:bg-accent-strong">
                Choose a file
              </button>
              <p className="text-sm text-ink-soft">MP4 or MOV, up to about 10 minutes</p>
            </>
          )}
          <input ref={input} type="file" accept="video/*,.mp4,.mov,.m4v,.webm" className="sr-only" aria-label="Choose a video file"
            onChange={(e) => { void start(e.target.files?.[0]); e.target.value = '' }} />
        </div>
        {error && <p role="alert" className="mt-4 rounded-lg border border-bad/30 bg-bad-soft px-4 py-3 text-sm text-bad">{error}</p>}
      </div>
      <aside className="rounded-2xl border border-line bg-card p-5">
        <h2 className="text-lg font-semibold">Filming tips</h2>
        <ul className="mt-3 space-y-3">
          {TIPS.map(([title, text]) => (
            <li key={title}>
              <p className="font-medium">{title}</p>
              <p className="text-sm leading-relaxed text-ink-soft">{text}</p>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  )
}
