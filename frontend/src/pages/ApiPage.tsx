import { useState } from 'react'

const BASE = typeof window !== 'undefined' ? window.location.origin : 'https://your-gauge-host'

function Code({ children, label }: { children: string; label: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="relative">
      <button type="button" aria-label={`Copy ${label}`}
        onClick={() => { void navigator.clipboard?.writeText(children).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) }) }}
        className="absolute right-2 top-2 rounded-md border border-white/20 bg-white/10 px-2 py-1 text-xs text-paper hover:bg-white/20">
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre className="overflow-x-auto rounded-xl bg-ink p-4 pr-16 text-[13px] leading-relaxed text-paper"><code className="font-mono">{children}</code></pre>
    </div>
  )
}

const ASK = `# 1. upload a video
curl -F "file=@my_video.mp4" ${BASE}/api/videos
# -> {"video_id": "2feb8f27…", "fps": 29.97, "frame_count": 362, "width": 720, "height": 1280, "duration": 12.05}

# 2. ask a question about it (any plain-English question: "How many parts crossed the line?",
#    "How fast is the wheel turning?", "What is the acceleration of the falling ball?" ...)
curl -X POST ${BASE}/api/videos/2feb8f27…/questions \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What is the swing period?"}'
# -> {"job_id": "cb217e29…"}

# 3. follow the progress live (server-sent events), or poll the job
curl -N ${BASE}/api/jobs/cb217e29…/events
curl ${BASE}/api/jobs/cb217e29…`

const RESULT = `{
  "status": "done",
  "progress": 1.0,
  "result": {
    "kind": "measurement",
    "tool": "pendulum_period",
    "headline": { "label": "Swing period", "display": "0.8582 ± 0.0016 s",
                  "value": 0.8582, "uncertainty": 0.0016, "units": "s" },
    "explanation": { "summary": "The swing period is 0.8582 ± 0.0016 s.", "source": "model" },
    "checks": [ { "step": "track", "message": "Tracked 'glass marble': found 1 object(s).", "ok": true } ],
    "tracks": { "url": "/api/jobs/cb217e29…/tracks" },
    "downloads": { "csv": "/api/jobs/cb217e29…/data.csv" }
  }
}`

const NEEDS = `# A question can pause because Gauge needs something it cannot guess:
#   "status": "needs_input", "needs": { "missing": [{ "name": "string_length", "question": "What is the length of the string…" }] }
# Answer it (units are converted for you) and a new job starts:
curl -X POST ${BASE}/api/jobs/<job_id>/answers \\
  -H "Content-Type: application/json" \\
  -d '{"answers": {"string_length": {"value": 50, "unit": "cm"}}}'`

const ENDPOINTS: [string, string, string][] = [
  ['POST', '/api/videos', 'Upload a video (multipart field "file"). Returns the video id and its facts.'],
  ['POST', '/api/videos/{id}/questions', 'Ask a question. Optional "known_values" such as string_length_m or scale_m_per_px. Returns a job id.'],
  ['GET', '/api/jobs/{id}', 'Status, progress, plan, result, or what is needed.'],
  ['GET', '/api/jobs/{id}/events', 'Live progress as server-sent events; reconnect with Last-Event-ID.'],
  ['POST', '/api/jobs/{id}/answers', 'Answer a paused question.'],
  ['GET', '/api/jobs/{id}/tracks', 'The final tracks (boxes and positions per frame).'],
  ['GET', '/api/jobs/{id}/data.csv', 'The tracked data as CSV.'],
  ['POST', '/api/transcribe', 'A spoken question (audio up to 30 s). Returns the words heard, for you to check before asking.'],
  ['GET', '/api/videos/{id}/video', 'The video, for playback (supports range requests).'],
]

export default function ApiPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-10">
      <h1 className="text-3xl font-semibold tracking-tight">Use Gauge from your own code</h1>
      <p className="mt-2 max-w-2xl text-lg leading-relaxed text-ink-soft">
        Everything on this site is a plain HTTP API. Other apps can plug Gauge in with a few calls.
      </p>

      <h2 className="mb-3 mt-10 text-xl font-semibold">Upload, ask, follow</h2>
      <Code label="the upload and question example">{ASK}</Code>

      <h2 className="mb-3 mt-10 text-xl font-semibold">What comes back</h2>
      <Code label="an example result">{RESULT}</Code>
      <p className="mt-3 text-sm leading-relaxed text-ink-soft">
        Every number is formatted by code. The explanation is written by a language model, and every number in it is checked
        against the measurement; if it fails the check, a template written in code is used and says so.
      </p>

      <h2 className="mb-3 mt-10 text-xl font-semibold">When Gauge needs something</h2>
      <Code label="the answers example">{NEEDS}</Code>

      <h2 className="mb-3 mt-10 text-xl font-semibold">All the calls</h2>
      <div className="overflow-x-auto rounded-xl border border-line bg-card">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-line bg-paper-2/60 text-xs uppercase tracking-wide text-ink-soft">
            <tr><th className="px-3 py-2">Method</th><th className="px-3 py-2">Path</th><th className="px-3 py-2">What it does</th></tr>
          </thead>
          <tbody>
            {ENDPOINTS.map(([m, p, d]) => (
              <tr key={p + m} className="border-b border-line/60 last:border-0">
                <td className="num px-3 py-2 font-semibold text-accent">{m}</td>
                <td className="num px-3 py-2 whitespace-nowrap">{p}</td>
                <td className="px-3 py-2 text-ink-soft">{d}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-sm">
        The complete, interactive reference is at <a className="font-semibold text-accent underline underline-offset-4" href="/docs">/docs</a>.
      </p>
    </div>
  )
}
