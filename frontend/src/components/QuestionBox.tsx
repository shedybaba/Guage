import { useRef, useState } from 'react'
import { MAX_SECONDS, useVoice } from '../hooks/useVoice'

const EXAMPLES = [
  'How many parts crossed the line?',
  'What is the acceleration of the falling ball?',
  'What is the swing period?',
  "What is the spring's frequency?",
  'How fast is the wheel turning?',
  'Is momentum conserved in the collision?',
]

export function QuestionBox({ onAsk, busy, disabledReason, hasHistory }: {
  onAsk: (q: string) => Promise<void>
  busy: boolean
  disabledReason?: string
  hasHistory: boolean
}) {
  const [text, setText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [sending, setSending] = useState(false)
  const [focused, setFocused] = useState(false)
  const area = useRef<HTMLTextAreaElement>(null)
  const locked = busy || sending
  const [heard, setHeard] = useState(false)
  const voice = useVoice((spoken) => {
    // add to what is already typed; the person reads and corrects it before asking
    setText((t) => (t.trim() ? `${t.trim()} ${spoken}` : spoken))
    setHeard(true)
    area.current?.focus()
  })
  const listening = voice.state !== 'idle'
  const clock = (s: number) => `0:${String(Math.floor(s)).padStart(2, '0')}`

  const submit = async () => {
    const q = text.trim()
    if (!q || locked) return
    setError(null)
    setSending(true)
    try {
      await onAsk(q)
      setText('')
      area.current?.blur() // so the example chips fold away and do not cover the answer
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not send the question.')
    } finally {
      setSending(false)
    }
  }

  return (
    <form onSubmit={(e) => { e.preventDefault(); void submit() }} className="rounded-2xl border border-line bg-card p-3 shadow-sm">
      <label htmlFor="question" className="sr-only">Your question about the video</label>
      <textarea
        id="question" ref={area} value={text} onChange={(e) => { setText(e.target.value); setHeard(false) }} readOnly={listening} rows={2} maxLength={1000}
        onFocus={() => setFocused(true)} onBlur={() => setTimeout(() => setFocused(false), 150)}
        onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey || !e.shiftKey)) { e.preventDefault(); void submit() } }}
        placeholder={hasHistory ? 'Ask a follow-up about the same video…' : 'Ask what you want to measure, e.g. “How many parts crossed the line?”'}
        className="w-full resize-none rounded-lg border border-line bg-paper px-3 py-2 text-base placeholder:text-ink-soft/70 focus:border-accent focus:outline-none"
      />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {listening ? (
          <div className="flex flex-1 items-center gap-3 text-sm" role="status" aria-live="polite">
            {voice.state === 'recording' ? (
              <>
                <span className="relative flex h-2.5 w-2.5" aria-hidden="true">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-bad opacity-60 motion-reduce:animate-none" />
                  <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-bad" />
                </span>
                <span className="font-mono tabular-nums">Listening {clock(voice.seconds)} / {clock(MAX_SECONDS)}</span>
                <button type="button" onClick={voice.cancel} className="text-xs font-medium text-ink-soft underline-offset-2 hover:underline">Cancel</button>
              </>
            ) : (
              <span className="text-ink-soft">Writing down what you said…</span>
            )}
          </div>
        ) : (
        <div className="flex flex-1 flex-wrap gap-1.5" aria-label="Example questions">
          {(!hasHistory || focused) && EXAMPLES.map((q) => (
            <button key={q} type="button" onClick={() => setText(q)} disabled={locked}
              className="rounded-full border border-line bg-paper px-3 py-1 text-xs font-medium text-ink-soft transition-colors hover:border-accent hover:text-accent disabled:opacity-50">
              {q}
            </button>
          ))}
        </div>
        )}
        {voice.supported && (
          voice.state === 'recording' ? (
            <button type="button" onClick={voice.stop} aria-label="Stop recording"
              className="flex h-10 w-10 items-center justify-center rounded-full bg-bad text-white transition-colors hover:opacity-90">
              <span className="h-3.5 w-3.5 rounded-sm bg-white" aria-hidden="true" />
            </button>
          ) : (
            <button type="button" onClick={() => void voice.start()} disabled={locked || listening} aria-label="Speak your question"
              title="Speak your question"
              className="flex h-10 w-10 items-center justify-center rounded-full border border-line bg-paper text-ink-soft transition-colors hover:border-accent hover:text-accent disabled:opacity-50">
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
              </svg>
            </button>
          )
        )}
        <button type="submit" disabled={!text.trim() || locked || listening}
          className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50">
          {locked ? 'Working…' : 'Ask'}
        </button>
      </div>
      {heard && !listening && <p className="mt-2 text-xs text-ink-soft">This is what Gauge heard. Check the numbers and units, fix anything, then press Ask.</p>}
      {disabledReason && <p className="mt-2 text-xs text-ink-soft">{disabledReason}</p>}
      {voice.error && <p role="alert" className="mt-2 text-sm text-bad">{voice.error}</p>}
      {error && <p role="alert" className="mt-2 text-sm text-bad">{error}</p>}
    </form>
  )
}
