import { useRef, useState } from 'react'

const EXAMPLES = [
  'What is the swing period?',
  "What is the spring's frequency?",
  'How many parts crossed the line?',
  'What is the acceleration of the ball?',
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
        id="question" ref={area} value={text} onChange={(e) => setText(e.target.value)} rows={2} maxLength={1000}
        onFocus={() => setFocused(true)} onBlur={() => setTimeout(() => setFocused(false), 150)}
        onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey || !e.shiftKey)) { e.preventDefault(); void submit() } }}
        placeholder={hasHistory ? 'Ask a follow-up about the same video…' : 'Ask a question about the video, e.g. “What is the swing period?”'}
        className="w-full resize-none rounded-lg border border-line bg-paper px-3 py-2 text-base placeholder:text-ink-soft/70 focus:border-accent focus:outline-none"
      />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <div className="flex flex-1 flex-wrap gap-1.5" aria-label="Example questions">
          {(!hasHistory || focused) && EXAMPLES.map((q) => (
            <button key={q} type="button" onClick={() => setText(q)} disabled={locked}
              className="rounded-full border border-line bg-paper px-3 py-1 text-xs font-medium text-ink-soft transition-colors hover:border-accent hover:text-accent disabled:opacity-50">
              {q}
            </button>
          ))}
        </div>
        <button type="submit" disabled={!text.trim() || locked}
          className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50">
          {locked ? 'Working…' : 'Ask'}
        </button>
      </div>
      {disabledReason && <p className="mt-2 text-xs text-ink-soft">{disabledReason}</p>}
      {error && <p role="alert" className="mt-2 text-sm text-bad">{error}</p>}
    </form>
  )
}
