import type { CheckEntry } from '../types'

const ICON: Record<string, { glyph: string; cls: string; label: string }> = {
  ok: { glyph: '✓', cls: 'bg-accent-soft text-accent', label: 'passed' },
  bad: { glyph: '!', cls: 'bg-copper-soft text-copper', label: 'problem found' },
  info: { glyph: '·', cls: 'bg-paper-2 text-ink-soft', label: 'step' },
}

export function CheckIcon({ ok }: { ok: boolean | null }) {
  const k = ok === true ? ICON.ok : ok === false ? ICON.bad : ICON.info
  return <span role="img" aria-label={k.label} className={`mt-0.5 inline-flex h-5 w-5 flex-none items-center justify-center rounded-full text-xs font-bold ${k.cls}`}>{k.glyph}</span>
}

/** "How Gauge checked its work": every retry and validation, in plain words. */
export function ChecksList({ checks }: { checks: CheckEntry[] }) {
  if (!checks.length) return <p className="text-sm text-ink-soft">No extra checks were needed.</p>
  return (
    <ul className="space-y-2">
      {checks.map((c, i) => (
        <li key={i} className="flex items-start gap-2.5 text-sm leading-relaxed">
          <CheckIcon ok={c.ok} />
          <span>{c.message}</span>
        </li>
      ))}
    </ul>
  )
}
