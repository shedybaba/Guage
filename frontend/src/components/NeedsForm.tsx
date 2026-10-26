import { useState } from 'react'
import { defaultUnit, parseNumber, UNITS } from '../lib/answers'
import type { AnswerBody } from '../api'
import type { Needs } from '../types'

interface Field { name: string; label: string; dimension: string | null; prefill: string; unit: string; confirm: boolean; reason?: string }

const dimensionOf = (unit: string): string | null =>
  Object.entries(UNITS).find(([, list]) => list.some((u) => u.value === unit))?.[0] ?? null

/** Gauge could not go on: it needs an input, or wants a number it heard checked. Nothing is measured until this is answered. */
export function NeedsForm({ needs, onSubmit }: { needs: Needs; onSubmit: (a: AnswerBody) => Promise<void> }) {
  const fields: Field[] = [
    ...needs.confirm.map((c) => {
      const dim = dimensionOf(c.unit)
      return { name: c.name, label: c.question, dimension: dim, prefill: String(Number(c.value.toPrecision(6))), unit: c.unit || defaultUnit(dim), confirm: true, reason: c.reason }
    }),
    ...needs.missing.map((m) => ({ name: m.name, label: m.question, dimension: m.dimension, prefill: '', unit: defaultUnit(m.dimension), confirm: false })),
  ]
  const [vals, setVals] = useState<Record<string, { text: string; unit: string }>>(
    Object.fromEntries(fields.map((f) => [f.name, { text: f.prefill, unit: f.unit }])),
  )
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [serverError, setServerError] = useState<string | null>(null)

  const submit = async () => {
    const body: AnswerBody = {}
    const errs: Record<string, string> = {}
    for (const f of fields) {
      const v = vals[f.name]
      if (f.name === 'reference_object') {
        if (!v.text.trim()) errs[f.name] = 'Enter a name.'
        else body[f.name] = { text: v.text.trim() }
        continue
      }
      const p = parseNumber(v.text)
      if (!p.ok) errs[f.name] = p.error
      else body[f.name] = { value: p.value, unit: v.unit }
    }
    setErrors(errs)
    if (Object.keys(errs).length) return
    setBusy(true)
    setServerError(null)
    try { await onSubmit(body) } catch (e) { setServerError(e instanceof Error ? e.message : 'Could not send that.') } finally { setBusy(false) }
  }

  return (
    <form onSubmit={(e) => { e.preventDefault(); void submit() }} className="space-y-4" aria-label="Gauge needs a little more">
      <p className="text-sm font-semibold text-copper">Gauge needs a little more before it measures.</p>
      {needs.interpretation && <p className="text-sm text-ink-soft">{needs.interpretation}</p>}
      {fields.map((f) => (
        <div key={f.name} className="rounded-lg border border-line bg-paper p-3">
          <label htmlFor={`f-${f.name}`} className="block text-sm font-medium leading-snug">{f.label}</label>
          {f.confirm && f.reason && <p className="mt-1 text-xs text-ink-soft">{f.reason}</p>}
          <div className="mt-2 flex gap-2">
            <input
              id={`f-${f.name}`} value={vals[f.name].text} inputMode={f.name === 'reference_object' ? 'text' : 'decimal'}
              placeholder={f.name === 'reference_object' ? 'for example: metre stick' : 'number'}
              onChange={(e) => setVals({ ...vals, [f.name]: { ...vals[f.name], text: e.target.value } })}
              aria-invalid={!!errors[f.name]} aria-describedby={errors[f.name] ? `e-${f.name}` : undefined}
              className="num min-w-0 flex-1 rounded-md border border-line bg-card px-3 py-2 focus:border-accent focus:outline-none"
            />
            {f.name !== 'reference_object' && f.dimension && (
              <select aria-label={`Unit for ${f.name.replace('_', ' ')}`} value={vals[f.name].unit}
                onChange={(e) => setVals({ ...vals, [f.name]: { ...vals[f.name], unit: e.target.value } })}
                className="rounded-md border border-line bg-card px-2 py-2 focus:border-accent focus:outline-none">
                {UNITS[f.dimension].map((u) => <option key={u.value} value={u.value}>{u.label}</option>)}
              </select>
            )}
          </div>
          {errors[f.name] && <p id={`e-${f.name}`} role="alert" className="mt-1 text-xs text-bad">{errors[f.name]}</p>}
        </div>
      ))}
      {needs.notes.length > 0 && (
        <ul className="space-y-1 text-xs text-ink-soft">{needs.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
      )}
      {serverError && <p role="alert" className="text-sm text-bad">{serverError}</p>}
      <button type="submit" disabled={busy} className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-accent-strong disabled:opacity-60">
        {busy ? 'Sending…' : 'Continue'}
      </button>
    </form>
  )
}
