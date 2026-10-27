/** Turning what a person types into an answer the API accepts (the server converts the units; this only checks). */

export const UNITS: Record<string, { value: string; label: string }[]> = {
  length: [{ value: 'm', label: 'm' }, { value: 'cm', label: 'cm' }, { value: 'mm', label: 'mm' }, { value: 'in', label: 'in' }, { value: 'ft', label: 'ft' }],
  mass: [{ value: 'kg', label: 'kg' }, { value: 'g', label: 'g' }, { value: 'lb', label: 'lb' }],
  angle: [{ value: 'degrees', label: 'degrees' }, { value: 'rad', label: 'radians' }],
  time: [{ value: 's', label: 's' }, { value: 'ms', label: 'ms' }],
  duration: [{ value: 'h', label: 'hours' }, { value: 'day', label: 'days' }, { value: 'min', label: 'minutes' }, { value: 's', label: 'seconds' }, { value: 'week', label: 'weeks' }],
}

export const defaultUnit = (dimension: string | null): string => (dimension && UNITS[dimension]?.[0].value) || ''

/** A duration in seconds shown in the biggest unit that keeps the number at 1 or more (259200 s -> 3 days). */
export function niceDuration(seconds: number): { value: number; unit: string } {
  const steps: [string, number][] = [['week', 604800], ['day', 86400], ['h', 3600], ['min', 60]]
  const hit = steps.find(([, f]) => seconds >= f)
  return hit ? { value: seconds / hit[1], unit: hit[0] } : { value: seconds, unit: 's' }
}

export type Parsed = { ok: true; value: number } | { ok: false; error: string }

/** "0.5", "50", " 1,5 " -> a positive number (a comma is accepted as the decimal mark). */
export function parseNumber(text: string): Parsed {
  const t = text.trim().replace(',', '.')
  if (!t) return { ok: false, error: 'Enter a number.' }
  if (!/^\d*\.?\d+(e[+-]?\d+)?$/i.test(t)) return { ok: false, error: 'That is not a number.' }
  const v = Number(t)
  if (!Number.isFinite(v) || v <= 0) return { ok: false, error: 'Enter a number greater than zero.' }
  return { ok: true, value: v }
}
