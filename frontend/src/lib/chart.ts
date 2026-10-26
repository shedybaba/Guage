import type { Fact } from '../types'

export const COLORS = { ink: '#17211f', accent: '#0e6a60', copper: '#b4561b', slate: '#8a968f', blue: '#2b5d9e', gold: '#a8741a' }

export interface Series { key: string; label: string; color: string; kind: 'line' | 'dots' | 'step' | 'dashed' }
export interface Marker { x: number; label: string; color: string }
export interface Area { x1: number; x2: number; label: string }
export type Row = Record<string, number | null>
export interface ChartSpec {
  title: string
  xLabel: string
  yLabel: string
  data: Row[]
  series: Series[]
  markers: Marker[]
  areas: Area[]
}

const MAX_POINTS = 900

/** Keep a long series light enough to draw smoothly: every n-th row, always keeping the last. */
export function decimate<T>(rows: T[], max = MAX_POINTS): T[] {
  if (rows.length <= max) return rows
  const stride = Math.ceil(rows.length / max)
  const out = rows.filter((_, i) => i % stride === 0)
  if (out[out.length - 1] !== rows[rows.length - 1]) out.push(rows[rows.length - 1])
  return out
}

const arr = (v: unknown): (number | null)[] | null => (Array.isArray(v) ? (v as (number | null)[]) : null)
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const unitsOf = (numbers: Fact[], id: string, fallback = '') => numbers.find((n) => n.id === id)?.units || fallback

function rows(t: (number | null)[], cols: Record<string, (number | null)[] | null>): Row[] {
  const out: Row[] = []
  for (let i = 0; i < t.length; i++) {
    const r: Row = { t: t[i] }
    for (const [k, v] of Object.entries(cols)) if (v) r[k] = num(v[i])
    out.push(r)
  }
  return out
}

/** The charts for a result, chosen by tool. An empty list when the result carries no series. */
export function chartSpecs(tool: string, chart: Record<string, unknown> | undefined, numbers: Fact[] = []): ChartSpec[] {
  const t = arr(chart?.t)
  if (!chart || !t || t.length < 2) return []
  const used = (id: string) => num(numbers.find((n) => n.id === id)?.value)

  if (tool === 'pendulum_period' || tool === 'oscillation_frequency') {
    const from = used('used_from'), to = used('used_to')
    const series: Series[] = [
      { key: 'measured', label: 'measured position', color: COLORS.slate, kind: 'dots' },
      { key: 'fitted', label: 'fitted wave', color: COLORS.accent, kind: 'line' },
    ]
    const cols: Record<string, (number | null)[] | null> = { measured: arr(chart.measured), fitted: arr(chart.fitted) }
    if (tool === 'oscillation_frequency' && arr(chart.envelope_upper)) {
      cols.upper = arr(chart.envelope_upper); cols.lower = arr(chart.envelope_lower)
      series.push({ key: 'upper', label: 'amplitude envelope', color: COLORS.copper, kind: 'dashed' },
        { key: 'lower', label: '', color: COLORS.copper, kind: 'dashed' })
    }
    return [{
      title: 'Position over time, with the fitted wave', xLabel: 'time (s)', yLabel: 'position (pixels)',
      data: decimate(rows(t, cols)), series, markers: [],
      areas: from != null && to != null ? [{ x1: from, x2: to, label: 'used for the fit' }] : [],
    }]
  }

  if (tool === 'free_fall') {
    const y = arr(chart.y_px), fit = arr(chart.fitted_y_px)
    if (!y) return []
    const y0 = num(y[0]) ?? 0
    const rel = (a: (number | null)[] | null) => (a ? a.map((v) => (v == null ? null : v - y0)) : null)
    return [{
      title: 'Distance fallen over time', xLabel: 'time (s)', yLabel: 'distance below the start (pixels)',
      data: decimate(rows(t, { measured: rel(y), fitted: rel(fit) })),
      series: [{ key: 'measured', label: 'measured', color: COLORS.slate, kind: 'dots' },
        { key: 'fitted', label: 'constant-acceleration fit', color: COLORS.accent, kind: 'line' }],
      markers: [], areas: [],
    }]
  }

  if (tool === 'velocity_acceleration') {
    const sp = unitsOf(numbers, 'peak_speed', 'speed')
    const acc = unitsOf(numbers, 'acceleration', '')
    const out: ChartSpec[] = [{
      title: 'Speed over time', xLabel: 'time (s)', yLabel: `speed (${sp})`,
      data: decimate(rows(t, { speed: arr(chart.speed), fitted: arr(chart.fitted_speed) })),
      series: [{ key: 'speed', label: 'speed', color: COLORS.ink, kind: 'line' },
        { key: 'fitted', label: 'steady-acceleration fit', color: COLORS.accent, kind: 'dashed' }],
      markers: [], areas: [],
    }]
    if (arr(chart.tangential_acceleration)) {
      out.push({
        title: 'Acceleration along the path', xLabel: 'time (s)', yLabel: `acceleration (${acc})`,
        data: decimate(rows(t, { accel: arr(chart.tangential_acceleration) })),
        series: [{ key: 'accel', label: 'acceleration', color: COLORS.copper, kind: 'line' }], markers: [], areas: [],
      })
    }
    return out
  }

  if (tool === 'collision_momentum') {
    const speeds = (chart.speeds ?? {}) as Record<string, (number | null)[]>
    const peak = (v: (number | null)[]) => Math.max(0, ...v.map((x) => x ?? 0))
    const top = Object.entries(speeds).sort((a, b) => peak(b[1]) - peak(a[1])).slice(0, 3)
    if (!top.length) return []
    const palette = [COLORS.accent, COLORS.copper, COLORS.blue]
    const cols: Record<string, (number | null)[]> = {}
    top.forEach(([id, v]) => { cols[`s${id}`] = v })
    const et = (arr(chart.event_t) ?? []) as number[]
    const ev = (chart.event_valid ?? []) as boolean[]
    return [{
      title: 'Speeds of the moving objects, with each impact', xLabel: 'time (s)', yLabel: 'speed (pixels/s, camera motion removed)',
      data: decimate(rows(t, cols)),
      series: top.map(([id], i) => ({ key: `s${id}`, label: `object ${id}`, color: palette[i], kind: 'line' as const })),
      markers: et.map((x, i) => ({ x, label: '', color: ev[i] ? COLORS.accent : COLORS.slate })), areas: [],
    }]
  }

  if (tool === 'count_objects' || tool === 'line_crossings') {
    const cum = arr(chart.cumulative)
    if (!cum) return []
    const crossing = tool === 'line_crossings'
    return [{
      title: crossing ? 'Crossings so far' : 'Objects seen so far',
      xLabel: crossing ? 'time of the crossing (s)' : 'time the object first appears (s)',
      yLabel: crossing ? 'crossings' : 'objects',
      data: rows(t, { count: cum }),
      series: [{ key: 'count', label: crossing ? 'crossings' : 'objects', color: COLORS.accent, kind: 'step' }],
      markers: [], areas: [],
    }]
  }
  return []
}
