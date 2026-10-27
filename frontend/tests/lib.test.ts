import { describe, expect, it } from 'vitest'
import { applyEvent, emptyStream, parseSse } from '../src/lib/events'
import { containFit, entryAt, nearestIndex } from '../src/lib/overlay'
import { chartSpecs, decimate } from '../src/lib/chart'
import { defaultUnit, niceDuration, parseNumber } from '../src/lib/answers'
import { formatBytes, formatClock, formatDuration, formatUsd, stageIndex } from '../src/lib/format'
import type { Fact, JobEvent, TrackObject } from '../src/types'

const ev = (id: number, kind: JobEvent['kind'], message = '', data: JobEvent['data'] = null): JobEvent => ({ id, ts: id, kind, message, data })

describe('event stream reducer', () => {
  it('follows the stages and the frame counter', () => {
    let s = emptyStream()
    s = applyEvent(s, ev(1, 'status', 'Reading your question and looking at the video'))
    expect(s.status).toBe('planning')
    s = applyEvent(s, ev(2, 'plan', 'You want the swing period.'))
    expect(s.interpretation).toBe('You want the swing period.')
    s = applyEvent(s, ev(3, 'status', 'Finding and following the object in every frame'))
    expect(s.status).toBe('tracking')
    s = applyEvent(s, ev(4, 'progress', "Tracking 'glass marble': frame 120 of 362", { done: 120, total: 362 }))
    expect(s.frames).toEqual({ done: 120, total: 362 })
    expect(s.progress).toBeCloseTo(0.1 + 0.55 * (120 / 362))
  })
  it('keeps the log with its ok flags', () => {
    let s = emptyStream()
    s = applyEvent(s, ev(1, 'log', "No match found for 'pendulum bob'.", { step: 'track', ok: false }))
    s = applyEvent(s, ev(2, 'log', 'Checks passed.', { step: 'check', ok: true }))
    expect(s.log.map((l) => l.ok)).toEqual([false, true])
  })
  it('ignores events it has already seen (a reconnect replays them)', () => {
    let s = emptyStream()
    s = applyEvent(s, ev(5, 'log', 'a', { ok: true }))
    const again = applyEvent(s, ev(5, 'log', 'a', { ok: true }))
    expect(again).toBe(s)
    expect(applyEvent(s, ev(3, 'log', 'older')).events).toHaveLength(1)
  })
  it('never lets progress go backwards', () => {
    let s = applyEvent(emptyStream(), ev(1, 'progress', 'x', { done: 300, total: 362 }))
    s = applyEvent(s, ev(2, 'progress', 'y', { done: 10, total: 362 }))
    expect(s.progress).toBeCloseTo(0.1 + 0.55 * (300 / 362))
  })
  it('parses server-sent event text', () => {
    const out = parseSse('id: 1\nevent: status\ndata: {"a":1}\n\n: keep-alive\n\nevent: end\ndata: {"status":"done"}\n\n')
    expect(out).toEqual([{ event: 'status', data: '{"a":1}', id: '1' }, { event: 'end', data: '{"status":"done"}', id: undefined }])
  })
})

describe('overlay geometry', () => {
  it('fits a tall video into a wide box without stretching', () => {
    const r = containFit(800, 400, 720, 1280)
    expect(r.h).toBeCloseTo(400)
    expect(r.w).toBeCloseTo(225)
    expect(r.x).toBeCloseTo((800 - 225) / 2)
    expect(r.scale).toBeCloseTo(400 / 1280)
  })
  it('fits a wide video into a tall box', () => {
    const r = containFit(300, 600, 1920, 1080)
    expect(r.w).toBeCloseTo(300)
    expect(r.y).toBeGreaterThan(0)
  })
  it('finds the nearest time', () => {
    const t = [0, 0.1, 0.2, 0.4]
    expect(nearestIndex(t, 0.16)).toBe(2)
    expect(nearestIndex(t, 0.14)).toBe(1)
    expect(nearestIndex(t, -5)).toBe(0)
    expect(nearestIndex(t, 9)).toBe(3)
    expect(nearestIndex([], 1)).toBe(-1)
  })
  it('hides an object that is not in view at that moment', () => {
    const obj: TrackObject = { frames: [0, 1, 20], t: [0, 1 / 30, 20 / 30], box: [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], centre: [[0, 0], [0, 0], [0, 0]], confidence: [1, 1, 1] }
    expect(entryAt(obj, 1 / 30, 30)).toBe(1)
    expect(entryAt(obj, 10 / 30, 30)).toBe(-1) // lost for a while: nothing is drawn
    expect(entryAt(obj, 20 / 30, 30)).toBe(2)
  })
})

describe('chart specs', () => {
  const t = Array.from({ length: 10 }, (_, i) => i / 10)
  const fact = (id: string, value: number | null, units = ''): Fact => ({ id, label: id, display: String(value), value, uncertainty: null, units })
  it('draws measured vs fitted and shades the part that was used', () => {
    const [c] = chartSpecs('pendulum_period', { t, measured: t.map((x) => x * 2), fitted: t.map((x, i) => (i < 3 ? null : x)) }, [fact('used_from', 0.3), fact('used_to', 0.9)])
    expect(c.series.map((s) => s.key)).toEqual(['measured', 'fitted'])
    expect(c.areas).toEqual([{ x1: 0.3, x2: 0.9, label: 'used for the fit' }])
    expect(c.data[0].fitted).toBeNull()
    expect(c.data[5].fitted).toBe(0.5)
  })
  it('adds the amplitude envelope for a spring', () => {
    const [c] = chartSpecs('oscillation_frequency', { t, measured: t, fitted: t, envelope_upper: t, envelope_lower: t })
    expect(c.series.map((s) => s.key)).toContain('upper')
  })
  it('measures a fall from where it started', () => {
    const [c] = chartSpecs('free_fall', { t, y_px: t.map((x) => 100 + x * 50), fitted_y_px: null })
    expect(c.data[0].measured).toBe(0)
    expect(c.data[9].measured).toBeCloseTo(45)
  })
  it('draws speed and acceleration as two charts', () => {
    const cs = chartSpecs('velocity_acceleration', { t, speed: t, fitted_speed: t, tangential_acceleration: t }, [fact('peak_speed', 1, 'm/s')])
    expect(cs).toHaveLength(2)
    expect(cs[0].yLabel).toBe('speed (m/s)')
  })
  it('shows only the three fastest objects of a collision and marks the impacts', () => {
    const speeds = { a: t.map(() => 1), b: t.map(() => 9), c: t.map(() => 5), d: t.map(() => 0) }
    const [c] = chartSpecs('collision_momentum', { t, speeds, event_t: [0.2, 0.5], event_valid: [true, false] })
    expect(c.series.map((s) => s.label)).toEqual(['object b', 'object c', 'object a'])
    expect(c.markers.map((m) => m.color)).toHaveLength(2)
    expect(c.markers[0].color).not.toBe(c.markers[1].color)
  })
  it('steps the count up', () => {
    const [c] = chartSpecs('count_objects', { t, cumulative: t.map((_, i) => i + 1) })
    expect(c.series[0].kind).toBe('step')
  })
  it('returns nothing when there are no series', () => {
    expect(chartSpecs('pendulum_period', {})).toEqual([])
    expect(chartSpecs('pendulum_period', undefined)).toEqual([])
    expect(chartSpecs('unknown_tool', { t })).toEqual([])
  })
  it('thins a long series but keeps its last point', () => {
    const rows = Array.from({ length: 5000 }, (_, i) => i)
    const d = decimate(rows, 900)
    expect(d.length).toBeLessThanOrEqual(901)
    expect(d[d.length - 1]).toBe(4999)
    expect(decimate([1, 2, 3], 900)).toEqual([1, 2, 3])
  })
})

describe('answers and formatting', () => {
  it('accepts plain numbers and decimal commas', () => {
    expect(parseNumber('0.5')).toEqual({ ok: true, value: 0.5 })
    expect(parseNumber(' 1,5 ')).toEqual({ ok: true, value: 1.5 })
    expect(parseNumber('50')).toEqual({ ok: true, value: 50 })
  })
  it('refuses nonsense', () => {
    for (const bad of ['', 'abc', '0', '-3', '1.2.3', '5 cm']) expect(parseNumber(bad).ok).toBe(false)
  })
  it('picks a first unit for each kind of quantity', () => {
    expect(defaultUnit('length')).toBe('m')
    expect(defaultUnit('mass')).toBe('kg')
    expect(defaultUnit(null)).toBe('')
  })
  it('formats the small things people read', () => {
    expect(formatBytes(2_873_335)).toBe('2.7 MB')
    expect(formatDuration(12.045)).toBe('12.0 s')
    expect(formatDuration(75)).toBe('1 min 15 s')
    expect(formatClock(5.5)).toBe('0:05.50')
    expect(formatUsd(0.000658)).toBe('$0.0007')
    expect(formatUsd(0)).toBe('$0')
    expect(formatUsd(null)).toBe('n/a')
  })
  it('orders the stages', () => {
    expect(['queued', 'planning', 'tracking', 'measuring', 'explaining', 'done'].map((s) => stageIndex(s as never))).toEqual([-1, 0, 1, 2, 3, 4])
  })
})

describe('growth and rotation charts', () => {
  const num = (id: string, value: number) => ({ id, label: id, display: String(value), value, uncertainty: null, units: '' })
  it('draws size over time in real time units when the time-lapse length is known', () => {
    const specs = chartSpecs('growth_over_time', { t: [0, 1, 2], t_unit: 'h', real_time: true, y_label: 'height (cm)',
      measured: [5, 6, 7], fitted: [5, 6, 7], fit_name: 'linear' })
    expect(specs).toHaveLength(1)
    expect(specs[0].xLabel).toBe('real time (h)')
    expect(specs[0].yLabel).toBe('height (cm)')
    expect(specs[0].series.map((s) => s.key)).toEqual(['measured', 'fitted'])
  })
  it('says it is video time when no real time was given', () => {
    const [spec] = chartSpecs('growth_over_time', { t: [0, 1, 2], t_unit: 's', real_time: false, y_label: 'area (px²)', measured: [1, 2, 3], fitted: [1, 2, 3] })
    expect(spec.xLabel).toBe('time in the video (s)')
  })
  it('draws turns over time and shades the stretch that was used', () => {
    const [spec] = chartSpecs('rotation_rate', { t: [0, 1, 2], turns: [0, 2, 4], fitted: [0, 2, 4] }, [num('used_from', 0.5), num('used_to', 1.5)])
    expect(spec.yLabel).toBe('turns (revolutions)')
    expect(spec.areas).toEqual([{ x1: 0.5, x2: 1.5, label: 'used for the fit' }])
  })
})

describe('niceDuration', () => {
  it('shows a duration in the biggest sensible unit', () => {
    expect(niceDuration(259200)).toEqual({ value: 3, unit: 'day' })
    expect(niceDuration(43200)).toEqual({ value: 12, unit: 'h' })
    expect(niceDuration(90)).toEqual({ value: 1.5, unit: 'min' })
    expect(niceDuration(20)).toEqual({ value: 20, unit: 's' })
    expect(niceDuration(1209600)).toEqual({ value: 2, unit: 'week' })
  })
})
