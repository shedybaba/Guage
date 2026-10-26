import type { JobStatus } from '../types'

export const STAGES: { key: JobStatus; label: string; hint: string }[] = [
  { key: 'planning', label: 'Understand', hint: 'Reading the question and looking at the video' },
  { key: 'tracking', label: 'Track', hint: 'Finding and following the object in every frame' },
  { key: 'measuring', label: 'Measure', hint: 'Fitting the physics to the tracked positions' },
  { key: 'explaining', label: 'Explain', hint: 'Writing the answer in plain words, number-checked' },
]

/** Which stage (0-based) a status belongs to; -1 before the first, 4 when finished. */
export function stageIndex(status: JobStatus): number {
  if (status === 'queued') return -1
  const i = STAGES.findIndex((s) => s.key === status)
  if (i >= 0) return i
  return status === 'done' ? STAGES.length : -1
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return ''
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const m = Math.floor(seconds / 60)
  return `${m} min ${Math.round(seconds - m * 60)} s`
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export function formatClock(seconds: number): string {
  const s = Math.max(0, seconds)
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`
}

/** "$0.0007" / "under $0.0001" / "n/a" for the small amounts Gauge spends. */
export function formatUsd(x: number | null | undefined): string {
  if (x == null) return 'n/a'
  if (x === 0) return '$0'
  if (x < 0.0001) return 'under $0.0001'
  return `$${x.toFixed(x < 0.01 ? 4 : 2)}`
}

export const TOOL_NAMES: Record<string, string> = {
  pendulum_period: 'Pendulum',
  oscillation_frequency: 'Oscillation',
  free_fall: 'Free fall',
  velocity_acceleration: 'Speed and acceleration',
  collision_momentum: 'Collision',
  count_objects: 'Counting',
  line_crossings: 'Line crossings',
}
