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
  growth_over_time: 'Growth',
  rotation_rate: 'Rotation speed',
}

/** One line of filming advice for the kind of measurement being made (shown while Gauge works). */
export const FILMING_TIPS: Record<string, string> = {
  pendulum_period: 'Let it swing at least five times, with the camera propped still and the whole swing in view.',
  oscillation_frequency: 'Film at least five bounces from the side, with the camera still and the mass fully in view.',
  free_fall: 'Film the drop in slow motion (120 or 240 frames per second) with a ruler or known length in the same plane as the fall.',
  velocity_acceleration: 'Keep a ruler or known length in the same plane as the motion, and use slow motion for fast movement.',
  collision_momentum: 'Use slow motion and a camera that does not move: at 30 frames per second an impact can fall between two frames, and a shaky camera breaks the momentum figures.',
  count_objects: 'Keep the camera still and every object fully visible at some moment; heavy overlap makes any count less sure.',
  line_crossings: 'Keep the camera still. A moving camera drags still objects across the line and they get counted.',
  growth_over_time: 'Keep the camera and the object in place the whole time, and tell Gauge how long the video covers so it can give a rate.',
  rotation_rate: 'Mark the spinning part with a sticker or dot. Fast rotation needs slow motion: from about half a turn per frame, no software can tell the speed.',
}
