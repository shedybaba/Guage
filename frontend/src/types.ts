// The shapes the Gauge API returns (see backend/app/main.py and jobs.py).

export type JobStatus =
  | 'queued' | 'planning' | 'tracking' | 'measuring' | 'explaining'
  | 'done' | 'needs_input' | 'failed'

export const ACTIVE_STATUSES: JobStatus[] = ['queued', 'planning', 'tracking', 'measuring', 'explaining']

export interface Fact {
  id: string
  label: string
  display: string
  value: number | null
  uncertainty: number | null
  units: string
}

export interface CheckEntry { step: string; message: string; ok: boolean | null }

export interface Explanation {
  summary: string
  how_measured: string[]
  caveats: string[]
  source: 'model' | 'fallback'
  attempts: number
  rejected_numbers: string[]
  reason: string
}

export interface Usage {
  sam_frames_billed: number
  sam_cost_usd: number
  llm_calls: number
  llm_tokens: number
  llm_cost_usd: number | null
}

export interface MeasurementResult {
  kind: 'measurement'
  answerable: true
  tool: string
  question: string
  interpretation: string
  headline: Fact
  numbers: Fact[]
  inputs: { label: string; display: string }[]
  statements: string[]
  explanation: Explanation
  warnings: string[]
  notes: string[]
  checks: CheckEntry[]
  chart: Record<string, unknown>
  tracks: { url: string; objects: string[]; phrase: string; video_url: string }
  downloads: { csv: string }
  usage: Usage
}

export interface UnsupportedResult {
  kind: 'unsupported'
  answerable: false
  question: string
  reason: string
  can_measure: string[]
  usage: Usage
}

export interface NoResult {
  kind: 'no_result'
  answerable: false
  question: string
  interpretation?: string
  reason: string
  advice: string[]
  warnings?: string[]
  checks: CheckEntry[]
  usage: Usage
}

export type Result = MeasurementResult | UnsupportedResult | NoResult

export interface NeedMissing { name: string; question: string; dimension: string | null }
export interface NeedConfirm {
  name: string; said: string; value: number; unit: string; reason: string; question: string
  dimension?: string
}
export interface Needs {
  missing: NeedMissing[]
  confirm: NeedConfirm[]
  notes: string[]
  interpretation: string
}

export interface PlanSummary {
  tool: string | null
  outputs: string[]
  interpretation: string
  targets: { role: string; phrases: string[] }[]
  notes: string[]
}

export interface Job {
  job_id: string
  video_id: string
  parent_id: string | null
  question: string
  status: JobStatus
  progress: number
  message: string | null
  created: number
  updated: number
  plan: PlanSummary | null
  result: Result | null
  needs: Needs | null
  error: string | null
}

export interface JobEvent {
  id: number
  ts: number
  kind: 'status' | 'plan' | 'progress' | 'log' | 'needs' | 'result' | 'error'
  message: string
  data: Record<string, unknown> | null
}

export interface VideoInfo {
  video_id: string
  fps: number
  frame_count: number
  width: number
  height: number
  duration: number
  filename?: string
  questions?: { job_id: string; question: string; status: JobStatus }[]
}

export interface TrackObject {
  frames: number[]
  t: number[]
  box: number[][]
  centre: number[][]
  confidence: (number | null)[]
}

export interface Tracks {
  width: number
  height: number
  fps: number
  frame_count: number
  objects: Record<string, TrackObject>
}
