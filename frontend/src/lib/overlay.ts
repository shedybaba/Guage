import { objectColor } from '@meta-sam/graphics'
import type { TrackObject, Tracks } from '../types'

export interface Rect { x: number; y: number; w: number; h: number; scale: number }

/** Where a video of size (vw, vh) lands inside a container of size (cw, ch) with object-fit: contain. */
export function containFit(cw: number, ch: number, vw: number, vh: number): Rect {
  if (!cw || !ch || !vw || !vh) return { x: 0, y: 0, w: cw, h: ch, scale: 1 }
  const scale = Math.min(cw / vw, ch / vh)
  const w = vw * scale
  const h = vh * scale
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h, scale }
}

/** Index of the entry in a sorted time list nearest to `time`; -1 if the list is empty. */
export function nearestIndex(times: number[], time: number): number {
  if (!times.length) return -1
  let lo = 0
  let hi = times.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (times[mid] < time) lo = mid + 1
    else hi = mid
  }
  if (lo > 0 && Math.abs(times[lo - 1] - time) <= Math.abs(times[lo] - time)) return lo - 1
  return lo
}

/** The object's entry at `time`, but only if it is really visible then (within about 1.5 frames). */
export function entryAt(obj: TrackObject, time: number, fps: number): number {
  const i = nearestIndex(obj.t, time)
  if (i < 0) return -1
  return Math.abs(obj.t[i] - time) <= 1.5 / Math.max(fps, 1) ? i : -1
}

export interface OverlayOptions { trailSeconds: number; showTrails: boolean; showBoxes: boolean; showLabels: boolean }
export const defaultOverlay: OverlayOptions = { trailSeconds: 2, showTrails: true, showBoxes: true, showLabels: true }

/** Draw Gauge's final tracks (boxes, numbers, motion paths) for the frame at `time`. Coordinates are in video pixels. */
export function drawOverlay(
  ctx: CanvasRenderingContext2D, tracks: Tracks, time: number, rect: Rect, opts: OverlayOptions = defaultOverlay,
  dpr = 1,
) {
  ctx.save()
  ctx.scale(dpr, dpr)
  ctx.translate(rect.x, rect.y)
  ctx.scale(rect.scale, rect.scale)
  const px = 1 / rect.scale // one screen pixel in video units
  for (const [id, obj] of Object.entries(tracks.objects)) {
    const color = objectColor(id)
    const i = entryAt(obj, time, tracks.fps)
    if (opts.showTrails) {
      // the path the object took over the last few seconds, fading towards the past
      const end = nearestIndex(obj.t, time)
      let start = end
      while (start > 0 && time - obj.t[start - 1] <= opts.trailSeconds) start--
      if (end - start >= 1) {
        for (let k = start + 1; k <= end; k++) {
          if (obj.t[k] > time + 1e-6) break
          const age = (time - obj.t[k]) / opts.trailSeconds
          ctx.globalAlpha = Math.max(0.1, 0.9 * (1 - age))
          ctx.strokeStyle = color
          ctx.lineWidth = 2.5 * px
          ctx.lineCap = 'round'
          ctx.beginPath()
          ctx.moveTo(obj.centre[k - 1][0], obj.centre[k - 1][1])
          ctx.lineTo(obj.centre[k][0], obj.centre[k][1])
          ctx.stroke()
        }
        ctx.globalAlpha = 1
      }
    }
    if (i < 0) continue
    const [x1, y1, x2, y2] = obj.box[i]
    if (opts.showBoxes) {
      ctx.strokeStyle = color
      ctx.lineWidth = 2 * px
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    }
    if (opts.showLabels) {
      const size = 13 * px
      ctx.font = `600 ${size}px "JetBrains Mono Variable", ui-monospace, monospace`
      const text = id
      const w = ctx.measureText(text).width + 8 * px
      const h = size + 6 * px
      const ly = y1 - h - 1 * px < 0 ? y2 + 1 * px : y1 - h - 1 * px
      ctx.fillStyle = color
      ctx.fillRect(x1, ly, w, h)
      ctx.fillStyle = '#ffffff'
      ctx.textBaseline = 'middle'
      ctx.fillText(text, x1 + 4 * px, ly + h / 2 + 0.5 * px)
    }
  }
  ctx.restore()
}
