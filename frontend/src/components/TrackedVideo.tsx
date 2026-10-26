import { Component, useEffect, useRef, useState, type ReactNode } from 'react'
import { Video, type VideoRef } from '@meta-sam/react'
import { parseVideoStream, type VideoSegmentationResult } from '@meta-sam/parser'
import { useSize } from '../hooks/useSize'
import { formatClock } from '../lib/format'
import { containFit, defaultOverlay, drawOverlay, type OverlayOptions } from '../lib/overlay'
import type { Tracks, VideoInfo } from '../types'

// ---- SAM's own output (with masks), parsed once per URL
const segCache = new Map<string, Promise<VideoSegmentationResult | null>>()
export function loadSegmentation(url: string): Promise<VideoSegmentationResult | null> {
  let p = segCache.get(url)
  if (!p) {
    p = (async () => {
      const r = await fetch(url)
      if (!r.ok) return null
      const text = await r.text()
      async function* events() {
        yield { type: 'response.output_text.delta', item_id: 'm', output_index: 0, content_index: 0, delta: text }
        yield { type: 'response.output_text.done', item_id: 'm', output_index: 0, content_index: 0, text }
        yield { type: 'response.completed' }
      }
      const parsed = parseVideoStream(events() as never)
      return await parsed.finalResult
    })().catch(() => null)
    segCache.set(url, p)
  }
  return p
}

class Boundary extends Component<{ onError: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  componentDidCatch() { this.props.onError() }
  render() { return this.state.failed ? null : this.props.children }
}

interface Props {
  src: string
  info: Pick<VideoInfo, 'width' | 'height' | 'fps' | 'duration'>
  tracks: Tracks | null
  rawUrl: string | null
}

const supportsWebCodecs = () => typeof VideoDecoder !== 'undefined'

export function TrackedVideo({ src, info, tracks, rawUrl }: Props) {
  const { ref: box, size } = useSize<HTMLDivElement>()
  const canvas = useRef<HTMLCanvasElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const meta = useRef<VideoRef>(null)
  const [seg, setSeg] = useState<VideoSegmentationResult | null>(null)
  const [segState, setSegState] = useState<'none' | 'loading' | 'ready'>('none')
  const [metaFailed, setMetaFailed] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(info.duration)
  const [playing, setPlaying] = useState(false)
  const [opts, setOpts] = useState<OverlayOptions>(defaultOverlay)

  // SAM's masks, when stored and the browser can decode video itself (WebCodecs)
  useEffect(() => {
    let cancelled = false
    setSeg(null)
    setMetaFailed(false)
    if (!rawUrl || !supportsWebCodecs()) { setSegState('none'); return }
    setSegState('loading')
    loadSegmentation(rawUrl).then((r) => {
      if (cancelled) return
      setSeg(r)
      setSegState(r ? 'ready' : 'none')
    })
    return () => { cancelled = true }
  }, [rawUrl])

  const useMeta = segState === 'ready' && seg !== null && !metaFailed

  // the native player: follow the clock exactly, frame by frame while it plays
  useEffect(() => {
    const v = video.current
    if (useMeta || !v) return
    const sync = () => { setTime(v.currentTime); if (Number.isFinite(v.duration)) setDuration(v.duration) }
    const onPlay = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    v.addEventListener('timeupdate', sync); v.addEventListener('seeked', sync); v.addEventListener('loadedmetadata', sync)
    v.addEventListener('play', onPlay); v.addEventListener('pause', onPause)
    return () => {
      v.removeEventListener('timeupdate', sync); v.removeEventListener('seeked', sync); v.removeEventListener('loadedmetadata', sync)
      v.removeEventListener('play', onPlay); v.removeEventListener('pause', onPause)
    }
  }, [useMeta, src])

  useEffect(() => {
    const v = video.current
    if (useMeta || !v || !playing) return
    const hasFrameCallback = 'requestVideoFrameCallback' in v
    let id = 0
    let alive = true
    const loop = () => {
      if (!alive) return
      setTime(v.currentTime)
      id = hasFrameCallback ? v.requestVideoFrameCallback(loop) : requestAnimationFrame(loop)
    }
    loop()
    return () => {
      alive = false
      if (hasFrameCallback) v.cancelVideoFrameCallback(id)
      else cancelAnimationFrame(id)
    }
  }, [playing, useMeta])

  // the overlay canvas: sized to the box, then redrawn for the current time
  useEffect(() => {
    const c = canvas.current
    if (!c || !size.w) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.round(size.w * dpr)
    c.height = Math.round(size.h * dpr)
  }, [size])
  useEffect(() => {
    const c = canvas.current
    if (!c || !size.w) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, c.width, c.height)
    if (tracks) drawOverlay(ctx, tracks, time, containFit(size.w, size.h, info.width, info.height), opts, window.devicePixelRatio || 1)
  }, [time, size, tracks, opts, info.width, info.height])

  const play = () => (useMeta ? meta.current?.play() : video.current?.play())
  const pause = () => (useMeta ? meta.current?.pause() : video.current?.pause())
  const seek = (t: number) => {
    const clamped = Math.min(Math.max(0, t), duration || t)
    if (useMeta) void meta.current?.seek(clamped)
    else if (video.current) video.current.currentTime = clamped
    setTime(clamped)
  }
  const step = (dir: 1 | -1) => {
    pause()
    if (useMeta) void (dir > 0 ? meta.current?.nextFrame() : meta.current?.previousFrame())
    else seek((video.current?.currentTime ?? time) + dir / Math.max(info.fps, 1))
  }
  const frame = Math.round(time * info.fps)

  return (
    <div className="overflow-hidden rounded-xl border border-line bg-card lg:rounded-2xl">
      <div
        ref={box}
        className="relative mx-auto max-h-[32vh] w-full bg-ink lg:max-h-[68vh]"
        style={{ aspectRatio: `${info.width} / ${info.height}` }}
      >
        {useMeta ? (
          <Boundary onError={() => setMetaFailed(true)}>
            <Video
              ref={meta}
              src={src}
              result={seg ?? undefined}
              objectFit="contain"
              loop
              muted
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
              onTimeChange={setTime}
              onFrame={(f) => setTime(f.time)}
              onPlayingChange={setPlaying}
              onDurationChange={setDuration}
              onError={() => setMetaFailed(true)}
            />
          </Boundary>
        ) : (
          <video ref={video} src={src} muted loop playsInline preload="auto" className="absolute inset-0 h-full w-full object-contain" aria-label="The uploaded video" />
        )}
        <canvas ref={canvas} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" />
        {segState === 'loading' && (
          <p className="absolute bottom-2 left-2 rounded bg-ink/80 px-2 py-1 text-xs text-paper">Loading SAM masks…</p>
        )}
      </div>

      <div className="space-y-2 border-t border-line p-3">
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => step(-1)} aria-label="Previous frame" className="rounded-md border border-line px-2 py-1.5 text-sm hover:bg-paper-2">‹</button>
          <button type="button" onClick={() => (playing ? pause() : play())} aria-label={playing ? 'Pause' : 'Play'}
            className="w-20 rounded-md bg-ink px-3 py-1.5 text-sm font-semibold text-paper hover:bg-ink/85">
            {playing ? 'Pause' : 'Play'}
          </button>
          <button type="button" onClick={() => step(1)} aria-label="Next frame" className="rounded-md border border-line px-2 py-1.5 text-sm hover:bg-paper-2">›</button>
          <input
            type="range" min={0} max={duration || 1} step={0.01} value={Math.min(time, duration || 1)}
            onChange={(e) => seek(Number(e.target.value))} aria-label="Position in the video" className="mx-1 h-2 flex-1 accent-accent"
          />
          <span className="num whitespace-nowrap text-xs text-ink-soft">{formatClock(time)} · frame {frame}</span>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-soft">
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={opts.showTrails} onChange={(e) => setOpts({ ...opts, showTrails: e.target.checked })} className="accent-accent" disabled={!tracks} /> Paths
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={opts.showBoxes} onChange={(e) => setOpts({ ...opts, showBoxes: e.target.checked, showLabels: e.target.checked })} className="accent-accent" disabled={!tracks} /> Boxes and numbers
          </label>
          <span className="ml-auto">
            {!tracks ? 'Ask a question to see what Gauge tracked.'
              : useMeta ? 'Coloured masks: SAM 3.1. Numbers and paths: Gauge’s final tracks.'
                : 'Numbers, boxes and paths: Gauge’s final tracks.'}
          </span>
        </div>
      </div>
    </div>
  )
}
