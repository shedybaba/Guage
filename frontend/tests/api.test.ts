import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, ask, followJob, getJob, rawSegmentationUrl, uploadVideo } from '../src/api'

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

describe('requests', () => {
  beforeEach(() => { vi.stubGlobal('fetch', vi.fn()) })
  afterEach(() => { vi.unstubAllGlobals() })

  it("uses the server's own words when it refuses", async () => {
    vi.mocked(fetch).mockResolvedValue(json({ detail: 'Gauge is busy with other questions right now.' }, 429))
    await expect(ask('v', 'q')).rejects.toMatchObject({ message: 'Gauge is busy with other questions right now.', status: 429 })
  })
  it('turns a validation error list into one readable sentence', async () => {
    vi.mocked(fetch).mockResolvedValue(json({ detail: [{ msg: 'String should have at least 1 character' }] }, 422))
    await expect(ask('v', '')).rejects.toThrow('String should have at least 1 character')
  })
  it('copes with an error that is not JSON', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('<html>Bad gateway</html>', { status: 502 }))
    const err = await getJob('x').catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.message).toBe('The server answered 502.')
  })
  it('posts the question and any values it was given', async () => {
    vi.mocked(fetch).mockResolvedValue(json({ job_id: 'j' }, 202))
    await ask('v1', 'What is g?', { string_length_m: 0.5 })
    const [url, init] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('/api/videos/v1/questions')
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({ question: 'What is g?', known_values: { string_length_m: 0.5 } })
  })
  it('builds the address of SAM masks for a phrase, safely encoded', () => {
    expect(rawSegmentationUrl('v1', 'glass marble & ball')).toBe('/api/videos/v1/segmentation?phrase=glass%20marble%20%26%20ball&format=raw')
  })
})

class FakeEventSource {
  static last: FakeEventSource
  listeners: Record<string, ((m: MessageEvent) => void)[]> = {}
  closed = false
  constructor(public url: string) { FakeEventSource.last = this }
  addEventListener(kind: string, fn: (m: MessageEvent) => void) { (this.listeners[kind] ??= []).push(fn) }
  close() { this.closed = true }
  emit(kind: string, data: unknown) { this.listeners[kind]?.forEach((fn) => fn({ data: JSON.stringify(data) } as MessageEvent)) }
}

describe('following a job live', () => {
  beforeEach(() => { vi.stubGlobal('EventSource', FakeEventSource) })
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })

  it('passes each event on once, even if the server repeats it after a reconnect', () => {
    const seen: number[] = []
    followJob('j1', (e) => seen.push(e.id), vi.fn())
    const es = FakeEventSource.last
    es.emit('status', { id: 1, kind: 'status', message: 'a' })
    es.emit('log', { id: 2, kind: 'log', message: 'b' })
    es.emit('log', { id: 2, kind: 'log', message: 'b' }) // replayed
    es.emit('log', { id: 1, kind: 'status', message: 'a' }) // older than what we have
    expect(seen).toEqual([1, 2])
    expect(es.url).toBe('/api/jobs/j1/events')
  })
  it('stops listening and says how it ended when the server closes the story', () => {
    const onEnd = vi.fn()
    followJob('j1', vi.fn(), onEnd)
    const es = FakeEventSource.last
    es.emit('end', { status: 'needs_input' })
    expect(es.closed).toBe(true)
    expect(onEnd).toHaveBeenCalledWith('needs_input')
  })
  it('ignores a damaged event instead of breaking', () => {
    const onEvent = vi.fn()
    followJob('j1', onEvent, vi.fn())
    FakeEventSource.last.listeners.log[0]({ data: '{not json' } as MessageEvent)
    expect(onEvent).not.toHaveBeenCalled()
  })
  it('lets the caller stop, without reporting an end', () => {
    const onEnd = vi.fn()
    const stop = followJob('j1', vi.fn(), onEnd)
    stop()
    FakeEventSource.last.emit('end', { status: 'done' })
    expect(FakeEventSource.last.closed).toBe(true)
    expect(onEnd).not.toHaveBeenCalled()
  })
  it('polls when the browser has no server-sent events', async () => {
    vi.unstubAllGlobals()
    vi.stubGlobal('EventSource', undefined)
    vi.useFakeTimers()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(json({ status: 'tracking' }))
      .mockResolvedValueOnce(json({ status: 'done' }))
    vi.stubGlobal('fetch', fetchMock)
    const onEnd = vi.fn()
    followJob('j1', vi.fn(), onEnd)
    await vi.advanceTimersByTimeAsync(10)
    expect(onEnd).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1600)
    expect(onEnd).toHaveBeenCalledWith('done')
    vi.unstubAllGlobals()
  })
})

describe('upload', () => {
  class FakeXhr {
    static last: FakeXhr
    status = 0; responseText = ''; upload: { onprogress?: (e: ProgressEvent) => void } = {}
    onload?: () => void; onerror?: () => void; onabort?: () => void
    opened: string[] = []; sent: unknown = null; aborted = false
    constructor() { FakeXhr.last = this }
    open(method: string, url: string) { this.opened = [method, url] }
    send(body: unknown) { this.sent = body }
    abort() { this.aborted = true; this.onabort?.() }
  }
  beforeEach(() => { vi.stubGlobal('XMLHttpRequest', FakeXhr) })
  afterEach(() => { vi.unstubAllGlobals() })

  it('reports progress and returns what the server learned', async () => {
    const progress: number[] = []
    const { promise } = uploadVideo(new File(['x'], 'a.mp4'), (f) => progress.push(f))
    const x = FakeXhr.last
    expect(x.opened).toEqual(['POST', '/api/videos'])
    x.upload.onprogress?.({ lengthComputable: true, loaded: 25, total: 100 } as ProgressEvent)
    x.status = 201; x.responseText = JSON.stringify({ video_id: 'v', fps: 30 }); x.onload?.()
    await expect(promise).resolves.toMatchObject({ video_id: 'v' })
    expect(progress).toEqual([0.25])
  })
  it("shows the server's reason for a refusal", async () => {
    const { promise } = uploadVideo(new File(['x'], 'a.mp4'), vi.fn())
    FakeXhr.last.status = 413; FakeXhr.last.responseText = JSON.stringify({ detail: 'That file is larger than the 300 MB limit.' }); FakeXhr.last.onload?.()
    await expect(promise).rejects.toThrow('larger than the 300 MB limit')
  })
  it('explains a lost connection', async () => {
    const { promise } = uploadVideo(new File(['x'], 'a.mp4'), vi.fn())
    FakeXhr.last.onerror?.()
    await expect(promise).rejects.toThrow('Could not reach Gauge')
  })
  it('can be cancelled', async () => {
    const { promise, cancel } = uploadVideo(new File(['x'], 'a.mp4'), vi.fn())
    cancel()
    await expect(promise).rejects.toThrow('cancelled')
    expect(FakeXhr.last.aborted).toBe(true)
  })
})
