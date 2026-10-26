import { useCallback, useEffect, useRef, useState } from 'react'
import { transcribe } from '../api'

export const MAX_SECONDS = 30
const MIN_SECONDS = 0.6
const PREFERRED = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']

export type VoiceState = 'idle' | 'recording' | 'transcribing'

/** Browsers differ: Chrome and Firefox record webm or ogg, Safari records mp4. The server converts whatever arrives. */
export function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') return undefined
  return PREFERRED.find((t) => MediaRecorder.isTypeSupported(t))
}

const extension = (mime: string) => (mime.includes('mp4') ? 'm4a' : mime.includes('ogg') ? 'ogg' : 'webm')

export function micErrorMessage(e: unknown): string {
  const name = e instanceof DOMException ? e.name : ''
  if (name === 'NotAllowedError' || name === 'SecurityError')
    return 'Microphone access is blocked. Allow it in your browser’s address bar, or type your question.'
  if (name === 'NotFoundError' || name === 'OverconstrainedError')
    return 'No microphone was found. Plug one in, or type your question.'
  if (name === 'NotReadableError') return 'The microphone is being used by another app. Close it and try again, or type your question.'
  return e instanceof Error && e.message ? e.message : 'Could not use the microphone. You can type your question instead.'
}

/** Record a spoken question and turn it into text. The text goes to `onText`; nothing is asked automatically. */
export function useVoice(onText: (text: string) => void) {
  const supported = typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== 'undefined'
  const [state, setState] = useState<VoiceState>('idle')
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)
  const started = useRef(0)
  const discard = useRef(false)
  const mounted = useRef(true)
  const onTextRef = useRef(onText)
  onTextRef.current = onText

  const release = useCallback(() => {
    if (timer.current) clearInterval(timer.current)
    timer.current = null
    stream.current?.getTracks().forEach((t) => t.stop())
    stream.current = null
  }, [])

  const stop = useCallback(() => {
    if (recorder.current && recorder.current.state !== 'inactive') recorder.current.stop()
  }, [])

  const cancel = useCallback(() => {
    discard.current = true
    stop()
    release()
    setState('idle')
  }, [release, stop])

  const start = useCallback(async () => {
    if (!supported || state !== 'idle') return
    setError(null)
    let media: MediaStream
    try {
      media = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
    } catch (e) {
      if (mounted.current) setError(micErrorMessage(e))
      return
    }
    if (!mounted.current) { media.getTracks().forEach((t) => t.stop()); return }
    stream.current = media
    const mime = pickMimeType()
    const rec = mime ? new MediaRecorder(media, { mimeType: mime }) : new MediaRecorder(media)
    const chunks: Blob[] = []
    discard.current = false
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data) }
    rec.onstop = async () => {
      const length = (Date.now() - started.current) / 1000
      release()
      if (discard.current || !mounted.current) return
      if (length < MIN_SECONDS) { setState('idle'); setError('That was too short. Hold on a little longer while you speak.'); return }
      setState('transcribing')
      const type = rec.mimeType || mime || 'audio/webm'
      try {
        const { text } = await transcribe(new Blob(chunks, { type }), `question.${extension(type)}`)
        if (mounted.current && !discard.current) onTextRef.current(text)
      } catch (e) {
        if (mounted.current) setError(e instanceof Error ? e.message : 'Could not transcribe that. You can type your question instead.')
      } finally {
        if (mounted.current) setState('idle')
      }
    }
    recorder.current = rec
    started.current = Date.now()
    setSeconds(0)
    rec.start()
    setState('recording')
    timer.current = setInterval(() => {
      const s = (Date.now() - started.current) / 1000
      setSeconds(Math.min(s, MAX_SECONDS))
      if (s >= MAX_SECONDS) stop()
    }, 200)
  }, [release, state, stop, supported])

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      discard.current = true
      if (recorder.current && recorder.current.state !== 'inactive') recorder.current.stop()
      release()
    }
  }, [release])

  return { supported, state, seconds, error, start, stop, cancel }
}
