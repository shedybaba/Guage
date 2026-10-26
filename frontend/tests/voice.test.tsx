import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../src/api', () => ({ transcribe: vi.fn() }))

import * as api from '../src/api'
import { QuestionBox } from '../src/components/QuestionBox'
import { pickMimeType, micErrorMessage } from '../src/hooks/useVoice'

const transcribe = vi.mocked(api.transcribe)

let recorders: FakeRecorder[]
let stopped: number
class FakeRecorder {
  static isTypeSupported = (t: string) => t === 'audio/webm;codecs=opus'
  state: 'inactive' | 'recording' = 'inactive'
  mimeType: string
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  constructor(_s: MediaStream, opts?: { mimeType?: string }) { this.mimeType = opts?.mimeType ?? ''; recorders.push(this) }
  start() { this.state = 'recording' }
  stop() {
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['abc'], { type: this.mimeType }) })
    this.onstop?.()
  }
}

function useMicrophone(result: 'ok' | Error = 'ok') {
  const track = { stop: () => { stopped++ } }
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
    getUserMedia: vi.fn(() => (result === 'ok' ? Promise.resolve({ getTracks: () => [track] }) : Promise.reject(result))),
  } })
}

beforeEach(() => {
  recorders = []
  stopped = 0
  vi.stubGlobal('MediaRecorder', FakeRecorder)
  transcribe.mockReset()
})
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

const setup = () => render(<QuestionBox onAsk={vi.fn().mockResolvedValue(undefined)} busy={false} hasHistory={false} />)
const micButton = () => screen.getByRole('button', { name: 'Speak your question' })

/** Let the recording run for a while: the code measures real time, so the clock is faked. */
async function speakFor(seconds: number) {
  await userEvent.click(micButton())
  await screen.findByRole('button', { name: 'Stop recording' })
  await act(async () => { vi.setSystemTime(Date.now() + seconds * 1000) })
}

describe('voice input', () => {
  it('records, sends the audio, and puts the words in the box without asking', async () => {
    useMicrophone()
    transcribe.mockResolvedValue({ text: 'What is the swing period?', seconds: 3 })
    const onAsk = vi.fn().mockResolvedValue(undefined)
    render(<QuestionBox onAsk={onAsk} busy={false} hasHistory={false} />)
    vi.useFakeTimers({ toFake: ['Date'] })
    await userEvent.setup({ advanceTimers: () => {} }).click(micButton())
    await screen.findByRole('button', { name: 'Stop recording' })
    expect(screen.getByRole('textbox')).toHaveAttribute('readonly')
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
    await act(async () => { vi.setSystemTime(Date.now() + 3000) })
    await userEvent.setup({ advanceTimers: () => {} }).click(screen.getByRole('button', { name: 'Stop recording' }))
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue('What is the swing period?'))
    expect(onAsk).not.toHaveBeenCalled() // a mis-heard number must be checked by the person first
    const [blob, name] = transcribe.mock.calls[0]
    expect(blob.size).toBeGreaterThan(0)
    expect(name).toBe('question.webm')
    expect(screen.getByText(/Check the numbers and units/)).toBeInTheDocument()
    expect(stopped).toBe(1) // the microphone is released
  })

  it('adds spoken words after what was already typed', async () => {
    useMicrophone()
    transcribe.mockResolvedValue({ text: 'The string is half a metre.', seconds: 3 })
    vi.useFakeTimers({ toFake: ['Date'] })
    const user = userEvent.setup({ advanceTimers: () => {} })
    setup()
    await user.type(screen.getByRole('textbox'), 'What value of g?')
    await speakFor(2)
    await user.click(screen.getByRole('button', { name: 'Stop recording' }))
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue('What value of g? The string is half a metre.'))
  })

  it('discards a recording that is cancelled', async () => {
    useMicrophone()
    vi.useFakeTimers({ toFake: ['Date'] })
    const user = userEvent.setup({ advanceTimers: () => {} })
    setup()
    await speakFor(2)
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(transcribe).not.toHaveBeenCalled()
    expect(micButton()).toBeEnabled()
    expect(stopped).toBe(1)
  })

  it('does not send a recording that is too short', async () => {
    useMicrophone()
    const user = userEvent.setup()
    setup()
    await user.click(micButton())
    await user.click(await screen.findByRole('button', { name: 'Stop recording' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('too short')
    expect(transcribe).not.toHaveBeenCalled()
  })

  it('explains a blocked microphone and keeps typing available', async () => {
    useMicrophone(new DOMException('no', 'NotAllowedError'))
    const user = userEvent.setup()
    setup()
    await user.click(micButton())
    expect(await screen.findByRole('alert')).toHaveTextContent('blocked')
    await user.type(screen.getByRole('textbox'), 'swing period')
    expect(screen.getByRole('textbox')).toHaveValue('swing period')
  })

  it("shows the server's message when nothing was heard, and the box stays as it was", async () => {
    useMicrophone()
    transcribe.mockRejectedValue(new Error('No speech was heard. Please try again.'))
    vi.useFakeTimers({ toFake: ['Date'] })
    const user = userEvent.setup({ advanceTimers: () => {} })
    setup()
    await speakFor(2)
    await user.click(screen.getByRole('button', { name: 'Stop recording' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('No speech was heard')
    expect(screen.getByRole('textbox')).toHaveValue('')
    expect(micButton()).toBeEnabled()
  })

  it('stops by itself at 30 seconds', async () => {
    useMicrophone()
    transcribe.mockResolvedValue({ text: 'a long question', seconds: 30 })
    vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] })
    const user = userEvent.setup({ advanceTimers: () => {} })
    setup()
    await user.click(micButton())
    await screen.findByRole('button', { name: 'Stop recording' })
    expect(screen.getByText(/Listening 0:00 \/ 0:30/)).toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(30500) })
    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue('a long question'))
  })

  it('hides the microphone button where the browser cannot record', () => {
    vi.stubGlobal('MediaRecorder', undefined)
    setup()
    expect(screen.queryByRole('button', { name: 'Speak your question' })).not.toBeInTheDocument()
  })

  it('releases the microphone if the page is left while recording', async () => {
    useMicrophone()
    const user = userEvent.setup()
    const { unmount } = setup()
    await user.click(micButton())
    await screen.findByRole('button', { name: 'Stop recording' })
    unmount()
    expect(stopped).toBe(1)
    expect(transcribe).not.toHaveBeenCalled()
  })
})

describe('voice helpers', () => {
  it('prefers webm with opus, and falls back to what the browser supports', () => {
    expect(pickMimeType()).toBe('audio/webm;codecs=opus')
    vi.stubGlobal('MediaRecorder', class { static isTypeSupported = (t: string) => t === 'audio/mp4' })
    expect(pickMimeType()).toBe('audio/mp4')
    vi.stubGlobal('MediaRecorder', class { static isTypeSupported = () => false })
    expect(pickMimeType()).toBeUndefined()
  })
  it('turns microphone errors into plain advice', () => {
    expect(micErrorMessage(new DOMException('', 'NotAllowedError'))).toMatch(/blocked/)
    expect(micErrorMessage(new DOMException('', 'NotFoundError'))).toMatch(/No microphone/)
    expect(micErrorMessage(new DOMException('', 'NotReadableError'))).toMatch(/another app/)
    expect(micErrorMessage(null)).toMatch(/type your question/)
  })
})
