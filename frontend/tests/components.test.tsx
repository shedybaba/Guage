import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Job, MeasurementResult, Needs } from '../src/types'
import { emptyStream, applyEvent } from '../src/lib/events'

vi.mock('../src/api', () => ({
  uploadVideo: vi.fn(),
  getJob: vi.fn(),
  answer: vi.fn(),
  followJob: vi.fn(() => () => {}),
  csvUrl: (id: string) => `/api/jobs/${id}/data.csv`,
}))
// charts need real layout, which jsdom does not have; they are checked in a real browser
vi.mock('../src/components/ResultChart', () => ({ ResultChart: ({ spec }: { spec: { title: string } }) => <div data-testid="chart">{spec.title}</div> }))

import * as api from '../src/api'
import { UploadCard } from '../src/components/UploadCard'
import { QuestionBox } from '../src/components/QuestionBox'
import { NeedsForm } from '../src/components/NeedsForm'
import { ProgressPanel } from '../src/components/ProgressPanel'
import { NoAnswerCard, ResultCard } from '../src/components/ResultCard'
import { JobCard } from '../src/components/JobCard'

const mocked = vi.mocked(api)

const usage = { sam_frames_billed: 0, sam_cost_usd: 0, llm_calls: 2, llm_tokens: 5000, llm_cost_usd: 0.000658 }
const measurement = (over: Partial<MeasurementResult> = {}): MeasurementResult => ({
  kind: 'measurement', answerable: true, tool: 'pendulum_period', question: 'What is the swing period?', interpretation: 'Measure the swing period.',
  headline: { id: 'period', label: 'Swing period', display: '0.8582 ± 0.0016 s', value: 0.8582, uncertainty: 0.0016, units: 's' },
  numbers: [
    { id: 'period', label: 'Swing period', display: '0.8582 ± 0.0016 s', value: 0.8582, uncertainty: 0.0016, units: 's' },
    { id: 'frequency', label: 'Frequency', display: '1.165 ± 0.002 Hz', value: 1.165, uncertainty: 0.002, units: 'Hz' },
  ],
  inputs: [{ label: 'string length (you said)', display: '0.5 m' }], statements: [],
  explanation: { summary: 'The swing period is 0.8582 ± 0.0016 s.', how_measured: ['The object was followed in every frame.'], caveats: ['Film a longer steady swing.'], source: 'model', attempts: 1, rejected_numbers: [], reason: '' },
  warnings: ['Used only 4.5 s to 12.0 s of the clip.'], notes: [],
  checks: [{ step: 'track', message: "No match found for 'pendulum bob'.", ok: false }, { step: 'track', message: "Tracked 'glass marble' in 362 of 362 frames.", ok: true }],
  chart: { t: [0, 1, 2], measured: [1, 2, 3], fitted: [1, 2, 3] },
  tracks: { url: '/api/jobs/j1/tracks', objects: ['0'], phrase: 'glass marble', video_url: '/api/videos/v1/video' },
  downloads: { csv: '/api/jobs/j1/data.csv' }, usage, ...over,
})
const job = (over: Partial<Job> = {}): Job => ({
  job_id: 'j1', video_id: 'v1', parent_id: null, question: 'What is the swing period?', status: 'done', progress: 1, message: 'done',
  created: 1, updated: 2, plan: null, result: measurement(), needs: null, error: null, ...over,
})

beforeEach(() => { vi.clearAllMocks(); mocked.followJob.mockReturnValue(() => {}) })

describe('UploadCard', () => {
  it('refuses a file that is not a video, without contacting the server', async () => {
    render(<UploadCard onUploaded={vi.fn()} />)
    await userEvent.upload(screen.getByLabelText('Choose a video file'), new File(['hi'], 'notes.txt', { type: 'text/plain' }), { applyAccept: false })
    expect(screen.getByRole('alert')).toHaveTextContent(/does not look like a video/i)
    expect(mocked.uploadVideo).not.toHaveBeenCalled()
  })
  it('uploads a video and hands over what the server learned', async () => {
    const info = { video_id: 'abc', fps: 30, frame_count: 360, width: 320, height: 240, duration: 12 }
    mocked.uploadVideo.mockReturnValue({ promise: Promise.resolve(info), cancel: vi.fn() })
    const done = vi.fn()
    render(<UploadCard onUploaded={done} />)
    await userEvent.upload(screen.getByLabelText('Choose a video file'), new File(['x'], 'clip.mp4', { type: 'video/mp4' }))
    await waitFor(() => expect(done).toHaveBeenCalledWith(info))
  })
  it("shows the server's reason when it refuses the file, and lets the person try again", async () => {
    mocked.uploadVideo.mockImplementation(() => ({ promise: Promise.reject(new Error('The video is too short to measure anything.')), cancel: vi.fn() }))
    render(<UploadCard onUploaded={vi.fn()} />)
    await userEvent.upload(screen.getByLabelText('Choose a video file'), new File(['x'], 'tiny.mp4', { type: 'video/mp4' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('too short')
    expect(screen.getByRole('button', { name: 'Choose a file' })).toBeInTheDocument()
  })
  it('lists the filming tips', () => {
    render(<UploadCard onUploaded={vi.fn()} />)
    expect(screen.getByText('Keep the phone completely still')).toBeInTheDocument()
  })
})

describe('QuestionBox', () => {
  it('fills the box from an example and sends it', async () => {
    const onAsk = vi.fn().mockResolvedValue(undefined)
    render(<QuestionBox onAsk={onAsk} busy={false} hasHistory={false} />)
    await userEvent.click(screen.getByRole('button', { name: 'What is the swing period?' }))
    expect(screen.getByRole('textbox')).toHaveValue('What is the swing period?')
    await userEvent.click(screen.getByRole('button', { name: 'Ask' }))
    expect(onAsk).toHaveBeenCalledWith('What is the swing period?')
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''))
  })
  it('does not send an empty question', async () => {
    const onAsk = vi.fn()
    render(<QuestionBox onAsk={onAsk} busy={false} hasHistory={false} />)
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
    await userEvent.type(screen.getByRole('textbox'), '   ')
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
  })
  it('sends with Enter and keeps a new line for Shift+Enter', async () => {
    const onAsk = vi.fn().mockResolvedValue(undefined)
    render(<QuestionBox onAsk={onAsk} busy={false} hasHistory={false} />)
    await userEvent.type(screen.getByRole('textbox'), 'How fast?{Enter}')
    expect(onAsk).toHaveBeenCalledWith('How fast?')
  })
  it('is locked while Gauge works, and says why', () => {
    render(<QuestionBox onAsk={vi.fn()} busy disabledReason="Gauge is working on your last question." hasHistory />)
    expect(screen.getByRole('button', { name: 'Working…' })).toBeDisabled()
    expect(screen.getByText('Gauge is working on your last question.')).toBeInTheDocument()
  })
  it("shows the server's refusal (a limit, a bad question) next to the box", async () => {
    render(<QuestionBox onAsk={vi.fn().mockRejectedValue(new Error('Gauge is busy right now.'))} busy={false} hasHistory={false} />)
    await userEvent.type(screen.getByRole('textbox'), 'swing period{Enter}')
    expect(await screen.findByRole('alert')).toHaveTextContent('busy')
    expect(screen.getByRole('textbox')).toHaveValue('swing period') // the text is kept so nothing is lost
  })
  it('folds the example chips away once there is a conversation', () => {
    render(<QuestionBox onAsk={vi.fn()} busy={false} hasHistory />)
    expect(screen.queryByRole('button', { name: 'What is the swing period?' })).not.toBeInTheDocument()
  })
})

describe('NeedsForm', () => {
  const needs = (over: Partial<Needs> = {}): Needs => ({ missing: [{ name: 'string_length', question: 'What is the length of the string?', dimension: 'length' }], confirm: [], notes: [], interpretation: 'You want g.', ...over })
  it('asks for what is missing and sends the answer with its unit', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    render(<NeedsForm needs={needs()} onSubmit={onSubmit} />)
    await userEvent.type(screen.getByLabelText(/length of the string/i), '50')
    await userEvent.selectOptions(screen.getByRole('combobox'), 'cm')
    await userEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect(onSubmit).toHaveBeenCalledWith({ string_length: { value: 50, unit: 'cm' } })
  })
  it('does not send something that is not a number', async () => {
    const onSubmit = vi.fn()
    render(<NeedsForm needs={needs()} onSubmit={onSubmit} />)
    await userEvent.type(screen.getByLabelText(/length of the string/i), 'fifty')
    await userEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect(screen.getByRole('alert')).toHaveTextContent('not a number')
    expect(onSubmit).not.toHaveBeenCalled()
  })
  it('accepts a decimal comma', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    render(<NeedsForm needs={needs()} onSubmit={onSubmit} />)
    await userEvent.type(screen.getByLabelText(/length of the string/i), '0,5')
    await userEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect(onSubmit).toHaveBeenCalledWith({ string_length: { value: 0.5, unit: 'm' } })
  })
  it('puts the number Gauge heard in front of the person to confirm, in the unit it used', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    const n = needs({ missing: [], confirm: [{ name: 'string_length', said: 'about a forearm', value: 0.3, unit: 'm', reason: 'Gauge could not re-read this number from your words; please check it.', question: 'Please confirm: is the string length about a forearm (0.3 m)?' }] })
    render(<NeedsForm needs={n} onSubmit={onSubmit} />)
    expect(screen.getByLabelText(/Please confirm/)).toHaveValue('0.3')
    expect(screen.getByText(/could not re-read/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect(onSubmit).toHaveBeenCalledWith({ string_length: { value: 0.3, unit: 'm' } })
  })
  it('asks for the name of a reference object as text', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    render(<NeedsForm needs={needs({ missing: [{ name: 'reference_object', question: 'Which object has that length?', dimension: null }] })} onSubmit={onSubmit} />)
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    await userEvent.type(screen.getByLabelText(/Which object/), 'metre stick')
    await userEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect(onSubmit).toHaveBeenCalledWith({ reference_object: { text: 'metre stick' } })
  })
  it("shows the server's objection (an implausible value) and stays usable", async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error('5000 m does not look right for the string length.'))
    render(<NeedsForm needs={needs()} onSubmit={onSubmit} />)
    await userEvent.type(screen.getByLabelText(/length of the string/i), '5000')
    await userEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect(await screen.findByText(/does not look right/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Continue' })).toBeEnabled()
  })
})

describe('ProgressPanel', () => {
  it('shows the stage, the frame count and the checks as they arrive', () => {
    let s = emptyStream()
    s = applyEvent(s, { id: 1, ts: 1, kind: 'status', message: 'Finding and following the object in every frame', data: null })
    s = applyEvent(s, { id: 2, ts: 2, kind: 'progress', message: "Tracking 'glass marble': frame 170 of 362", data: { done: 170, total: 362 } })
    s = applyEvent(s, { id: 3, ts: 3, kind: 'log', message: "No match found for 'pendulum bob'.", data: { step: 'track', ok: false } })
    s = applyEvent(s, { id: 4, ts: 4, kind: 'plan', message: 'You want the swing period.', data: null })
    render(<ProgressPanel job={job({ status: 'queued', progress: 0, result: null })} stream={s} />)
    expect(screen.getByText("Tracking 'glass marble': frame 170 of 362")).toBeInTheDocument()
    expect(screen.getByText(/Now:/).parentElement).toHaveTextContent('Track')
    expect(screen.getByText(/Gauge understood/)).toBeInTheDocument()
    expect(screen.getByText("No match found for 'pendulum bob'.")).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'problem found' })).toBeInTheDocument()
    expect(screen.getByText('36%')).toBeInTheDocument()
  })
  it('starts from a clean slate', () => {
    render(<ProgressPanel job={job({ status: 'queued', progress: 0, result: null, message: 'Waiting to start' })} stream={emptyStream()} />)
    expect(screen.getByText('Waiting to start')).toBeInTheDocument()
    expect(screen.queryByText(/How Gauge is checking/)).not.toBeInTheDocument()
  })
})

describe('ResultCard', () => {
  it('shows the answer, its explanation and what to be careful about', () => {
    render(<ResultCard job={job()} result={measurement()} shown={false} onShow={vi.fn()} />)
    expect(screen.getByLabelText('0.8582 ± 0.0016 s')).toBeInTheDocument()
    expect(screen.getByText('The swing period is 0.8582 ± 0.0016 s.')).toBeInTheDocument()
    expect(screen.getByText(/Explained by Muse Spark/)).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent('Film a longer steady swing.')
    expect(screen.getByTestId('chart')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Download the data/ })).toHaveAttribute('href', '/api/jobs/j1/data.csv')
  })
  it('says plainly when the explanation was written by code instead of the model', () => {
    const r = measurement({ explanation: { ...measurement().explanation, source: 'fallback' } })
    render(<ResultCard job={job()} result={r} shown={false} onShow={vi.fn()} />)
    expect(screen.getByText(/written by a template written in code|template written in code/i)).toBeInTheDocument()
    expect(screen.queryByText(/Explained by Muse Spark/)).not.toBeInTheDocument()
  })
  it('opens the checks when something went wrong along the way', () => {
    render(<ResultCard job={job()} result={measurement()} shown={false} onShow={vi.fn()} />)
    const details = screen.getByText(/How Gauge checked its work/).closest('details')!
    expect(details).toHaveAttribute('open')
    expect(within(details).getByText("No match found for 'pendulum bob'.")).toBeInTheDocument()
  })
  it('keeps the checks closed when everything passed', () => {
    const r = measurement({ checks: [{ step: 'check', message: 'Checks passed.', ok: true }] })
    render(<ResultCard job={job()} result={r} shown={false} onShow={vi.fn()} />)
    expect(screen.getByText(/How Gauge checked its work/).closest('details')).not.toHaveAttribute('open')
  })
  it('lets the person put this answer on the video', async () => {
    const onShow = vi.fn()
    const { rerender } = render(<ResultCard job={job()} result={measurement()} shown={false} onShow={onShow} />)
    await userEvent.click(screen.getByRole('button', { name: 'Show on the video' }))
    expect(onShow).toHaveBeenCalled()
    rerender(<ResultCard job={job()} result={measurement()} shown onShow={onShow} />)
    expect(screen.getByRole('button', { name: 'Shown on the video' })).toHaveAttribute('aria-pressed', 'true')
  })
  it('reports what it cost, honestly', () => {
    render(<ResultCard job={job()} result={measurement()} shown={false} onShow={vi.fn()} />)
    expect(screen.getByText(/Tracking reused from before/)).toBeInTheDocument()
    expect(screen.getByText(/\$0\.0007/)).toBeInTheDocument()
  })
  it('lists every number, and what the person said', () => {
    render(<ResultCard job={job()} result={measurement()} shown={false} onShow={vi.fn()} />)
    expect(screen.getByText('1.165 ± 0.002 Hz')).toBeInTheDocument()
    expect(screen.getByText(/string length \(you said\)/)).toBeInTheDocument()
  })
})

describe('NoAnswerCard', () => {
  it('refuses an unsupported question with a reason and what Gauge can do', () => {
    render(<NoAnswerCard result={{ kind: 'unsupported', answerable: false, question: 'q', reason: 'Gauge does not measure sports.', can_measure: ['Period of a pendulum.'], usage }} />)
    expect(screen.getByText('Gauge cannot answer this one')).toBeInTheDocument()
    expect(screen.getByText('Gauge does not measure sports.')).toBeInTheDocument()
    expect(screen.getByText('Period of a pendulum.')).toBeInTheDocument()
  })
  it('explains a measurement that could not be made, with advice', () => {
    render(<NoAnswerCard result={{ kind: 'no_result', answerable: false, question: 'q', reason: 'No clear oscillation was found.', advice: ['Keep the object in view.'], checks: [], usage }} />)
    expect(screen.getByText('Gauge could not measure this')).toBeInTheDocument()
    expect(screen.getByText('Keep the object in view.')).toBeInTheDocument()
  })
})

describe('JobCard', () => {
  const props = { superseded: false, shown: false, onShow: vi.fn(), onFinished: vi.fn(), onAnswered: vi.fn(), onRetry: vi.fn() }
  it('shows an answered question and tells its owner it finished', async () => {
    mocked.getJob.mockResolvedValue(job())
    render(<JobCard jobId="j1" {...props} />)
    expect(await screen.findByLabelText('Answer')).toBeInTheDocument()
    expect(props.onFinished).toHaveBeenCalled()
  })
  it('follows a running question live and reloads it when it ends', async () => {
    mocked.getJob.mockResolvedValueOnce(job({ status: 'planning', progress: 0.03, result: null }))
    let end: (s: string) => void = () => {}
    mocked.followJob.mockImplementation((_id, _onEvent, onEnd) => { end = onEnd; return () => {} })
    render(<JobCard jobId="j1" {...props} />)
    expect(await screen.findByRole('status', { name: 'Progress' })).toBeInTheDocument()
    mocked.getJob.mockResolvedValue(job())
    end('done')
    expect(await screen.findByLabelText('Answer')).toBeInTheDocument()
  })
  it('pauses for input, and the answer starts a new question', async () => {
    const needs: Needs = { missing: [{ name: 'string_length', question: 'What is the length of the string?', dimension: 'length' }], confirm: [], notes: [], interpretation: 'You want g.' }
    mocked.getJob.mockResolvedValue(job({ status: 'needs_input', result: null, needs }))
    mocked.answer.mockResolvedValue({ job_id: 'j2' })
    const onAnswered = vi.fn()
    render(<JobCard jobId="j1" {...props} onAnswered={onAnswered} />)
    await userEvent.type(await screen.findByLabelText(/length of the string/i), '0.5')
    await userEvent.click(screen.getByRole('button', { name: 'Continue' }))
    await waitFor(() => expect(onAnswered).toHaveBeenCalledWith('j2'))
    expect(mocked.answer).toHaveBeenCalledWith('j1', { string_length: { value: 0.5, unit: 'm' } })
  })
  it('does not offer the same paused question again once it has been answered', async () => {
    const needs: Needs = { missing: [{ name: 'string_length', question: 'q', dimension: 'length' }], confirm: [], notes: [], interpretation: '' }
    mocked.getJob.mockResolvedValue(job({ status: 'needs_input', result: null, needs }))
    render(<JobCard jobId="j1" {...props} superseded />)
    expect(await screen.findByText('Answered below.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Continue' })).not.toBeInTheDocument()
  })
  it('shows a failure plainly and offers to ask again', async () => {
    mocked.getJob.mockResolvedValue(job({ status: 'failed', result: null, message: 'This took too long and was stopped.' }))
    const onRetry = vi.fn()
    render(<JobCard jobId="j1" {...props} onRetry={onRetry} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('took too long')
    await userEvent.click(screen.getByRole('button', { name: 'Ask again' }))
    expect(onRetry).toHaveBeenCalledWith('What is the swing period?')
  })
  it('shows a load error instead of spinning forever', async () => {
    mocked.getJob.mockRejectedValue(new Error('No such question.'))
    render(<JobCard jobId="j1" {...props} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('No such question.')
  })
})
