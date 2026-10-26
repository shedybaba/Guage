import { csvUrl } from '../api'
import { chartSpecs } from '../lib/chart'
import { formatUsd, TOOL_NAMES } from '../lib/format'
import type { Job, MeasurementResult, NoResult, UnsupportedResult } from '../types'
import { ChecksList } from './ChecksList'
import { ResultChart } from './ResultChart'

function Section({ title, children, open = false }: { title: string; children: React.ReactNode; open?: boolean }) {
  return (
    <details open={open} className="group rounded-lg border border-line bg-paper">
      <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-sm font-semibold">
        {title}
        <span aria-hidden="true" className="text-ink-soft transition-transform group-open:rotate-90">›</span>
      </summary>
      <div className="border-t border-line px-3 py-3">{children}</div>
    </details>
  )
}

/** Split "1.419 ± 0.002 s" so the number is big and the uncertainty and unit are quieter. */
function Headline({ display }: { display: string }) {
  const [main, rest] = display.split(' ± ')
  const m = rest?.match(/^(\S+)\s*(.*)$/)
  return (
    <p className="num flex flex-wrap items-baseline gap-x-2 text-[2.1rem] font-semibold leading-tight sm:text-4xl" aria-label={display}>
      <span>{rest ? main : display}</span>
      {m && <span className="text-xl font-medium text-ink-soft">± {m[1]}</span>}
      {m && m[2] && <span className="text-xl font-medium text-ink-soft">{m[2]}</span>}
    </p>
  )
}

export function ResultCard({ job, result, shown, onShow }: { job: Job; result: MeasurementResult; shown: boolean; onShow: () => void }) {
  const ex = result.explanation
  const charts = chartSpecs(result.tool, result.chart, result.numbers)
  const others = result.numbers.filter((n) => n.id !== result.headline.id)
  return (
    <article className="rise space-y-4" aria-label="Answer">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-accent">{TOOL_NAMES[result.tool] ?? result.tool}</p>
        <p className="mt-1 text-sm text-ink-soft">{result.headline.label}</p>
        <Headline display={result.headline.display} />
      </div>

      <p className="text-base leading-relaxed">{ex.summary}</p>
      <p className="text-xs text-ink-soft">
        {ex.source === 'model'
          ? 'Explained by Muse Spark. Every number in the text was checked against the measurement.'
          : 'Explained by a template written in code: the language model’s text did not pass the number check, so it was not used.'}
      </p>

      {ex.caveats.length > 0 && (
        <div className="rounded-lg border border-copper/30 bg-copper-soft px-3 py-2.5" role="note">
          <p className="text-xs font-semibold uppercase tracking-wide text-copper">Worth knowing</p>
          <ul className="mt-1 space-y-1 text-sm leading-relaxed">{ex.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={onShow} aria-pressed={shown}
          className={`rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${shown ? 'bg-accent-soft text-accent' : 'bg-ink text-paper hover:bg-ink/85'}`}>
          {shown ? 'Shown on the video' : 'Show on the video'}
        </button>
        <a href={csvUrl(job.job_id)} download className="rounded-lg border border-line bg-card px-4 py-2 text-sm font-semibold hover:border-ink/40">Download the data (CSV)</a>
      </div>

      {charts.map((c) => <ResultChart key={c.title} spec={c} />)}

      <div className="space-y-2">
        <Section title="How it was measured">
          <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed">{ex.how_measured.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </Section>
        <Section title={`How Gauge checked its work (${result.checks.length})`} open={result.checks.some((c) => c.ok === false)}>
          <ChecksList checks={result.checks} />
        </Section>
        {others.length > 0 && (
          <Section title="All the numbers">
            <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 text-sm">
              {others.map((n) => (
                <div key={n.id} className="col-span-2 grid grid-cols-subgrid items-baseline border-b border-line/60 pb-1.5 last:border-0">
                  <dt className="text-ink-soft">{n.label}</dt>
                  <dd className="num text-right font-medium">{n.display}</dd>
                </div>
              ))}
            </dl>
            {result.inputs.length > 0 && (
              <>
                <p className="mb-1 mt-3 text-xs font-semibold uppercase tracking-wide text-ink-soft">What you told Gauge</p>
                <ul className="text-sm">{result.inputs.map((i) => <li key={i.label}>{i.label}: <span className="num font-medium">{i.display}</span></li>)}</ul>
              </>
            )}
          </Section>
        )}
        {result.warnings.length > 0 && (
          <Section title={`All warnings (${result.warnings.length})`}>
            <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed">{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </Section>
        )}
      </div>

      <p className="num text-xs text-ink-soft">
        {result.usage.sam_frames_billed > 0 ? `${result.usage.sam_frames_billed} frames tracked (new). ` : 'Tracking reused from before. '}
        Language model: {formatUsd(result.usage.llm_cost_usd)}.
      </p>
    </article>
  )
}

/** Gauge's honest "I could not": an unsupported question, or a measurement that could not be made. */
export function NoAnswerCard({ result }: { result: UnsupportedResult | NoResult }) {
  const unsupported = result.kind === 'unsupported'
  return (
    <article className="rise space-y-3" aria-label="No answer">
      <p className="text-sm font-semibold text-copper">{unsupported ? 'Gauge cannot answer this one' : 'Gauge could not measure this'}</p>
      <p className="text-base leading-relaxed">{result.reason}</p>
      {unsupported ? (
        <div>
          <p className="mb-1.5 text-sm font-semibold">What Gauge can measure</p>
          <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-ink-soft">{result.can_measure.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </div>
      ) : (
        <>
          {result.advice.length > 0 && (
            <div className="rounded-lg border border-copper/30 bg-copper-soft px-3 py-2.5">
              <p className="text-xs font-semibold uppercase tracking-wide text-copper">What to try</p>
              <ul className="mt-1 space-y-1 text-sm leading-relaxed">{result.advice.map((a, i) => <li key={i}>{a}</li>)}</ul>
            </div>
          )}
          {result.checks.length > 0 && (
            <details className="rounded-lg border border-line bg-paper">
              <summary className="cursor-pointer px-3 py-2.5 text-sm font-semibold">How Gauge checked its work ({result.checks.length})</summary>
              <div className="border-t border-line px-3 py-3"><ChecksList checks={result.checks} /></div>
            </details>
          )}
        </>
      )}
    </article>
  )
}
