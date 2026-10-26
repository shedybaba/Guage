import { Link } from 'react-router-dom'
import { HeroDemo } from '../components/HeroDemo'

const STEPS = [
  { n: '1', title: 'Upload a video', text: 'Any phone clip of an experiment or a machine. Keep the phone still and the object in view.' },
  { n: '2', title: 'Ask in plain English', text: '"What is the swing period?" "How many parts crossed the line?" Type it, and Gauge answers only what you ask.' },
  { n: '3', title: 'Get a measured answer', text: 'A number with its uncertainty, the video with the tracked objects drawn on it, a chart, and how it was checked.' },
]

const PILLARS = [
  { title: 'AI sees', text: 'SAM 3.1 finds the object by name and follows it through every frame. Muse Spark reads your question and writes the explanation.' },
  { title: 'Code measures', text: 'Every number comes from plain Python physics on the tracked positions, with an uncertainty. A language model never calculates.' },
  { title: 'You can check', text: 'Each number in the explanation is verified against the measured results. Gauge shows how it checked its own work, and warns when a result is shaky.' },
]

const TOOLS = ['Pendulum period and g', 'Spring frequency, damping and k', 'Free fall', 'Speed and acceleration on a ramp', 'Momentum in a collision', 'Counting parts', 'Line crossings per minute']

export default function Home() {
  return (
    <>
      <section className="graph-paper border-b border-line">
        <div className="mx-auto grid max-w-6xl items-center gap-10 px-4 py-14 md:grid-cols-[1.05fr_1fr] md:py-20">
          <div className="rise">
            <p className="mb-3 inline-block rounded-full border border-line bg-card px-3 py-1 text-xs font-medium text-ink-soft">
              Built for the Meta Global AI Developer Hackathon
            </p>
            <h1 className="text-4xl font-semibold leading-[1.08] tracking-tight sm:text-5xl">
              Turn any phone camera into a <span className="text-accent">measuring instrument</span>.
            </h1>
            <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-soft">
              Chatbots guess. Manual tools take hours. Gauge measures, from one question, and shows its proof.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/analyze" className="rounded-lg bg-accent px-5 py-3 text-base font-semibold text-white shadow-sm transition-colors hover:bg-accent-strong">
                Try it on a video
              </Link>
              <Link to="/developers" className="rounded-lg border border-ink/20 bg-card px-5 py-3 text-base font-semibold text-ink transition-colors hover:border-ink/40">
                Use the API
              </Link>
            </div>
            <p className="mt-6 max-w-xl text-sm text-ink-soft">
              For students and teachers without lab sensors, researchers who track motion by hand, and engineers checking a machine.
            </p>
          </div>
          <HeroDemo />
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-14">
        <h2 className="text-2xl font-semibold tracking-tight">How it works</h2>
        <ol className="mt-6 grid gap-4 md:grid-cols-3">
          {STEPS.map((s) => (
            <li key={s.n} className="rounded-xl border border-line bg-card p-5">
              <span className="num inline-flex h-8 w-8 items-center justify-center rounded-md bg-ink text-sm font-semibold text-paper">{s.n}</span>
              <h3 className="mt-3 text-lg font-semibold">{s.title}</h3>
              <p className="mt-1.5 leading-relaxed text-ink-soft">{s.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="border-y border-line bg-paper-2/60">
        <div className="mx-auto max-w-6xl px-4 py-14">
          <h2 className="text-2xl font-semibold tracking-tight">Measured, not guessed</h2>
          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {PILLARS.map((p) => (
              <div key={p.title} className="rounded-xl border border-line bg-card p-5">
                <h3 className="text-lg font-semibold text-accent">{p.title}</h3>
                <p className="mt-1.5 leading-relaxed text-ink-soft">{p.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-14">
        <h2 className="text-2xl font-semibold tracking-tight">What it can measure today</h2>
        <ul className="mt-5 flex flex-wrap gap-2">
          {TOOLS.map((t) => (
            <li key={t} className="rounded-full border border-line bg-card px-4 py-2 text-sm font-medium">{t}</li>
          ))}
        </ul>
        <div className="mt-8">
          <Link to="/analyze" className="font-semibold text-accent underline underline-offset-4 hover:text-accent-strong">Start with a video of your own →</Link>
        </div>
      </section>
    </>
  )
}
