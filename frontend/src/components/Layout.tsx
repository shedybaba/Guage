import { Link, NavLink } from 'react-router-dom'
import type { ReactNode } from 'react'
import { LogoMark } from './Logo'

const link = ({ isActive }: { isActive: boolean }) =>
  `rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${isActive ? 'bg-ink text-paper' : 'text-ink-soft hover:text-ink hover:bg-paper-2'}`

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-30 border-b border-line bg-paper/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
          <Link to="/" className="flex items-center gap-2.5" aria-label="Gauge home">
            <LogoMark />
            <span className="text-lg font-semibold tracking-tight">Gauge</span>
          </Link>
          <nav className="flex items-center gap-1" aria-label="Main">
            <NavLink to="/analyze" className={link}>Analyze</NavLink>
            <NavLink to="/developers" className={link}>API</NavLink>
          </nav>
        </div>
        <div className="ruler" aria-hidden="true" />
      </header>
      <main className="flex-1">{children}</main>
      <footer className="border-t border-line bg-paper-2/60">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-sm text-ink-soft sm:flex-row sm:items-center sm:justify-between">
          <p>Built for the Meta Global AI Developer Hackathon. All of it was written during the event.</p>
          <p>
            Perception: <span className="font-medium text-ink">SAM 3.1</span> · Language:{' '}
            <span className="font-medium text-ink">Muse Spark</span> · Numbers: <span className="font-medium text-ink">plain Python</span>
          </p>
        </div>
      </footer>
    </div>
  )
}
