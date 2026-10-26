import { useEffect, useRef, useState } from 'react'

// An illustration, not a measurement: a simulated pendulum shows the idea (video -> tracked object -> fitted wave -> period).
const T = 2.4 // seconds
const AMP = 0.62 // radians
const LEN = 150
const PIVOT = { x: 260, y: 26 }
const LOOP = 9 // seconds before the demo starts over
const CH = { x0: 44, x1: 496, yc: 338, h: 56 }

const theta = (t: number) => AMP * Math.cos((2 * Math.PI * t) / T)
const bobX = (t: number) => PIVOT.x + LEN * Math.sin(theta(t))
const bobY = (t: number) => PIVOT.y + LEN * Math.cos(theta(t))
const chartX = (u: number) => CH.x0 + (u / LOOP) * (CH.x1 - CH.x0)
const chartY = (u: number) => CH.yc - ((LEN * Math.sin(theta(u))) / (LEN * Math.sin(AMP))) * CH.h

export function HeroDemo() {
  const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  const [t, setT] = useState(reduce ? 7.2 : 0)
  const raf = useRef(0)

  useEffect(() => {
    if (reduce) return
    const start = performance.now()
    const tick = (now: number) => {
      setT(((now - start) / 1000) % LOOP)
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [reduce])

  const trail = []
  for (let u = Math.max(0, t - 0.9); u <= t; u += 0.05) trail.push([bobX(u), bobY(u)])
  const points = []
  for (let u = 0; u <= t; u += 0.06) points.push([chartX(u), chartY(u)])
  const fitted = []
  for (let u = 0; u <= LOOP; u += 0.1) fitted.push(`${chartX(u).toFixed(1)},${chartY(u).toFixed(1)}`)
  const solved = t > 2 * T + 0.4
  const bx = bobX(t)
  const by = bobY(t)

  return (
    <figure className="rise rounded-2xl border border-line bg-card p-3 shadow-sm" aria-label="An illustration of Gauge tracking a pendulum">
      <svg viewBox="0 0 520 430" className="w-full" role="img" aria-label="A simulated pendulum being tracked, with its position plotted over time">
        <defs>
          <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
            <path d="M26 0H0V26" fill="none" stroke="#17211f" strokeOpacity="0.06" />
          </pattern>
        </defs>
        <rect x="0" y="0" width="520" height="230" rx="10" fill="#efe9db" />
        <rect x="0" y="0" width="520" height="230" rx="10" fill="url(#grid)" />
        {/* the stand and string */}
        <rect x="170" y="14" width="180" height="8" rx="3" fill="#8a7b5c" />
        <line x1={PIVOT.x} y1={PIVOT.y} x2={bx} y2={by} stroke="#6f6a5a" strokeWidth="1.5" />
        <circle cx={PIVOT.x} cy={PIVOT.y} r="3.5" fill="#4a463c" />
        {/* the path the tracker has seen */}
        <polyline points={trail.map((p) => p.join(',')).join(' ')} fill="none" stroke="#0e6a60" strokeWidth="2.5" strokeLinecap="round" strokeOpacity="0.55" />
        {/* the bob and Gauge's tracking box */}
        <circle cx={bx} cy={by} r="13" fill="#9fb8b3" stroke="#5d7a75" strokeWidth="1.5" />
        <circle cx={bx - 4} cy={by - 4} r="3.5" fill="#fff" fillOpacity="0.7" />
        <rect x={bx - 19} y={by - 19} width="38" height="38" fill="none" stroke="#0e6a60" strokeWidth="2" />
        <rect x={bx - 19} y={by - 35} width="20" height="16" fill="#0e6a60" />
        <text x={bx - 9} y={by - 23.5} textAnchor="middle" fontSize="11" fontWeight="600" fill="#fff" fontFamily="JetBrains Mono Variable, monospace">0</text>
        <text x="14" y="216" fontSize="11" fill="#52605b" fontFamily="Inter Variable, sans-serif">frame {Math.round(t * 30)}</text>
        <text x="506" y="216" fontSize="11" fill="#52605b" textAnchor="end" fontFamily="Inter Variable, sans-serif">SAM 3.1 finds and follows the bob</text>

        {/* the measurement */}
        <line x1={CH.x0} y1={CH.yc} x2={CH.x1} y2={CH.yc} stroke="#d8d1bf" />
        <line x1={CH.x0} y1={CH.yc - CH.h - 10} x2={CH.x0} y2={CH.yc + CH.h + 10} stroke="#d8d1bf" />
        <text x={CH.x1} y={CH.yc + CH.h + 30} textAnchor="end" fontSize="11" fill="#52605b" fontFamily="Inter Variable, sans-serif">time</text>
        <text x={CH.x0 + 6} y={CH.yc - CH.h - 14} fontSize="11" fill="#52605b" fontFamily="Inter Variable, sans-serif">position</text>
        {solved && <polyline points={fitted.join(' ')} fill="none" stroke="#0e6a60" strokeWidth="2" strokeDasharray="5 4" />}
        {points.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="2.1" fill="#17211f" fillOpacity="0.72" />)}
        {solved && (
          <g className="rise">
            <rect x="262" y="246" width="238" height="30" rx="6" fill="#17211f" />
            <text x="276" y="266" fontSize="14" fill="#f5f2ea" fontFamily="JetBrains Mono Variable, monospace" fontWeight="600">T = 2.400 ± 0.004 s</text>
          </g>
        )}
      </svg>
      <figcaption className="px-1 pb-1 pt-2 text-xs text-ink-soft">
        Illustration: a simulated pendulum. Gauge does this on your video: find the object, follow it, fit the physics, report the number with its uncertainty.
      </figcaption>
    </figure>
  )
}
