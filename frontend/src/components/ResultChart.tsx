import { CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ChartSpec } from '../lib/chart'

const fmt = (v: unknown) => (typeof v === 'number' ? Number(v.toPrecision(4)).toString() : String(v ?? ''))

/** One chart from a spec (see lib/chart.ts). Axis titles are always shown: a chart without units is not a measurement. */
export function ResultChart({ spec }: { spec: ChartSpec }) {
  return (
    <figure className="rounded-xl border border-line bg-card p-3" aria-label={spec.title}>
      <figcaption className="mb-2 px-1 text-sm font-semibold">{spec.title}</figcaption>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={160}>
          <ComposedChart data={spec.data} margin={{ top: 6, right: 12, bottom: 22, left: 4 }}>
            <CartesianGrid stroke="#d8d1bf" strokeOpacity={0.6} />
            <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tickFormatter={fmt} tick={{ fontSize: 11 }} stroke="#52605b"
              label={{ value: spec.xLabel, position: 'insideBottom', offset: -14, fontSize: 11, fill: '#52605b' }} />
            <YAxis tickFormatter={fmt} tick={{ fontSize: 11 }} stroke="#52605b" width={56}
              label={{ value: spec.yLabel, angle: -90, position: 'insideLeft', offset: 8, fontSize: 11, fill: '#52605b', style: { textAnchor: 'middle' } }} />
            <Tooltip formatter={(v) => fmt(v)} labelFormatter={(l) => `t = ${fmt(l)} s`} contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #d8d1bf', background: '#fbf9f4' }} />
            {spec.series.some((s) => s.label) && <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 11 }} formatter={(v) => v} />}
            {spec.areas.map((a) => (
              <ReferenceArea key={a.label + a.x1} x1={a.x1} x2={a.x2} fill="#0e6a60" fillOpacity={0.07} stroke="#0e6a60" strokeOpacity={0.25}
                label={{ value: a.label, position: 'insideTopLeft', fontSize: 10, fill: '#0e6a60' }} />
            ))}
            {spec.markers.map((m, i) => <ReferenceLine key={i} x={m.x} stroke={m.color} strokeOpacity={0.55} strokeDasharray="2 3" />)}
            {spec.series.map((s) => (
              <Line key={s.key} dataKey={s.key} name={s.label} stroke={s.kind === 'dots' ? 'none' : s.color} strokeWidth={s.kind === 'line' || s.kind === 'step' ? 2.2 : 1.5}
                strokeDasharray={s.kind === 'dashed' ? '5 4' : undefined} type={s.kind === 'step' ? 'stepAfter' : 'linear'}
                dot={s.kind === 'dots' ? { r: 1.3, fill: s.color, fillOpacity: 0.55, stroke: 'none' } : false} activeDot={{ r: 3 }} isAnimationActive={false} connectNulls={false}
                legendType={s.label ? (s.kind === 'dots' ? 'circle' : 'plainline') : 'none'} />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </figure>
  )
}
