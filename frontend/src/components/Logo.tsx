/** The Gauge mark: a dial with a needle. */
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="#17211f" />
      <path d="M6 22a10 10 0 0 1 20 0" fill="none" stroke="#f5f2ea" strokeWidth="2.2" strokeLinecap="round" />
      <path d="M16 22l5-8" stroke="#3fbfae" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="16" cy="22" r="2" fill="#f5f2ea" />
    </svg>
  )
}
