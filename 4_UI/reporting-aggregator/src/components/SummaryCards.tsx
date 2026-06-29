import type { SummaryStats, ImpactKey } from '../lib/types'
import { IMPACT_ORDER } from '../lib/types'
import ImpactBadge from './ImpactBadge'

export default function SummaryCards({ stats }: { stats: SummaryStats }) {
  const impactBars: Array<{ key: ImpactKey; value: number }> = IMPACT_ORDER.filter(
    (k) => k !== 'none' && stats.impactTotals[k] > 0,
  ).map((k) => ({ key: k, value: stats.impactTotals[k] }))
  const maxImpact = Math.max(...impactBars.map((b) => b.value), 1)

  const cards = [
    {
      label: 'Sites Scanned',
      value: stats.totalSites.toLocaleString(),
      sub: `${stats.cleanSites.toLocaleString()} clean (${Math.round((stats.cleanSites / stats.totalSites) * 100)}%)`,
    },
    {
      label: 'Total Violations',
      value: stats.totalViolations.toLocaleString(),
      sub: `${stats.avgViolations.toLocaleString()} avg / site`,
    },
    {
      label: 'Sites w/ Critical Issues',
      value: stats.criticalSites.toLocaleString(),
      sub: `${Math.round((stats.criticalSites / stats.totalSites) * 100)}% of scanned`,
    },
    {
      label: 'Distinct Rules Violated',
      value: stats.topRules.length > 0 ? new Set(stats.topRules.map((r) => r.rule)).size.toLocaleString() : '0',
      sub: `across ${stats.totalRuleViolations.toLocaleString()} (site × rule) rows`,
    },
  ]

  return (
    <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((c) => (
        <article
          key={c.label}
          className="island-shell rounded-2xl p-5"
        >
          <p className="island-kicker mb-2">{c.label}</p>
          <p className="m-0 text-3xl font-bold tracking-tight text-[var(--sea-ink)]">
            {c.value}
          </p>
          <p className="m-0 mt-1 text-xs text-[var(--sea-ink-soft)]">{c.sub}</p>
        </article>
      ))}

      <article className="island-shell col-span-full rounded-2xl p-5 lg:col-span-2">
        <p className="island-kicker mb-3">Violations by Impact</p>
        <div className="space-y-2">
          {impactBars.length === 0 && (
            <p className="text-sm text-[var(--sea-ink-soft)]">No violations detected.</p>
          )}
          {impactBars.map((b) => (
            <div key={b.key} className="flex items-center gap-3">
              <div className="w-20 shrink-0">
                <ImpactBadge impact={b.key} />
              </div>
              <div className="h-3 flex-1 overflow-hidden rounded-full bg-[var(--line)]">
                <div
                  className="h-full rounded-full bg-[var(--lagoon-deep)]"
                  style={{ width: `${(b.value / maxImpact) * 100}%` }}
                />
              </div>
              <span className="w-16 shrink-0 text-right text-sm font-semibold text-[var(--sea-ink)]">
                {b.value.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </article>

      <article className="island-shell col-span-full rounded-2xl p-5 lg:col-span-2">
        <p className="island-kicker mb-3">Top 10 Violated Rules</p>
        <ol className="m-0 space-y-1.5 p-0">
          {stats.topRules.map((r, i) => (
            <li key={r.rule} className="flex items-start gap-2 text-sm">
              <span className="w-6 shrink-0 text-right font-bold text-[var(--sea-ink-soft)]">
                {i + 1}.
              </span>
              <code className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
                {r.rule}
              </code>
              <span className="shrink-0 font-semibold text-[var(--sea-ink)]">
                {r.count.toLocaleString()}
              </span>
            </li>
          ))}
        </ol>
      </article>
    </section>
  )
}