import { createFileRoute } from '@tanstack/react-router'
import { useState, useEffect } from 'react'
import SummaryCards from '../components/SummaryCards'
import SitesTable from '../components/SitesTable'
import { getSummaryStats } from '../server/queries'
import type { SummaryStats } from '../lib/types'

export const Route = createFileRoute('/')({
  component: HomeComponent,
})

function HomeComponent() {
  const [stats, setStats] = useState<SummaryStats | null>(null)

  useEffect(() => {
    getSummaryStats().then(setStats).catch(() => setStats(null))
  }, [])

  return (
    <main className="page-wrap px-4 pb-8 pt-8">
      <section className="island-shell mb-6 rounded-2xl p-6 sm:p-8">
        <p className="island-kicker mb-2">axe-core analysis</p>
        <h1 className="display-title mb-3 text-3xl font-bold text-[var(--sea-ink)] sm:text-4xl">
          Accessibility Reports
        </h1>
        <p className="m-0 max-w-3xl text-base text-[var(--sea-ink-soft)]">
          Aggregated WCAG 2.0/2.1 conformance results from axe-core scans of
          3,000+ websites. Use the table below to sort, filter, and drill into
          violations by site, or browse all rule-level violations across every
          scanned site.
        </p>
      </section>

      {stats ? (
        <SummaryCards stats={stats} />
      ) : (
        <p className="text-sm text-[var(--sea-ink-soft)]">Loading summary…</p>
      )}

      <section className="island-shell mt-6 rounded-2xl p-4 sm:p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="m-0 text-lg font-bold text-[var(--sea-ink)]">Sites overview</h2>
          <a
            href="/violations"
            className="text-sm font-semibold text-[var(--lagoon-deep)] hover:underline"
          >
            View all violations →
          </a>
        </div>
        <SitesTable />
      </section>
    </main>
  )
}