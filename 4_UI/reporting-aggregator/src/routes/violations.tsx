import { createFileRoute } from '@tanstack/react-router'
import ViolationsTable from '../components/ViolationsTable'

export const Route = createFileRoute('/violations')({
  component: ViolationsComponent,
})

function ViolationsComponent() {
  return (
    <main className="page-wrap px-4 pb-8 pt-8">
      <section className="island-shell mb-6 rounded-2xl p-6 sm:p-8">
        <p className="island-kicker mb-2">All violations</p>
        <h1 className="display-title mb-2 text-2xl font-bold text-[var(--sea-ink)] sm:text-3xl">
          Violations Across All Sites
        </h1>
        <p className="m-0 max-w-3xl text-base text-[var(--sea-ink-soft)]">
          One row per (site × rule) aggregation. Filter by impact level or WCAG
          tag, sort any column, and click a row to inspect a sample node with
          HTML, target selector, and axe failure summary.
        </p>
      </section>

      <section className="island-shell rounded-2xl p-4 sm:p-6">
        <ViolationsTable />
      </section>
    </main>
  )
}