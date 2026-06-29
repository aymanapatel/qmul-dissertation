import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/about')({
  component: About,
})

function About() {
  return (
    <main className="page-wrap px-4 py-12">
      <section className="island-shell rounded-2xl p-6 sm:p-8">
        <p className="island-kicker mb-2">About this tool</p>
        <h1 className="display-title mb-3 text-3xl font-bold text-[var(--sea-ink)] sm:text-4xl">
          axe-core Accessibility Reporter
        </h1>
        <p className="m-0 max-w-3xl text-base leading-8 text-[var(--sea-ink-soft)]">
          This dashboard aggregates axe-core scans for 3,000+ websites, performed
          against WCAG 2.0 and 2.1 A/AA conformance levels. Data is pre-aggregated
          at build time into static JSON datasets and served via TanStack Start
          server functions; the tables use TanStack Table v8 with server-side sort,
          filter, and pagination for performance on tens of thousands of rows.
        </p>
        <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-[var(--sea-ink-soft)]">
          <li>
            <strong>Sites overview</strong> — one row per scanned domain with
            violation totals broken down by impact.
          </li>
          <li>
            <strong>Violations</strong> — one row per (site × rule) with node
            counts, rule metadata, and a drill-down drawer for sample HTML.
          </li>
          <li>
            <strong>Site detail</strong> — click any domain to view its full
            rule-level breakdown.
          </li>
        </ul>
        <p className="mt-4 text-sm text-[var(--sea-ink-soft)]">
          Built with TanStack Start, TanStack Table v8, and Tailwind CSS.
        </p>
      </section>
    </main>
  )
}