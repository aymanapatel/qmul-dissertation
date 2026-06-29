import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { getSiteDetail } from '../server/queries'
import ImpactBadge from '../components/ImpactBadge'
import TagChips from '../components/TagChips'
import type { SiteDetail, ImpactKey } from '../lib/types'

export const Route = createFileRoute('/sites/$domain')({
  component: SiteDetailComponent,
})

function SiteDetailComponent() {
  const { domain } = Route.useParams()
  const [detail, setDetail] = useState<SiteDetail | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    getSiteDetail({ data: { domain: decodeURIComponent(domain) } })
      .then((d) => setDetail(d))
      .finally(() => setLoading(false))
  }, [domain])

  if (loading) {
    return (
      <main className="page-wrap px-4 py-12">
        <p className="text-[var(--sea-ink-soft)]">Loading site detail…</p>
      </main>
    )
  }

  if (!detail) {
    return (
      <main className="page-wrap px-4 py-12">
        <p className="text-[var(--sea-ink-soft)]">
          Site “{domain}” not found.{' '}
          <Link to="/" className="text-[var(--lagoon-deep)] hover:underline">
            Back to sites
          </Link>
        </p>
      </main>
    )
  }

  const { site, violations } = detail

  return (
    <main className="page-wrap px-4 pb-8 pt-8">
      <div className="mb-4">
        <Link to="/" className="text-sm text-[var(--lagoon-deep)] hover:underline">
          ← Back to sites
        </Link>
      </div>

      <section className="island-shell mb-6 rounded-2xl p-6 sm:p-8">
        <p className="island-kicker mb-2">Site report</p>
        <h1 className="display-title mb-2 break-all text-2xl font-bold text-[var(--sea-ink)] sm:text-3xl">
          {site.domain}
        </h1>
        <a
          href={site.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-[var(--lagoon-deep)] hover:underline"
        >
          {site.url} ↗
        </a>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div>
            <p className="m-0 text-xs font-semibold text-[var(--sea-ink-soft)]">Total violations</p>
            <p className="m-0 text-2xl font-bold text-[var(--sea-ink)]">
              {site.total_violations.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="m-0 text-xs font-semibold text-[var(--sea-ink-soft)]">Rules violated</p>
            <p className="m-0 text-2xl font-bold text-[var(--sea-ink)]">{site.rules_violated}</p>
          </div>
          <div>
            <p className="m-0 text-xs font-semibold text-[var(--sea-ink-soft)]">Pages scanned</p>
            <p className="m-0 text-2xl font-bold text-[var(--sea-ink)]">{site.total_pages}</p>
          </div>
          <div>
            <p className="m-0 text-xs font-semibold text-[var(--sea-ink-soft)]">Scan date</p>
            <p className="m-0 text-sm font-semibold text-[var(--sea-ink)]">
              {site.timestamp ? new Date(site.timestamp).toLocaleString() : '—'}
            </p>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {(['critical', 'serious', 'moderate', 'minor'] as ImpactKey[]).map((k) =>
            site.impact_counts[k] > 0 ? (
              <ImpactBadge key={k} impact={k} count={site.impact_counts[k]} />
            ) : null,
          )}
          {site.total_violations === 0 && (
            <span className="rounded-full border border-green-300 bg-green-100 px-2.5 py-0.5 text-xs font-semibold text-green-800">
              No violations detected
            </span>
          )}
        </div>
      </section>

      <section className="island-shell rounded-2xl p-4 sm:p-6">
        <h2 className="mb-4 text-lg font-bold text-[var(--sea-ink)]">
          Violations ({violations.length})
        </h2>
        {violations.length === 0 ? (
          <p className="text-sm text-[var(--sea-ink-soft)]">
            No violations recorded for this site.
          </p>
        ) : (
          <div className="space-y-3">
            {violations.map((v) => (
              <article
                key={`${v.rule_id}`}
                className="rounded-xl border border-[var(--line)] bg-[var(--chip-bg)] p-4"
              >
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <code className="text-sm font-bold text-[var(--sea-ink)]">{v.rule_id}</code>
                    <p className="m-0 mt-1 text-sm text-[var(--sea-ink-soft)]">{v.help}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <ImpactBadge impact={v.impact} />
                    <span className="text-sm font-semibold text-[var(--sea-ink)]">
                      {v.node_count.toLocaleString()} node{v.node_count === 1 ? '' : 's'}
                    </span>
                  </div>
                </div>
                <div className="mt-2">
                  <TagChips tags={v.tags} max={10} />
                </div>
                {v.sample_html && (
                  <details className="mt-3">
                    <summary className="cursor-pointer text-xs font-semibold text-[var(--lagoon-deep)]">
                      Sample HTML / target
                    </summary>
                    <pre className="mt-2 overflow-x-auto rounded bg-[var(--surface)] p-3 text-xs">
                      <code>{v.sample_html}</code>
                    </pre>
                    <code className="mt-2 block whitespace-pre-wrap break-all text-xs text-[var(--sea-ink-soft)]">
                      target: {v.sample_target}
                    </code>
                  </details>
                )}
                <a
                  href={v.helpUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-2 inline-block text-xs text-[var(--lagoon-deep)] hover:underline"
                >
                  Rule documentation ↗
                </a>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  )
}