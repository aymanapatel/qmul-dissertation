import { useEffect, useState, useRef } from 'react'
import type { ViolationRow, ImpactKey } from '../lib/types'
import ImpactBadge from './ImpactBadge'
import TagChips from './TagChips'

const IFRAME_TIMEOUT_MS = 6000

export default function NodeSamplesDrawer({
  row,
  onClose,
}: {
  row: ViolationRow
  onClose: () => void
}) {
  const [iframeLoaded, setIframeLoaded] = useState(false)
  const [iframeBlocked, setIframeBlocked] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // Reset iframe state whenever the row (and therefore URL) changes
  useEffect(() => {
    setIframeLoaded(false)
    setIframeBlocked(false)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      // If onLoad hasn't fired by now, assume the page blocked embedding
      setIframeBlocked((prev) => prev || true)
    }, IFRAME_TIMEOUT_MS)
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [row.url])

  return (
    <div
      className="fixed inset-0 z-50 flex items-stretch justify-end bg-black/40"
      onClick={onClose}
    >
      <div
        className="island-shell h-full w-full max-w-5xl overflow-y-auto p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="island-kicker mb-1">Violation detail</p>
            <h2 className="m-0 text-xl font-bold text-[var(--sea-ink)]">
              <code>{row.rule_id}</code>
            </h2>
            <p className="m-0 mt-1 text-sm text-[var(--sea-ink-soft)]">{row.help}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-full border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-sm font-semibold text-[var(--sea-ink-soft)] hover:bg-[var(--link-bg-hover)]"
          >
            ✕
          </button>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <ImpactBadge impact={row.impact as ImpactKey} />
          <span className="text-sm text-[var(--sea-ink-soft)]">
            {row.node_count.toLocaleString()} node{row.node_count === 1 ? '' : 's'} affected
          </span>
          <a
            href={row.helpUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-[var(--lagoon-deep)] hover:underline"
          >
            View rule docs ↗
          </a>
        </div>

        <div className="mt-3">
          <TagChips tags={row.tags} max={20} />
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          {/* Left column: violation metadata + sample node */}
          <div className="flex flex-col gap-5">
            <div>
              <p className="island-kicker mb-2">Description</p>
              <p className="m-0 text-sm text-[var(--sea-ink)]">{row.description}</p>
            </div>

            <div>
              <p className="island-kicker mb-2">Sample node</p>
              <div className="space-y-3 rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] p-4">
                <div>
                  <p className="m-0 mb-1 text-xs font-semibold text-[var(--sea-ink-soft)]">
                    HTML
                  </p>
                  <pre className="m-0 overflow-x-auto rounded bg-[var(--surface)] p-3 text-xs text-[var(--sea-ink)]">
                    <code>{row.sample_html}</code>
                  </pre>
                </div>
                <div>
                  <p className="m-0 mb-1 text-xs font-semibold text-[var(--sea-ink-soft)]">
                    Target selector
                  </p>
                  <code className="block whitespace-pre-wrap break-all text-xs text-[var(--sea-ink)]">
                    {row.sample_target}
                  </code>
                </div>
                {row.sample_failure_summary && (
                  <div>
                    <p className="m-0 mb-1 text-xs font-semibold text-[var(--sea-ink-soft)]">
                      Failure summary
                    </p>
                    <p className="m-0 whitespace-pre-wrap text-xs text-[var(--sea-ink-soft)]">
                      {row.sample_failure_summary}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right column: live site iframe preview */}
          <div className="flex flex-col">
            <p className="island-kicker mb-2">Live site preview</p>
            <div className="relative flex-1 overflow-hidden rounded-lg border border-[var(--line)] bg-[var(--chip-bg)]">
              {!iframeLoaded && !iframeBlocked && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-[var(--sea-ink-soft)]">
                  Loading iframe…
                </div>
              )}

              {iframeBlocked && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center">
                  <p className="m-0 text-sm text-[var(--sea-ink-soft)]">
                    This site blocks embedding via <code>X-Frame-Options</code> or{' '}
                    <code>CSP: frame-ancestors</code>.
                  </p>
                  <a
                    href={row.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="demo-button text-sm"
                  >
                    Open {row.domain} in new tab ↗
                  </a>
                </div>
              )}

              <iframe
                key={row.url}
                src={row.url}
                title={`${row.domain} live preview`}
                className="h-full w-full border-0"
                style={{ minHeight: '400px' }}
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
                referrerPolicy="no-referrer"
                onLoad={() => {
                  setIframeLoaded(true)
                  if (timerRef.current) clearTimeout(timerRef.current)
                }}
                onError={() => {
                  setIframeBlocked(true)
                  if (timerRef.current) clearTimeout(timerRef.current)
                }}
              />
            </div>

            <div className="mt-2 flex items-center justify-between text-xs text-[var(--sea-ink-soft)]">
              <span className="truncate">
                <a
                  href={row.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:underline"
                >
                  {row.url}
                </a>
              </span>
              <span>
                Scanned: {row.timestamp ? new Date(row.timestamp).toLocaleString() : '—'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}