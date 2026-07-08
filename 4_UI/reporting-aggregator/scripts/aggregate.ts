import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(__dirname, '..')
const AXE_DIR = resolve(ROOT, '../../2_Data/browser-use/outputs/axe-core')
const OUT_DIR = resolve(ROOT, 'public/data')

const PAGE_FILE_RE = /^page-\d+_.*\.json$/

interface AxeCheck {
  id: string
  impact: string | null
  tags: string[]
  description: string
  help: string
  helpUrl: string
  nodes: Array<{
    html: string
    target: string[]
    impact: string | null
    failureSummary?: string
    any?: Array<{ id: string; message: string }>
    all?: Array<{ id: string; message: string }>
    none?: Array<{ id: string; message: string }>
  }>
}

interface AxePage {
  url: string
  timestamp: string
  violations: AxeCheck[]
  passes: AxeCheck[]
  incomplete: AxeCheck[]
  inapplicable: AxeCheck[]
}

interface Summary {
  total_pages: number
  total_violations: number
  by_rule: Record<string, number>
  by_page?: Array<{ page_index: number; url: string; violations: number }>
  scrapped_first?: string
}

type ImpactKey = 'critical' | 'serious' | 'moderate' | 'minor' | 'none'

const IMPACTS: ImpactKey[] = ['critical', 'serious', 'moderate', 'minor', 'none']

interface ViolationRow {
  domain: string
  url: string
  timestamp: string
  rule_id: string
  impact: ImpactKey
  description: string
  help: string
  helpUrl: string
  tags: string[]
  node_count: number
  sample_html: string
  sample_target: string
  sample_failure_summary: string
}

export const impactRank: Record<string, number> = {
  critical: 0,
  serious: 1,
  moderate: 2,
  minor: 3,
  none: 4,
}

function impactKey(s: string | null): ImpactKey {
  if (!s) return 'none'
  return IMPACTS.includes(s as ImpactKey) ? (s as ImpactKey) : 'none'
}

export interface SiteRow {
  domain: string
  url: string
  timestamp: string
  total_pages: number
  total_violations: number
  impact_counts: Record<ImpactKey, number>
  rules_violated: number
  top_rules: Array<{ rule: string; count: number }>
  scraps_failed: boolean
}

async function main() {
  mkdirSync(OUT_DIR, { recursive: true })
  const entries = readdirSync(AXE_DIR, { withFileTypes: true })
  const sites: SiteRow[] = []
  const violations: ViolationRow[] = []
  let processed = 0
  let skipped = 0

  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    if (entry.name === '.DS_Store') continue
    const domain = entry.name
    const dir = join(AXE_DIR, domain)
    const summaryPath = join(dir, 'summary.json')
    let summary: Summary | null = null
    try {
      summary = JSON.parse(readFileSync(summaryPath, 'utf-8')) as Summary
    } catch {
      summary = null
    }

    // Find page files - prefer page-0_home.json but handle others
    const files = readdirSync(dir).filter((f) => PAGE_FILE_RE.test(f)).sort()
    if (files.length === 0 && !summary) {
      skipped++
      continue
    }

    let lastUrl = ''
    let lastTimestamp = ''
    const pageImpactCounts: Record<ImpactKey, number> = {
      critical: 0,
      serious: 0,
      moderate: 0,
      minor: 0,
      none: 0,
    }
    const rulesByCount = new Map<string, number>()
    let totalPages = 0
    let totalViolations = 0
    let scrapsFailed = false
    const perSiteViolations = new Map<string, ViolationRow>()

    if (summary) {
      totalPages = summary.total_pages
      totalViolations = summary.total_violations
      scrapsFailed = summary.scrapped_first === 'no'
      for (const [rule, count] of Object.entries(summary.by_rule || {})) {
        rulesByCount.set(rule, count)
      }
    }

    for (const file of files) {
      try {
        const data = JSON.parse(readFileSync(join(dir, file), 'utf-8')) as AxePage
        if (!lastUrl) lastUrl = data.url
        if (!lastTimestamp) lastTimestamp = data.timestamp
        // Fallback counts when summary missing
        if (!summary) {
          totalPages++
          totalViolations += data.violations?.length || 0
        }
        // Build violation rows + impact counts
        for (const v of data.violations || []) {
          const ik = impactKey(v.impact)
          if (!summary) {
            rulesByCount.set(v.id, (rulesByCount.get(v.id) || 0) + v.nodes.length)
          }
          const nodeCount = v.nodes?.length || 0
          // Per-check tuple (site x rule) aggregations
          const existing = perSiteViolations.get(v.id)
          if (existing) {
            existing.node_count += nodeCount
            if (impactRank[ik] < impactRank[existing.impact]) {
              existing.impact = ik
            }
          } else {
            const firstNode = v.nodes?.[0]
            perSiteViolations.set(v.id, {
              domain,
              url: data.url,
              timestamp: data.timestamp,
              rule_id: v.id,
              impact: ik,
              description: v.description || '',
              help: v.help || '',
              helpUrl: v.helpUrl || '',
              tags: v.tags || [],
              node_count: nodeCount,
              sample_html: firstNode?.html || '',
              sample_target: firstNode?.target?.join(' | ') || '',
              sample_failure_summary: firstNode?.failureSummary || '',
            })
          }
          pageImpactCounts[ik] += nodeCount
        }
      } catch {
        // skip unreadable page file
      }
    }

    // Final pass: ensure impact counts sum to totalViolations if summary had mismatches
    if (summary) {
      const counted =
        pageImpactCounts.critical +
        pageImpactCounts.serious +
        pageImpactCounts.moderate +
        pageImpactCounts.minor +
        pageImpactCounts.none
      // Only use pageImpactCounts if files present and they sum > 0; else leave zeros
      if (counted === 0 && totalViolations > 0 && files.length === 0) {
        // no page files, approximate via none (unknown) — flag via top_rules
        pageImpactCounts.none = totalViolations
      }
    }

    const topRules = Array.from(rulesByCount.entries())
      .map(([rule, count]) => ({ rule, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5)

    sites.push({
      domain,
      url: lastUrl || 'unknown',
      timestamp: lastTimestamp || '',
      total_pages: totalPages,
      total_violations: totalViolations,
      impact_counts: pageImpactCounts,
      rules_violated: rulesByCount.size,
      top_rules: topRules,
      scraps_failed: scrapsFailed,
    })

    for (const row of perSiteViolations.values()) {
      violations.push(row)
    }
    processed++
  }

  sites.sort((a, b) => b.total_violations - a.total_violations)
  violations.sort((a, b) => impactRank[a.impact] - impactRank[b.impact] || b.node_count - a.node_count)

  const siteOut = { generated_at: new Date().toISOString(), count: sites.length, sites }
  const violationsOut = {
    generated_at: new Date().toISOString(),
    count: violations.length,
    sites_count: sites.length,
    violations,
  }

  writeFileSync(join(OUT_DIR, 'sites.json'), JSON.stringify(siteOut))
  writeFileSync(join(OUT_DIR, 'violations.json'), JSON.stringify(violationsOut))

  const totalV = sites.reduce((s, r) => s + r.total_violations, 0)
  console.log(`Processed: ${processed}, Skipped: ${skipped}`)
  console.log(`Sites: ${sites.length}, Violation rows: ${violations.length}`)
  console.log(`Sum total_violations: ${totalV}`)
  console.log(`Wrote: ${OUT_DIR}/sites.json, ${OUT_DIR}/violations.json`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
