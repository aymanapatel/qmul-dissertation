import { createServerFn } from '@tanstack/react-start'
import { z } from 'zod'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type {
  SiteRow,
  SitesFile,
  ViolationRow,
  ViolationsFile,
  PaginatedResult,
  SummaryStats,
  ImpactKey,
  SiteDetail,
} from '../lib/types'
import { IMPACT_RANK } from '../lib/types'

let sitesCache: SiteRow[] | null = null
let violationsCache: ViolationRow[] | null = null

const ROOT = resolve(process.cwd())

function loadSites(): SiteRow[] {
  if (sitesCache) return sitesCache
  try {
    const raw = readFileSync(resolve(ROOT, 'public/data/sites.json'), 'utf-8')
    const parsed = JSON.parse(raw) as SitesFile
    sitesCache = parsed.sites
    return sitesCache
  } catch (e) {
    console.error('Failed to load sites.json', e)
    sitesCache = []
    return sitesCache
  }
}

function loadViolations(): ViolationRow[] {
  if (violationsCache) return violationsCache
  try {
    const raw = readFileSync(resolve(ROOT, 'public/data/violations.json'), 'utf-8')
    const parsed = JSON.parse(raw) as ViolationsFile
    violationsCache = parsed.violations
    return violationsCache
  } catch (e) {
    console.error('Failed to load violations.json', e)
    violationsCache = []
    return violationsCache
  }
}

function compare(a: unknown, b: unknown, direction: 'asc' | 'desc'): number {
  let cmp = 0
  if (typeof a === 'number' && typeof b === 'number') {
    cmp = a - b
  } else if (a == null && b == null) {
    cmp = 0
  } else if (a == null) {
    cmp = -1
  } else if (b == null) {
    cmp = 1
  } else {
    cmp = String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' })
  }
  return direction === 'asc' ? cmp : -cmp
}

function paginate<T>(rows: T[], page: number, pageSize: number): PaginatedResult<T> {
  const total = rows.length
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const safePage = Math.min(Math.max(1, page), totalPages)
  const start = (safePage - 1) * pageSize
  const slice = rows.slice(start, start + pageSize)
  return { rows: slice, total, page: safePage, pageSize, totalPages }
}

const SiteSortSchema = z.object({
  field: z.enum([
    'domain',
    'url',
    'total_violations',
    'rules_violated',
    'critical',
    'serious',
    'moderate',
    'minor',
    'timestamp',
    'total_pages',
  ]),
  direction: z.enum(['asc', 'desc']),
})

const SiteFiltersSchema = z.object({
  domain: z.string().optional(),
  url: z.string().optional(),
  minViolations: z.number().optional(),
  maxViolations: z.number().optional(),
  hasCritical: z.boolean().optional(),
  cleanOnly: z.boolean().optional(),
  rule: z.string().optional(),
})

const PaginationSchema = z.object({
  page: z.number().min(1).default(1),
  pageSize: z.number().min(1).max(500).default(50),
})

const SitesQuerySchema = z.object({
  sort: SiteSortSchema.optional(),
  filters: SiteFiltersSchema.optional(),
  pagination: PaginationSchema.optional(),
})

const ViolationSortSchema = z.object({
  field: z.enum(['domain', 'rule_id', 'impact', 'node_count', 'description']),
  direction: z.enum(['asc', 'desc']),
})

const ViolationFiltersSchema = z.object({
  domain: z.string().optional(),
  ruleId: z.string().optional(),
  description: z.string().optional(),
  impacts: z.array(z.string()).optional(),
  tags: z.array(z.string()).optional(),
  minNodeCount: z.number().optional(),
})

const ViolationsQuerySchema = z.object({
  sort: ViolationSortSchema.optional(),
  filters: ViolationFiltersSchema.optional(),
  pagination: PaginationSchema.optional(),
})

function impactKey(s: string): ImpactKey {
  const k = s as ImpactKey
  return IMPACT_RANK[k] !== undefined ? k : 'none'
}

export const getSites = createServerFn({ method: 'GET' })
  .validator(SitesQuerySchema)
  .handler(async ({ data }) => {
    const all = loadSites()
    let filtered = all
    if (data.filters) {
      const f = data.filters
      filtered = filtered.filter((row) => {
        if (f.domain && !row.domain.toLowerCase().includes(f.domain.toLowerCase())) return false
        if (f.url && !row.url.toLowerCase().includes(f.url.toLowerCase())) return false
        if (f.minViolations !== undefined && row.total_violations < f.minViolations) return false
        if (f.maxViolations !== undefined && row.total_violations > f.maxViolations) return false
        if (f.hasCritical && row.impact_counts.critical === 0) return false
        if (f.cleanOnly && row.total_violations > 0) return false
        if (f.rule && !row.top_rules.some((r) => r.rule.includes(f.rule!))) return false
        return true
      })
    }
    if (data.sort) {
      const { field, direction } = data.sort
      filtered = [...filtered].sort((a, b) => {
        let av: unknown
        let bv: unknown
        if (field === 'critical' || field === 'serious' || field === 'moderate' || field === 'minor') {
          av = a.impact_counts[field]
          bv = b.impact_counts[field]
        } else {
          av = (a as unknown as Record<string, unknown>)[field]
          bv = (b as unknown as Record<string, unknown>)[field]
        }
        return compare(av, bv, direction)
      })
    }
    const page = data.pagination?.page ?? 1
    const pageSize = data.pagination?.pageSize ?? 50
    return paginate(filtered, page, pageSize)
  })

export const getViolations = createServerFn({ method: 'GET' })
  .validator(ViolationsQuerySchema)
  .handler(async ({ data }) => {
    const all = loadViolations()
    let filtered = all
    if (data.filters) {
      const f = data.filters
      filtered = filtered.filter((row) => {
        if (f.domain && !row.domain.toLowerCase().includes(f.domain.toLowerCase())) return false
        if (f.ruleId && !row.rule_id.toLowerCase().includes(f.ruleId.toLowerCase())) return false
        if (f.description && !row.description.toLowerCase().includes(f.description.toLowerCase())) return false
        if (f.impacts && f.impacts.length > 0 && !f.impacts.includes(row.impact)) return false
        if (f.tags && f.tags.length > 0 && !f.tags.some((t) => row.tags.includes(t))) return false
        if (f.minNodeCount !== undefined && row.node_count < f.minNodeCount) return false
        return true
      })
    }
    if (data.sort) {
      const { field, direction } = data.sort
      filtered = [...filtered].sort((a, b) => {
        let av: unknown = (a as unknown as Record<string, unknown>)[field]
        let bv: unknown = (b as unknown as Record<string, unknown>)[field]
        if (field === 'impact') {
          av = IMPACT_RANK[impactKey(String(av))]
          bv = IMPACT_RANK[impactKey(String(bv))]
        }
        return compare(av, bv, direction)
      })
    }
    const page = data.pagination?.page ?? 1
    const pageSize = data.pagination?.pageSize ?? 50
    return paginate(filtered, page, pageSize)
  })

export const getSummaryStats = createServerFn({ method: 'GET' }).handler(async () => {
  const sites = loadSites()
  const violations = loadViolations()
  const totalSites = sites.length
  const totalViolations = sites.reduce((s, r) => s + r.total_violations, 0)
  const criticalSites = sites.filter((s) => s.impact_counts.critical > 0).length
  const cleanSites = sites.filter((s) => s.total_violations === 0).length
  const avgViolations = totalSites > 0 ? Math.round(totalViolations / totalSites) : 0

  const ruleCounts = new Map<string, number>()
  for (const v of violations) {
    ruleCounts.set(v.rule_id, (ruleCounts.get(v.rule_id) || 0) + v.node_count)
  }
  const topRules = Array.from(ruleCounts.entries())
    .map(([rule, count]) => ({ rule, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)

  const impactTotals: Record<ImpactKey, number> = {
    critical: 0,
    serious: 0,
    moderate: 0,
    minor: 0,
    none: 0,
  }
  for (const v of violations) {
    impactTotals[v.impact] += v.node_count
  }

  const worst = [...sites]
    .sort((a, b) => b.total_violations - a.total_violations)
    .slice(0, 5)
    .map((s) => ({ domain: s.domain, total_violations: s.total_violations }))
  const cleanest = sites
    .filter((s) => s.total_violations === 0)
    .slice(0, 5)
    .map((s) => ({ domain: s.domain, total_violations: 0 }))

  return {
    totalSites,
    totalViolations,
    criticalSites,
    cleanSites,
    avgViolations,
    totalRuleViolations: violations.length,
    topRules,
    impactTotals,
    worstSites: worst,
    cleanestSites: cleanest,
  } as SummaryStats
})

export const getSiteDetail = createServerFn({ method: 'GET' })
  .validator(z.object({ domain: z.string() }))
  .handler(async ({ data }) => {
    const sites = loadSites()
    const violations = loadViolations()
    const site = sites.find((s) => s.domain === data.domain)
    if (!site) return null
    const siteViolations = violations
      .filter((v) => v.domain === data.domain)
      .sort((a, b) => IMPACT_RANK[a.impact] - IMPACT_RANK[b.impact] || b.node_count - a.node_count)
    return { site, violations: siteViolations } as SiteDetail
  })

export const getAllViolationsForSite = createServerFn({ method: 'GET' })
  .validator(z.object({ domain: z.string() }))
  .handler(async ({ data }) => {
    const violations = loadViolations()
    return violations
      .filter((v) => v.domain === data.domain)
      .sort((a, b) => IMPACT_RANK[a.impact] - IMPACT_RANK[b.impact] || b.node_count - a.node_count)
  })