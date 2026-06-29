export type ImpactKey = 'critical' | 'serious' | 'moderate' | 'minor' | 'none'

export const IMPACT_ORDER: ImpactKey[] = [
  'critical',
  'serious',
  'moderate',
  'minor',
  'none',
]

export const IMPACT_RANK: Record<ImpactKey, number> = {
  critical: 0,
  serious: 1,
  moderate: 2,
  minor: 3,
  none: 4,
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

export interface SitesFile {
  generated_at: string
  count: number
  sites: SiteRow[]
}

export interface ViolationRow {
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

export interface ViolationsFile {
  generated_at: string
  count: number
  sites_count: number
  violations: ViolationRow[]
}

export type SortDirection = 'asc' | 'desc'

export interface SortSpec {
  field: string
  direction: SortDirection
}

export interface PaginationSpec {
  page: number
  pageSize: number
}

export interface SiteFilters {
  domain?: string
  url?: string
  minViolations?: number
  maxViolations?: number
  hasCritical?: boolean
  cleanOnly?: boolean
  rule?: string
}

export interface ViolationFilters {
  domain?: string
  ruleId?: string
  description?: string
  impacts?: ImpactKey[]
  tags?: string[]
  minNodeCount?: number
}

export interface PaginatedResult<T> {
  rows: T[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

export interface SummaryStats {
  totalSites: number
  totalViolations: number
  criticalSites: number
  cleanSites: number
  avgViolations: number
  totalRuleViolations: number
  topRules: Array<{ rule: string; count: number }>
  impactTotals: Record<ImpactKey, number>
  worstSites: Array<{ domain: string; total_violations: number }>
  cleanestSites: Array<{ domain: string; total_violations: number }>
}

export interface SiteDetail {
  site: SiteRow
  violations: ViolationRow[]
}