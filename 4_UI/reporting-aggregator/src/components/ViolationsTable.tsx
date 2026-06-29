import { useState, useEffect, useCallback } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
  type ColumnFiltersState,
  type PaginationState,
} from '@tanstack/react-table'
import type { ViolationRow, ImpactKey, PaginatedResult } from '../lib/types'
import { IMPACT_ORDER } from '../lib/types'
import { getViolations } from '../server/queries'
import ImpactBadge from './ImpactBadge'
import TagChips from './TagChips'
import NodeSamplesDrawer from './NodeSamplesDrawer'

const columnHelper = createColumnHelper<ViolationRow>()

const columns = [
  columnHelper.accessor('domain', {
    header: 'Site',
    cell: (info) => (
      <a
        href={`/sites/${encodeURIComponent(info.getValue())}`}
        className="font-semibold text-[var(--lagoon-deep)] hover:underline"
      >
        {info.getValue()}
      </a>
    ),
  }),
  columnHelper.accessor('rule_id', {
    header: 'Rule',
    cell: (info) => <code className="text-xs">{info.getValue()}</code>,
  }),
  columnHelper.accessor('impact', {
    header: 'Impact',
    cell: (info) => <ImpactBadge impact={info.getValue() as ImpactKey} />,
  }),
  columnHelper.accessor('description', {
    header: 'Description',
    cell: (info) => (
      <span className="text-sm text-[var(--sea-ink-soft)]">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor('node_count', {
    header: 'Nodes',
    cell: (info) => (
      <span className="font-semibold text-[var(--sea-ink)]">{info.getValue().toLocaleString()}</span>
    ),
  }),
  columnHelper.accessor('tags', {
    header: 'Tags',
    enableSorting: false,
    cell: (info) => <TagChips tags={info.getValue()} />,
  }),
]

export default function ViolationsTable() {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'impact', desc: false }])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 25 })
  const [data, setData] = useState<PaginatedResult<ViolationRow> | null>(null)
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<ViolationRow | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    const sort = sorting[0]
      ? {
          field: sorting[0].id as 'domain' | 'rule_id' | 'impact' | 'node_count' | 'description',
          direction: (sorting[0].desc ? 'desc' : 'asc') as 'asc' | 'desc',
        }
      : undefined
    const filters: Record<string, unknown> = {}
    const domainF = columnFilters.find((f) => f.id === 'domain')?.value as string
    const ruleF = columnFilters.find((f) => f.id === 'rule_id')?.value as string
    const descF = columnFilters.find((f) => f.id === 'description')?.value as string
    const impactsF = columnFilters.find((f) => f.id === 'impacts')?.value as ImpactKey[]
    const tagsF = columnFilters.find((f) => f.id === 'tags')?.value as string[]
    const minNodes = columnFilters.find((f) => f.id === 'node_count')?.value as number
    if (domainF) filters.domain = domainF
    if (ruleF) filters.ruleId = ruleF
    if (descF) filters.description = descF
    if (impactsF && impactsF.length > 0) filters.impacts = impactsF
    if (tagsF && tagsF.length > 0) filters.tags = tagsF
    if (minNodes) filters.minNodeCount = minNodes
    const result = await getViolations({
      data: {
        sort,
        filters: Object.keys(filters).length > 0 ? filters : undefined,
        pagination: { page: pagination.pageIndex + 1, pageSize: pagination.pageSize },
      },
    })
    setData(result)
    setLoading(false)
  }, [sorting, columnFilters, pagination])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const toggleImpact = useCallback((impact: ImpactKey) => {
    setColumnFilters((prev) => {
      const existing = prev.find((f) => f.id === 'impacts')
      const current = (existing?.value as ImpactKey[]) ?? []
      const next = current.includes(impact)
        ? current.filter((i) => i !== impact)
        : [...current, impact]
      const rest = prev.filter((f) => f.id !== 'impacts')
      if (next.length > 0) rest.push({ id: 'impacts', value: next })
      return rest
    })
  }, [])

  const toggleTag = useCallback((tag: string) => {
    setColumnFilters((prev) => {
      const existing = prev.find((f) => f.id === 'tags')
      const current = (existing?.value as string[]) ?? []
      const next = current.includes(tag) ? current.filter((t) => t !== tag) : [...current, tag]
      const rest = prev.filter((f) => f.id !== 'tags')
      if (next.length > 0) rest.push({ id: 'tags', value: next })
      return rest
    })
  }, [])

  const table = useReactTable({
    data: data?.rows ?? [],
    columns,
    state: { sorting, columnFilters, pagination },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    manualPagination: true,
    manualSorting: true,
    manualFiltering: true,
    pageCount: data?.totalPages ?? -1,
  })

  const domainF = columnFilters.find((f) => f.id === 'domain')?.value as string ?? ''
  const ruleF = columnFilters.find((f) => f.id === 'rule_id')?.value as string ?? ''
  
  const impactsF = (columnFilters.find((f) => f.id === 'impacts')?.value as ImpactKey[]) ?? []
  const tagsF = (columnFilters.find((f) => f.id === 'tags')?.value as string[]) ?? []
  const minNodes = columnFilters.find((f) => f.id === 'node_count')?.value as number ?? 0

  const commonTags = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'ACT', 'section508']

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-semibold text-[var(--sea-ink-soft)]">Site</span>
          <input
            type="text"
            value={domainF}
            onChange={(e) => {
              const v = e.target.value
              setColumnFilters((prev) => {
                const next = prev.filter((f) => f.id !== 'domain')
                if (v) next.push({ id: 'domain', value: v })
                return next
              })
            }}
            placeholder="domain…"
            className="w-40 rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-sm text-[var(--sea-ink)] outline-none focus:border-[var(--lagoon-deep)]"
          />
        </label>
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-semibold text-[var(--sea-ink-soft)]">Rule</span>
          <input
            type="text"
            value={ruleF}
            onChange={(e) => {
              const v = e.target.value
              setColumnFilters((prev) => {
                const next = prev.filter((f) => f.id !== 'rule_id')
                if (v) next.push({ id: 'rule_id', value: v })
                return next
              })
            }}
            placeholder="e.g. color-contrast"
            className="w-44 rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-sm text-[var(--sea-ink)] outline-none focus:border-[var(--lagoon-deep)]"
          />
        </label>
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-semibold text-[var(--sea-ink-soft)]">Min nodes</span>
          <input
            type="number"
            min="0"
            value={minNodes || ''}
            onChange={(e) => {
              const v = Number(e.target.value) || 0
              setColumnFilters((prev) => {
                const next = prev.filter((f) => f.id !== 'node_count')
                if (v > 0) next.push({ id: 'node_count', value: v })
                return next
              })
            }}
            placeholder="0"
            className="w-20 rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-sm text-[var(--sea-ink)] outline-none focus:border-[var(--lagoon-deep)]"
          />
        </label>
        <span className="ml-auto text-sm text-[var(--sea-ink-soft)]">
          {loading ? 'Loading…' : `${data?.total.toLocaleString() ?? 0} rows`}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-[var(--sea-ink-soft)]">Impact:</span>
        {IMPACT_ORDER.filter((i) => i !== 'none').map((impact) => {
          const active = impactsF.includes(impact)
          return (
            <button
              key={impact}
              type="button"
              onClick={() => toggleImpact(impact)}
              className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold capitalize transition ${
                active
                  ? 'border-[var(--lagoon-deep)] bg-[var(--lagoon)] text-white'
                  : 'border-[var(--line)] bg-[var(--chip-bg)] text-[var(--sea-ink-soft)] hover:bg-[var(--link-bg-hover)]'
              }`}
            >
              {impact}
            </button>
          )
        })}
        <span className="mx-2 h-4 w-px bg-[var(--line)]" />
        <span className="text-xs font-semibold text-[var(--sea-ink-soft)]">Tag:</span>
        {commonTags.map((tag) => {
          const active = tagsF.includes(tag)
          return (
            <button
              key={tag}
              type="button"
              onClick={() => toggleTag(tag)}
              className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold transition ${
                active
                  ? 'border-[var(--lagoon-deep)] bg-[var(--lagoon)] text-white'
                  : 'border-[var(--line)] bg-[var(--chip-bg)] text-[var(--sea-ink-soft)] hover:bg-[var(--link-bg-hover)]'
              }`}
            >
              {tag}
            </button>
          )
        })}
      </div>

      <div className="demo-table-shell">
        <table className="demo-table">
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((h) => {
                  const sortable = h.column.getCanSort()
                  const sortDir = h.column.getIsSorted()
                  return (
                    <th
                      key={h.id}
                      onClick={sortable ? h.column.getToggleSortingHandler() : undefined}
                      className={sortable ? 'cursor-pointer select-none' : ''}
                    >
                      <div className="flex items-center gap-1">
                        {flexRender(h.column.columnDef.header, h.getContext())}
                        {sortable && (
                          <span className="text-xs">
                            {sortDir === 'asc' ? ' ▲' : sortDir === 'desc' ? ' ▼' : ' ↕'}
                          </span>
                        )}
                      </div>
                    </th>
                  )
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={columns.length} className="py-10 text-center text-[var(--sea-ink-soft)]">
                  Loading…
                </td>
              </tr>
            )}
            {!loading && table.getRowModel().rows.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="py-10 text-center text-[var(--sea-ink-soft)]">
                  No violations match the current filters.
                </td>
              </tr>
            )}
            {!loading &&
              table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  onClick={() => setSelected(row.original)}
                  className="cursor-pointer"
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                  ))}
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-sm text-[var(--sea-ink-soft)]">
        <button
          className="demo-button demo-button-secondary"
          onClick={() => table.previousPage()}
          disabled={!table.getCanPreviousPage()}
        >
          ← Prev
        </button>
        <span>
          Page {pagination.pageIndex + 1} of {data?.totalPages ?? 1}
        </span>
        <button
          className="demo-button demo-button-secondary"
          onClick={() => table.nextPage()}
          disabled={!table.getCanNextPage()}
        >
          Next →
        </button>
        <select
          value={pagination.pageSize}
          onChange={(e) => setPagination((p) => ({ ...p, pageSize: Number(e.target.value), pageIndex: 0 }))}
          className="rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] px-2 py-1 text-sm"
        >
          {[10, 25, 50, 100].map((n) => (
            <option key={n} value={n}>
              {n} / page
            </option>
          ))}
        </select>
      </div>

      {selected && (
        <NodeSamplesDrawer row={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  )
}