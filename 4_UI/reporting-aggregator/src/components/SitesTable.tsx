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
import type { SiteRow, PaginatedResult } from '../lib/types'
import { getSites } from '../server/queries'
import ImpactBadge from './ImpactBadge'

const columnHelper = createColumnHelper<SiteRow>()

const columns = [
  columnHelper.accessor('domain', {
    header: 'Domain',
    cell: (info) => {
      const domain = info.getValue()
      return (
        <a
          href={`/sites/${encodeURIComponent(domain)}`}
          className="font-semibold text-[var(--lagoon-deep)] hover:underline"
        >
          {domain}
        </a>
      )
    },
  }),
  columnHelper.accessor('url', {
    header: 'URL',
    cell: (info) => (
      <a
        href={info.getValue()}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm text-[var(--sea-ink-soft)] hover:underline"
      >
        {info.getValue()}
      </a>
    ),
  }),
  columnHelper.accessor('total_violations', {
    header: 'Total',
    cell: (info) => (
      <span className="font-semibold text-[var(--sea-ink)]">
        {info.getValue().toLocaleString()}
      </span>
    ),
  }),
  columnHelper.accessor((row) => row.impact_counts.critical, {
    id: 'critical',
    header: 'Critical',
    cell: (info) =>
      info.getValue() > 0 ? <ImpactBadge impact="critical" count={info.getValue()} /> : <span className="text-[var(--sea-ink-soft)]">—</span>,
  }),
  columnHelper.accessor((row) => row.impact_counts.serious, {
    id: 'serious',
    header: 'Serious',
    cell: (info) =>
      info.getValue() > 0 ? <ImpactBadge impact="serious" count={info.getValue()} /> : <span className="text-[var(--sea-ink-soft)]">—</span>,
  }),
  columnHelper.accessor((row) => row.impact_counts.moderate, {
    id: 'moderate',
    header: 'Moderate',
    cell: (info) =>
      info.getValue() > 0 ? <ImpactBadge impact="moderate" count={info.getValue()} /> : <span className="text-[var(--sea-ink-soft)]">—</span>,
  }),
  columnHelper.accessor((row) => row.impact_counts.minor, {
    id: 'minor',
    header: 'Minor',
    cell: (info) =>
      info.getValue() > 0 ? <ImpactBadge impact="minor" count={info.getValue()} /> : <span className="text-[var(--sea-ink-soft)]">—</span>,
  }),
  columnHelper.accessor('rules_violated', {
    header: 'Rules',
    cell: (info) => info.getValue(),
  }),
  columnHelper.accessor('total_pages', {
    header: 'Pages',
    cell: (info) => info.getValue(),
  }),
  columnHelper.accessor('timestamp', {
    header: 'Scanned',
    cell: (info) => {
      const v = info.getValue()
      if (!v) return <span className="text-[var(--sea-ink-soft)]">—</span>
      return <span className="text-xs text-[var(--sea-ink-soft)]">{new Date(v).toLocaleDateString()}</span>
    },
  }),
]

declare module '@tanstack/react-table' {
  interface SortingFns {}
}

export default function SitesTable() {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'total_violations', desc: true }])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 25 })
  const [data, setData] = useState<PaginatedResult<SiteRow> | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchData = useCallback(async () => {
    setLoading(true)
    const sort = sorting[0]
      ? {
          field: sorting[0].id as
            | 'domain'
            | 'url'
            | 'total_violations'
            | 'rules_violated'
            | 'critical'
            | 'serious'
            | 'moderate'
            | 'minor'
            | 'timestamp'
            | 'total_pages',
          direction: (sorting[0].desc ? 'desc' : 'asc') as 'asc' | 'desc',
        }
      : undefined
    const filters: Record<string, unknown> = {}
    for (const f of columnFilters) {
      if (f.id === 'domain' || f.id === 'url') filters[f.id] = String(f.value ?? '')
      if (f.id === 'total_violations') filters.minViolations = Number(f.value) || 0
      if (f.id === 'hasCritical' && f.value) filters.hasCritical = true
      if (f.id === 'cleanOnly' && f.value) filters.cleanOnly = true
    }
    const result = await getSites({
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

  const domainFilter = columnFilters.find((f) => f.id === 'domain')?.value as string ?? ''
  const minViolations = columnFilters.find((f) => f.id === 'total_violations')?.value as number ?? 0
  const hasCritical = columnFilters.find((f) => f.id === 'hasCritical')?.value as boolean ?? false
  const cleanOnly = columnFilters.find((f) => f.id === 'cleanOnly')?.value as boolean ?? false

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-semibold text-[var(--sea-ink-soft)]">Domain contains</span>
          <input
            type="text"
            value={domainFilter}
            onChange={(e) => {
              const v = e.target.value
              setColumnFilters((prev) => {
                const next = prev.filter((f) => f.id !== 'domain')
                if (v) next.push({ id: 'domain', value: v })
                return next
              })
            }}
            placeholder="e.g. google"
            className="w-48 rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-sm text-[var(--sea-ink)] outline-none focus:border-[var(--lagoon-deep)]"
          />
        </label>
        <label className="flex flex-col text-xs">
          <span className="mb-1 font-semibold text-[var(--sea-ink-soft)]">Min violations</span>
          <input
            type="number"
            min="0"
            value={minViolations || ''}
            onChange={(e) => {
              const v = Number(e.target.value) || 0
              setColumnFilters((prev) => {
                const next = prev.filter((f) => f.id !== 'total_violations')
                if (v > 0) next.push({ id: 'total_violations', value: v })
                return next
              })
            }}
            placeholder="0"
            className="w-24 rounded-lg border border-[var(--line)] bg-[var(--chip-bg)] px-3 py-1.5 text-sm text-[var(--sea-ink)] outline-none focus:border-[var(--lagoon-deep)]"
          />
        </label>
        <label className="flex items-center gap-2 text-sm text-[var(--sea-ink)]">
          <input
            type="checkbox"
            checked={hasCritical}
            onChange={(e) => {
              const v = e.target.checked
              setColumnFilters((prev) => {
                const next = prev.filter((f) => f.id !== 'hasCritical')
                if (v) next.push({ id: 'hasCritical', value: true })
                return next
              })
            }}
          />
          Has critical
        </label>
        <label className="flex items-center gap-2 text-sm text-[var(--sea-ink)]">
          <input
            type="checkbox"
            checked={cleanOnly}
            onChange={(e) => {
              const v = e.target.checked
              setColumnFilters((prev) => {
                const next = prev.filter((f) => f.id !== 'cleanOnly')
                if (v) next.push({ id: 'cleanOnly', value: true })
                return next
              })
            }}
          />
          Clean only
        </label>
        <span className="ml-auto text-sm text-[var(--sea-ink-soft)]">
          {loading ? 'Loading…' : `${data?.total.toLocaleString() ?? 0} sites`}
        </span>
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
                      style={{ width: h.getSize() !== 150 ? h.getSize() : undefined }}
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
                  No sites match the current filters.
                </td>
              </tr>
            )}
            {!loading &&
              table.getRowModel().rows.map((row) => (
                <tr key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
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
    </div>
  )
}