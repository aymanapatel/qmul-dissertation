import type { ImpactKey } from '../lib/types'

const IMPACT_STYLES: Record<ImpactKey, string> = {
  critical:
    'bg-red-100 text-red-800 border-red-300 dark:bg-red-900/40 dark:text-red-200 dark:border-red-700',
  serious:
    'bg-orange-100 text-orange-800 border-orange-300 dark:bg-orange-900/40 dark:text-orange-200 dark:border-orange-700',
  moderate:
    'bg-yellow-100 text-yellow-800 border-yellow-300 dark:bg-yellow-900/40 dark:text-yellow-200 dark:border-yellow-700',
  minor:
    'bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900/40 dark:text-blue-200 dark:border-blue-700',
  none:
    'bg-gray-100 text-gray-700 border-gray-300 dark:bg-gray-800 dark:text-gray-300 dark:border-gray-600',
}

export default function ImpactBadge({
  impact,
  count,
}: {
  impact: ImpactKey
  count?: number
}) {
  const label = count !== undefined ? `${impact}: ${count.toLocaleString()}` : impact
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold capitalize ${IMPACT_STYLES[impact]}`}
    >
      {label}
    </span>
  )
}