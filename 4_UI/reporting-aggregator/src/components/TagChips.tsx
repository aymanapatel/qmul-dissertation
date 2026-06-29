export default function TagChips({ tags, max = 6 }: { tags: string[]; max?: number }) {
  if (!tags || tags.length === 0) return null
  const shown = tags.slice(0, max)
  const extra = tags.length - shown.length
  return (
    <div className="flex flex-wrap gap-1">
      {shown.map((tag) => (
        <span
          key={tag}
          className="inline-block rounded border border-[var(--line)] bg-[var(--chip-bg)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--sea-ink-soft)]"
        >
          {tag}
        </span>
      ))}
      {extra > 0 && (
        <span className="inline-block px-1 py-0.5 text-[10px] font-medium text-[var(--sea-ink-soft)]">
          +{extra}
        </span>
      )}
    </div>
  )
}