# Scanner

Web accessibility scanner using axe-core + Playwright. Batch-scan domains from a CSV and extract WCAG violation metrics.

## Setup

```bash
cd 2_Data/personal/scanner
uv sync
uv run python -m playwright install chromium
```

## Usage

### Single URL

```bash
uv run scanner single https://example.com
uv run scanner single example.com --json
```

### Batch CSV scan

```bash
uv run scanner scan ../data.csv
uv run scanner scan ../data.csv -n 10            # limit to 10 domains
uv run scanner scan ../data.csv -s 50 -n 100     # rows 50–149
uv run scanner scan ../data.csv --skip-populated # skip already-scanned rows
uv run scanner scan ../data.csv --json           # also save raw axe JSON
uv run scanner scan ../data.csv -o results.csv   # custom output path
```

## Output columns

| Column | Description |
|---|---|
| `popularity_rank` | From input CSV |
| `domain` | From input CSV |
| `Populated` | Set to `1` after scan |
| `Accessibility_Rank_2026` | Reserved for ranking data |
| `Accessibility_Rank_2025` | Reserved for ranking data |
| `Accessibility_Rank_2024` | Reserved for ranking data |
| `Accessibility_Rank_2023` | Reserved for ranking data |
| `Number_of_accessibility_errors_detected` | Total violation nodes found |
| `WCAG_2_A/AA_failure_detected` | Count of WCAG 2 A/AA violations |
| `Number_of_page_elements` | Passes + violations |
| `Error_density_Percentage` | Violations / elements × 100 |
| `Top_error_types_detected` | Top 5 violation IDs by node count |