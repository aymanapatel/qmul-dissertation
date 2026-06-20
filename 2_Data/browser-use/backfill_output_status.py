import argparse
import csv
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOMAINS_CSV = BASE_DIR / "domains.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "axe-core"


def safe_dir_name(domain: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", domain).strip("._")


def read_domain_statuses(csv_path: Path) -> dict[str, dict[str, str]]:
    statuses: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            domain = row.get("domain", "").strip()
            if not domain:
                continue
            statuses[domain] = {
                "scrapped_first": row.get("scrapped_first", "").strip(),
                "scrapped_full": row.get("scrapped_full", "").strip(),
                "notes": row.get("notes", "").strip(),
            }
    return statuses


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {
            "total_pages": 0,
            "total_violations": 0,
            "by_rule": {},
            "by_page": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def backfill(domains_csv: Path, output_dir: Path, create_missing: bool) -> dict[str, int]:
    statuses = read_domain_statuses(domains_csv)
    counts = {
        "updated": 0,
        "unchanged": 0,
        "created": 0,
        "skipped_missing_output": 0,
        "skipped_no_status": 0,
    }

    for domain, status in statuses.items():
        first_status = status["scrapped_first"]
        full_status = status["scrapped_full"]
        notes = status["notes"]
        if not first_status and not full_status and not notes:
            counts["skipped_no_status"] += 1
            continue

        summary_path = output_dir / safe_dir_name(domain) / "summary.json"
        if not summary_path.exists() and not create_missing:
            counts["skipped_missing_output"] += 1
            continue

        existed = summary_path.exists()
        summary = load_summary(summary_path)
        before = json.dumps(summary, sort_keys=True, ensure_ascii=False)

        if first_status:
            summary["scrapped_first"] = first_status
        if full_status:
            summary["scrapped_full"] = full_status
        if notes:
            summary["notes"] = notes

        after = json.dumps(summary, sort_keys=True, ensure_ascii=False)
        if before == after:
            counts["unchanged"] += 1
            continue

        write_summary(summary_path, summary)
        if existed:
            counts["updated"] += 1
        else:
            counts["created"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill outputs/axe-core summary.json status fields from domains.csv."
    )
    parser.add_argument("--domains-csv", type=Path, default=DEFAULT_DOMAINS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create placeholder summary.json files for CSV rows with status but no existing output.",
    )
    args = parser.parse_args()

    counts = backfill(args.domains_csv, args.output_dir, args.create_missing)
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
