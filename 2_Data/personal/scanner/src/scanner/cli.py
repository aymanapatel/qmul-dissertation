import csv
import json
from pathlib import Path
from typing import Optional

import typer
from playwright.sync_api import sync_playwright
from axe_playwright_python.sync_playwright import Axe
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

app = typer.Typer(help="Web accessibility scanner using axe-core + Playwright")
console = Console()


def _scan_domain(domain: str, axe: Axe, playwright) -> Optional[dict]:
    url = f"https://{domain}"
    browser = playwright.chromium.launch()
    page = browser.new_page()
    try:
        page.goto(url, timeout=30000, wait_until="networkidle")
        axe_result = axe.run(page)
        return axe_result.response
    except Exception as e:
        console.print(f"[red]Error scanning {domain}: {e}[/red]")
        return None
    finally:
        browser.close()


def _extract_row(result: Optional[dict], domain: str) -> dict:
    if result is None:
        return {
            "domain": domain,
            "Number_of_accessibility_errors_detected": "",
            "WCAG_2_A/AA_failure_detected": "",
            "Number_of_page_elements": "",
            "Error_density_Percentage": "",
            "Top_error_types_detected": "",
        }

    violations = result.get("violations", [])
    wcag_aa_failures = [v for v in violations if any(
        "wcag2a" in tag or "wcag2aa" in tag
        for tag in v.get("tags", [])
    )]
    top_errors = sorted(violations, key=lambda v: len(v.get("nodes", [])), reverse=True)[:5]
    top_error_names = "; ".join(v.get("id", "") for v in top_errors)

    num_violations = sum(len(v.get("nodes", [])) for v in violations)
    num_elements = len(result.get("passes", [])) + num_violations
    error_density = round((num_violations / num_elements) * 100, 2) if num_elements > 0 else 0.0

    return {
        "domain": domain,
        "Accessibility_Rank_2026": "",
        "Accessibility_Rank_2025": "",
        "Accessibility_Rank_2024": "",
        "Accessibility_Rank_2023": "",
        "Number_of_accessibility_errors_detected": num_violations,
        "WCAG_2_A/AA_failure_detected": len(wcag_aa_failures),
        "Number_of_page_elements": num_elements,
        "Error_density_Percentage": error_density,
        "Top_error_types_detected": top_error_names,
    }


CSV_COLUMNS = [
    "popularity_rank",
    "domain",
    "Populated",
    "Accessibility_Rank_2026",
    "Accessibility_Rank_2025",
    "Accessibility_Rank_2024",
    "Accessibility_Rank_2023",
    "Number_of_accessibility_errors_detected",
    "WCAG_2_A/AA_failure_detected",
    "Number_of_page_elements",
    "Error_density_Percentage",
    "Top_error_types_detected",
]


@app.command()
def scan(
    csv_path: Path = typer.Argument(..., help="Path to the input CSV file"),
    output: Path = typer.Option(None, "-o", "--output", help="Output CSV path (default: <input>_results.csv)"),
    limit: int = typer.Option(None, "-n", "--limit", help="Limit number of domains to scan"),
    start: int = typer.Option(None, "-s", "--start", help="Start from this row number (1-based)"),
    skip_populated: bool = typer.Option(False, "--skip-populated", help="Skip rows already populated"),
    json_report: bool = typer.Option(False, "--json", help="Also save raw axe results as JSON"),
):
    """Scan domains from a CSV for accessibility violations using axe-core."""
    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        raise typer.Exit(1)

    if output is None:
        output = csv_path.with_name(csv_path.stem + "_results.csv")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if start:
        rows = rows[start - 1:]
    if limit:
        rows = rows[:limit]

    console.print(f"[bold]Scanning {len(rows)} domains[/bold]")

    axe = Axe()
    results_data = []
    raw_results = []

    with sync_playwright() as p:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning...", total=len(rows))
            for row in rows:
                domain = row.get("domain", "").strip()
                if not domain:
                    progress.advance(task)
                    continue
                if skip_populated and row.get("Populated", "").strip() == "1":
                    progress.advance(task)
                    continue

                progress.update(task, description=f"[cyan]{domain}[/cyan]")
                result = _scan_domain(domain, axe, p)

                extracted = _extract_row(result, domain)
                extracted["popularity_rank"] = row.get("popularity_rank", "")
                extracted["Populated"] = "1"
                results_data.append(extracted)

                if json_report and result:
                    raw_results.append({"domain": domain, "result": result})

                progress.advance(task)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rd in results_data:
            writer.writerow(rd)

    console.print(f"\n[green]Results saved to {output}[/green]")

    if json_report:
        json_path = output.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2)
        console.print(f"[green]Raw results saved to {json_path}[/green]")

    _print_summary(results_data)


def _print_summary(results: list[dict]):
    table = Table(title="Scan Summary")
    table.add_column("Domain", style="cyan", max_width=30)
    table.add_column("Violations", justify="right")
    table.add_column("WCAG A/AA", justify="right")
    table.add_column("Elements", justify="right")
    table.add_column("Error %", justify="right")
    table.add_column("Top Errors", max_width=40)

    for r in results[:20]:
        table.add_row(
            r.get("domain", ""),
            str(r.get("Number_of_accessibility_errors_detected", "")),
            str(r.get("WCAG_2_A/AA_failure_detected", "")),
            str(r.get("Number_of_page_elements", "")),
            str(r.get("Error_density_Percentage", "")),
            r.get("Top_error_types_detected", ""),
        )

    console.print(table)
    if len(results) > 20:
        console.print(f"... and {len(results) - 20} more. See output CSV for full results.")


@app.command()
def single(
    url: str = typer.Argument(..., help="URL to scan (e.g. https://example.com)"),
    json_report: bool = typer.Option(False, "--json", help="Output raw axe results as JSON"),
):
    """Scan a single URL for accessibility violations."""
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    console.print(f"[bold]Scanning {url}[/bold]")

    axe = Axe()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(url, timeout=30000, wait_until="networkidle")
            axe_result = axe.run(page)
            result = axe_result.response
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        finally:
            browser.close()

    if json_report:
        console.print_json(json.dumps(result, indent=2))
    else:
        _print_single_result(result)


def _print_single_result(result: dict):
    violations = result.get("violations", [])
    table = Table(title=f"Violations ({len(violations)})")
    table.add_column("ID", style="cyan")
    table.add_column("Description")
    table.add_column("Impact")
    table.add_column("Nodes", justify="right")

    for v in violations[:20]:
        table.add_row(
            v.get("id", ""),
            v.get("description", "")[:60],
            v.get("impact", ""),
            str(len(v.get("nodes", []))),
        )

    console.print(table)

    wcag_count = sum(
        1 for v in violations if any("wcag2a" in t or "wcag2aa" in t for t in v.get("tags", []))
    )
    num_violations = sum(len(v.get("nodes", [])) for v in violations)
    num_elements = len(result.get("passes", [])) + num_violations
    density = round((num_violations / num_elements) * 100, 2) if num_elements else 0

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total violations:   {num_violations}")
    console.print(f"  WCAG 2 A/AA failures: {wcag_count}")
    console.print(f"  Page elements:     {num_elements}")
    console.print(f"  Error density:     {density}%")


if __name__ == "__main__":
    app()