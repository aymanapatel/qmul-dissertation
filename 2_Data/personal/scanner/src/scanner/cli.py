import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Optional

import typer
from playwright.async_api import async_playwright
from axe_playwright_python.async_playwright import Axe as AsyncAxe
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from scanner.robots_policy import RobotsPolicy

app = typer.Typer(help="Web accessibility scanner using axe-core + Playwright")
console = Console()

LOGS_DIR = Path("logs")
PLAYWRIGHT_LOGS = LOGS_DIR / "playwright"
AXE_CORE_LOGS = LOGS_DIR / "axe-core-result"
ROBOTS = RobotsPolicy()

CSV_COLUMNS = [
    "popularity_rank",
    "domain",
    "Populated",
    "Number_of_accessibility_errors_detected",
    "WCAG_2_A/AA_failure_detected",
    "Number_of_page_elements",
    "Error_density_Percentage",
    "Top_error_types_detected",
]


def _safe_filename(domain: str, rank: str = "") -> str:
    if rank:
        padded = rank.zfill(7)
    else:
        padded = ""
    clean = re.sub(r"[^\w\-.]", "_", domain)
    return f"{padded}_{clean}" if padded else clean


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
        "Number_of_accessibility_errors_detected": num_violations,
        "WCAG_2_A/AA_failure_detected": len(wcag_aa_failures),
        "Number_of_page_elements": num_elements,
        "Error_density_Percentage": error_density,
        "Top_error_types_detected": top_error_names,
    }


async def _scan_single(domain: str, browser, semaphore, rank: str = "") -> Optional[dict]:
    url = f"https://{domain}"
    safe_name = _safe_filename(domain, rank)

    async with semaphore:
        if not await ROBOTS.can_fetch(url):
            console.print(f"[yellow]Skipping {url}: disallowed by robots.txt or robots.txt unavailable[/yellow]")
            return None

        page = await browser.new_page()
        try:
            await page.goto(url, timeout=45000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            axe = AsyncAxe()
            axe_result = await axe.run(page)
            result = axe_result.response

            PLAYWRIGHT_LOGS.mkdir(parents=True, exist_ok=True)
            AXE_CORE_LOGS.mkdir(parents=True, exist_ok=True)

            try:
                html = await page.content()
                (PLAYWRIGHT_LOGS / f"{safe_name}.html").write_text(html, encoding="utf-8")
            except Exception as e:
                console.print(f"[yellow]Could not save HTML for {domain}: {e}[/yellow]")

            try:
                (AXE_CORE_LOGS / f"{safe_name}.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
            except Exception as e:
                console.print(f"[yellow]Could not save axe result for {domain}: {e}[/yellow]")

            return result

        except Exception as e:
            console.print(f"[red]Error scanning {domain}: {e}[/red]")
            return None
        finally:
            await page.close()


async def _run_scan(rows: list[dict], workers: int):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        semaphore = asyncio.Semaphore(workers)

        tasks = []
        for row in rows:
            domain = row.get("domain", "").strip()
            if not domain:
                continue
            rank = row.get("popularity_rank", "")
            tasks.append((row, _scan_single(domain, browser, semaphore, rank=rank)))

        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        await browser.close()
        return [(tasks[i][0], r if not isinstance(r, Exception) else None) for i, r in enumerate(results)]


@app.command()
def scan(
    csv_path: Path = typer.Argument(..., help="Path to the input CSV file"),
    output: Path = typer.Option(None, "-o", "--output", help="Output CSV path (default: <input>_results.csv)"),
    limit: int = typer.Option(None, "-n", "--limit", help="Limit number of domains to scan"),
    start: int = typer.Option(None, "-s", "--start", help="Start from this row number (1-based)"),
    skip_populated: bool = typer.Option(False, "--skip-populated", help="Skip rows already populated"),
    workers: int = typer.Option(5, "-w", "--workers", help="Number of parallel browser tabs"),
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

    if skip_populated:
        rows = [r for r in rows if r.get("Populated", "").strip().lower() != "yes"]

    rows = [r for r in rows if r.get("domain", "").strip()]

    console.print(f"[bold]Scanning {len(rows)} domains with {workers} workers[/bold]")
    console.print(f"[dim]Playwright HTML logs: {PLAYWRIGHT_LOGS.resolve()}[/dim]")
    console.print(f"[dim]Axe-core result logs:  {AXE_CORE_LOGS.resolve()}[/dim]")

    file = open(output, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    file.flush()

    console.print(f"[dim]Writing results incrementally to {output}[/dim]")

    async def _scan_batch(batch: list[dict]):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            semaphore = asyncio.Semaphore(workers)

            tasks = []
            for row in batch:
                domain = row.get("domain", "").strip()
                rank = row.get("popularity_rank", "")
                tasks.append(_scan_single(domain, browser, semaphore, rank=rank))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            await browser.close()

        for i, result in enumerate(results):
            row = batch[i]
            domain = row.get("domain", "").strip()
            r = result if not isinstance(result, Exception) else None
            extracted = _extract_row(r if isinstance(r, dict) else None, domain)
            extracted["popularity_rank"] = row.get("popularity_rank", "")
            extracted["Populated"] = "Yes" if r and not isinstance(r, Exception) else "No"
            writer.writerow(extracted)

        file.flush()

    batch_size = workers * 2
    total = len(rows)
    completed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning...", total=total)

        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            asyncio.run(_scan_batch(batch))
            completed += len(batch)
            progress.update(task, completed=completed)

    file.close()
    console.print(f"\n[green]Results saved to {output}[/green]")

    _print_summary_from_csv(output)


def _print_summary_from_csv(output: Path):
    results = []
    with open(output, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    table = Table(title="Scan Summary")
    table.add_column("Domain", style="cyan", max_width=30)
    table.add_column("Violations", justify="right")
    table.add_column("WCAG A/AA", justify="right")
    table.add_column("Elements", justify="right")
    table.add_column("Error %", justify="right")
    table.add_column("Top Errors", max_width=40)

    for r in rows[:20]:
        table.add_row(
            r.get("domain", ""),
            str(r.get("Number_of_accessibility_errors_detected", "")),
            str(r.get("WCAG_2_A/AA_failure_detected", "")),
            str(r.get("Number_of_page_elements", "")),
            str(r.get("Error_density_Percentage", "")),
            r.get("Top_error_types_detected", ""),
        )

    console.print(table)
    if len(rows) > 20:
        console.print(f"... and {len(rows) - 20} more. See output CSV for full results.")


@app.command()
def single(
    url: str = typer.Argument(..., help="URL to scan (e.g. https://example.com)"),
    json_report: bool = typer.Option(False, "--json", help="Output raw axe results as JSON"),
):
    """Scan a single URL for accessibility violations."""
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    console.print(f"[bold]Scanning {url}[/bold]")

    async def _run():
        if not await ROBOTS.can_fetch(url):
            console.print(f"[yellow]Skipping {url}: disallowed by robots.txt or robots.txt unavailable[/yellow]")
            return None

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=45000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                axe = AsyncAxe()
                axe_result = await axe.run(page)
                result = axe_result.response
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                return None
            finally:
                await browser.close()
            return result

    result = asyncio.run(_run())
    if result is None:
        raise typer.Exit(1)

    PLAYWRIGHT_LOGS.mkdir(parents=True, exist_ok=True)
    AXE_CORE_LOGS.mkdir(parents=True, exist_ok=True)

    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
    safe_name = _safe_filename(domain)

    async def _save_html():
        if not await ROBOTS.can_fetch(url):
            return

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=45000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                html = await page.content()
                (PLAYWRIGHT_LOGS / f"{safe_name}.html").write_text(html, encoding="utf-8")
                console.print(f"[green]HTML saved to {PLAYWRIGHT_LOGS / f'{safe_name}.html'}[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not save HTML: {e}[/yellow]")
            finally:
                await browser.close()

    asyncio.run(_save_html())

    axe_path = AXE_CORE_LOGS / f"{safe_name}.json"
    axe_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green]Axe result saved to {axe_path}[/green]")

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
