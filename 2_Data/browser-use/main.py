import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel

from browser_use import Agent
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.llm.openrouter.chat import ChatOpenRouter
from browser_use.tools.service import Tools
from dotenv import load_dotenv

from auth import generate_random_password, get_saved_password, login, make_password, save_password, signup
from axe import accept_cookies, inject_axe, run_axe, save_report, slug_from_url, summarize_reports
from rendered_snapshot import capture_rendered_snapshot

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "domains.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "axe-core"

MAX_PAGES = 15
PAGE_LOAD_TIMEOUT_MS = 10_000
NOT_RENDERED_STATUS = "yes_but_not_rendered"
CLOUDFLARE_CHALLENGE_TIMEOUT_SECONDS = 30.0

load_dotenv(BASE_DIR / ".env")


parser = argparse.ArgumentParser(description="Run axe-core accessibility audits on websites.")
parser.add_argument("--no-nav", action="store_true", help="Skip navigation link extraction; only audit the current page after login.")
parser.add_argument("--no-nav-landing-page", action="store_true", help="Skip authentication and navigation extraction; only audit the public landing page.")
parser.add_argument("--site", type=str, help="Run on a single site directly (bypasses CSV).")
parser.add_argument("--signup", action="store_true", help="Sign up for a new account instead of logging in. Password will be saved to .env.passwords.")
parser.add_argument("--workers", type=int, default=1, help="Number of domains to process in parallel.")
parser.add_argument("--headful", action="store_true", help="Run a real browser window offscreen/minimized instead of headless.")
parser.add_argument("--foreground", action="store_true", help="When used with --headful, keep browser windows visible for debugging.")
parser.add_argument("--skip-completed", action="store_true", help="Skip domains already marked complete in domains.csv.")
parser.add_argument("--skip-existing-output", action="store_true", help="Skip domains that already have an output summary.json.")
parser.add_argument("--no-csv-status", action="store_true", help="Do not update domains.csv while running.")
parser.add_argument("--settle-seconds", type=float, default=2.0, help="Seconds to wait after page load before running axe-core.")
parser.add_argument("--page-timeout-ms", type=int, default=PAGE_LOAD_TIMEOUT_MS, help="Page navigation timeout in milliseconds.")
parser.add_argument(
    "--start-row",
    "--source-row",
    dest="start_row",
    type=int,
    default=1,
    help="First 1-based data row from domains.csv to process. Header is not counted.",
)
parser.add_argument(
    "--end-row",
    "--destination-row",
    dest="end_row",
    type=int,
    help="Last 1-based data row from domains.csv to process, inclusive. Header is not counted.",
)


def make_browser_profile(args) -> BrowserProfile:
    if not args.headful:
        return BrowserProfile(keep_alive=True, headless=True)

    if args.foreground:
        return BrowserProfile(keep_alive=True, headless=False)

    return BrowserProfile(
        keep_alive=True,
        headless=False,
        window_size={"width": 1280, "height": 720},
        window_position={"width": 10000, "height": 10000},
        args=[
            "--start-minimized",
            "--disable-focus-on-load",
            "--disable-window-activation",
        ],
    )


def read_domains(
    csv_path: Path,
    start_row: int = 1,
    end_row: int | None = None,
    *,
    skip_completed: bool = False,
    completion_field: str = "scrapped_first",
) -> list[str]:
    domains: list[str] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        data_row = 0
        for row in reader:
            domain = row.get("domain", "").strip()
            if not domain:
                continue
            data_row += 1
            if data_row < start_row:
                continue
            if end_row is not None and data_row > end_row:
                break
            if skip_completed:
                status = row.get(completion_field, "").strip()
                completed_statuses = {"yes"}
                if completion_field == "scrapped_first":
                    completed_statuses.add(NOT_RENDERED_STATUS)
                if status in completed_statuses:
                    continue
            domains.append(domain)
    return domains


def update_domain_status(
    csv_path: Path,
    domain: str,
    *,
    first: bool = False,
    full: bool = False,
    first_status: str | None = None,
) -> None:
    if not csv_path.exists():
        return

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return

        fieldnames = list(reader.fieldnames)
        for field in ("scrapped_first", "scrapped_full", "notes"):
            if field not in fieldnames:
                fieldnames.append(field)

        rows = list(reader)

    changed = False
    for row in rows:
        if row.get("domain", "").strip() != domain:
            continue
        current_first_status = row.get("scrapped_first", "")
        if first_status is not None and current_first_status != first_status:
            row["scrapped_first"] = first_status
            changed = True
        elif first and current_first_status != "yes":
            row["scrapped_first"] = "yes"
            changed = True
        if full and row.get("scrapped_full", "") != "yes":
            row["scrapped_full"] = "yes"
            changed = True
        break

    if not changed:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_dir_name(domain: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", domain).strip("._")


def save_domain_summary(reports: list[dict], path: Path, first_status: str | None = None) -> dict:
    summary = summarize_reports(reports)
    summary["scrapped_first"] = first_status or ("yes" if reports else "")
    save_report(summary, path)
    return summary


async def save_rendered_html(page, path: Path) -> None:
    await capture_rendered_snapshot(page, path)


def parse_urls_from_output(output: str, base_domain: str) -> list[str]:
    urls: list[str] = []
    try:
        data = json.loads(output)
        if isinstance(data, list):
            urls = [str(u) for u in data]
        elif isinstance(data, dict) and "urls" in data:
            urls = [str(u) for u in data["urls"]]
    except (json.JSONDecodeError, TypeError):
        pass

    if not urls:
        found = re.findall(r'https?://[^\s"\'<>\]]+', output)
        urls = list(dict.fromkeys(found))

    base_domain_clean = base_domain.lstrip("www.")
    filtered: list[str] = []
    for url in urls:
        url = url.rstrip("/.,;!")
        if base_domain_clean in url:
            filtered.append(url)
    return list(dict.fromkeys(filtered))[:MAX_PAGES]


def get_content_type(headers: dict) -> str:
    for key, value in headers.items():
        if key.lower() == "content-type":
            return str(value)
    return ""


async def navigate_page(page, url: str, timeout_ms: int = PAGE_LOAD_TIMEOUT_MS) -> tuple[bool, str]:
    if hasattr(page, "_ensure_session") and hasattr(page, "_client"):
        session_id = await page._ensure_session()
        client = page._client
        loop = asyncio.get_running_loop()
        response_future = loop.create_future()

        def on_response(event, event_session_id):
            if event_session_id != session_id or response_future.done():
                return
            if event.get("type") != "Document":
                return
            response = event.get("response", {})
            response_future.set_result(get_content_type(response.get("headers", {})))

        client.register.Network.responseReceived(on_response)
        try:
            await client.send.Network.enable(session_id=session_id)
            nav_result = await asyncio.wait_for(
                client.send.Page.navigate({"url": url}, session_id=session_id),
                timeout=timeout_ms / 1000,
            )
            if nav_result.get("errorText"):
                return False, ""
            try:
                return True, await asyncio.wait_for(response_future, timeout=timeout_ms / 1000)
            except TimeoutError:
                return False, ""
        finally:
            client._event_registry.unregister("Network.responseReceived")

    try:
        response = await page.goto(url, timeout=timeout_ms)
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise
        response = await asyncio.wait_for(page.goto(url), timeout=timeout_ms / 1000)

    if not response:
        return False, ""
    headers = response.headers
    if callable(headers):
        headers = headers()
    return True, get_content_type(headers)


CLOUDFLARE_TURNSTILE_CHECK_SCRIPT = """() => {
    const html = document.documentElement?.outerHTML || '';
    const selectors = [
        'iframe[src*="challenges.cloudflare.com"]',
        'iframe[src*="/turnstile/"]',
        '.cf-turnstile',
        '[data-sitekey][data-callback]',
        'input[name="cf-turnstile-response"]',
        '#challenge-stage',
        '#cf-challenge-running',
        '#cf-challenge-hcaptcha-wrapper',
        '#turnstile-wrapper'
    ];
    const markers = selectors.filter((selector) => document.querySelector(selector));
    const text = (document.body?.innerText || '').replace(/\\s+/g, ' ').trim();
    const title = document.title || '';
    const combined = `${title} ${text} ${html}`;
    const challengeHtml = /challenges\\.cloudflare\\.com|cf-turnstile|cf-challenge|turnstile-wrapper|cf-browser-verification/i.test(html);
    const challengeText = /checking if the site connection is secure|verify you are human|verifying you are human|just a moment|ray id/i.test(combined);
    const active = markers.length > 0 || challengeHtml || challengeText;
    return JSON.stringify({ active, markers: markers.slice(0, 5), title });
}"""


async def detect_cloudflare_turnstile(page) -> tuple[bool, list[str], str]:
    raw = await page.evaluate(CLOUDFLARE_TURNSTILE_CHECK_SCRIPT)
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        return False, [], ""
    return bool(data.get("active")), list(data.get("markers") or []), str(data.get("title") or "")


async def wait_for_cloudflare_turnstile(page, url: str) -> None:
    started_waiting = False
    deadline = asyncio.get_running_loop().time() + CLOUDFLARE_CHALLENGE_TIMEOUT_SECONDS
    while True:
        try:
            active, markers, title = await detect_cloudflare_turnstile(page)
        except Exception as e:
            print(f"  Cloudflare/Turnstile check skipped on {url}: {e}")
            return

        if not active:
            if started_waiting:
                print(f"  Cloudflare/Turnstile challenge cleared on {url}")
            return

        if not started_waiting:
            marker_text = ", ".join(markers) if markers else title or "challenge text"
            print(f"  Cloudflare/Turnstile challenge HTML detected on {url}; waiting for it to clear ({marker_text})")
            started_waiting = True

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            print(f"  Cloudflare/Turnstile wait timed out on {url}; continuing to axe-core")
            return

        await asyncio.sleep(min(2.0, remaining))


_SIGNUP_ERROR_KEYWORDS = [
    "password must be", "password is too", "weak password", "invalid password",
    "password error", "password requirements", "password criteria",
    "email already", "already registered", "already exists", "already taken",
    "account already", "user already", "signup failed", "sign up failed",
    "registration failed", "could not sign up", "unable to create",
    "unfortunately", "sorry", "unable to", "failed to"
]


def signup_looks_broken(output: str, result) -> bool:
    if result.errors:
        return True
    output_lower = output.lower()
    return any(kw in output_lower for kw in _SIGNUP_ERROR_KEYWORDS)


@login
async def login_and_extract_task(domain: str, email: str, password: str) -> str:
    return (
        f"Go to https://{domain} and log in to an existing account. "
        f"Use the email {email} and the password {password}. "
        f"After logging in, extract all unique navigation links visible on the page "
        f"(header nav, sidebar, footer links, main menu items). "
        f"Return ONLY a JSON array of full URLs, no other text. "
        f'Example: ["https://{domain}/dashboard", "https://{domain}/settings"]'
    )


@login
async def login_only_task(domain: str, email: str, password: str) -> str:
    return (
        f"Go to https://{domain} and log in to an existing account. "
        f"Use the email {email} and the password {password}. "
        f"After logging in, stop. Do nothing else."
    )


@signup
async def signup_and_extract_task(domain: str, email: str, password: str) -> str:
    return (
        f"Go to https://{domain} and create a new account. "
        f"Use the email {email} and the password {password}. "
        f"Fill out the signup form completely. "
        f"IMPORTANT: If the form rejects the password (e.g., too weak, too short, missing special characters), "
        f"call the tool generate_new_password with domain='{domain}' to get a new secure password, then retry the signup. "
        f"After signing up and landing on the main page, extract all unique navigation links visible on the page "
        f"(header nav, sidebar, footer links, main menu items). "
        f"Return ONLY a JSON array of full URLs, no other text. "
        f'Example: ["https://{domain}/dashboard", "https://{domain}/settings"]'
    )


@signup
async def signup_only_task(domain: str, email: str, password: str) -> str:
    return (
        f"Go to https://{domain} and create a new account. "
        f"Use the email {email} and the password {password}. "
        f"Fill out the signup form completely. "
        f"IMPORTANT: If the form rejects the password (e.g., too weak, too short, missing special characters), "
        f"call the tool generate_new_password with domain='{domain}' to get a new secure password, then retry the signup. "
        f"After signing up, stop. Do nothing else."
    )


async def pre_page_waiting(page, url: str) -> None:
    await wait_for_cloudflare_turnstile(page, url)

    try:
        accepted = await accept_cookies(page)
        if accepted:
            sources = ", ".join(item.get("source", "heuristic") for item in accepted)
            print(f"  Accepted cookie prompt on {url} via {sources}")
    except Exception as e:
        print(f"  Cookie prompt handling skipped on {url}: {e}")


async def analyze_page(
    page,
    url: str,
    output_path: Path,
    *,
    html_path: Path,
    settle_seconds: float,
    timeout_ms: int,
) -> tuple[dict | None, bool]:
    try:
        has_response, content_type = await navigate_page(page, url, timeout_ms=timeout_ms)
    except Exception as e:
        print(f"  Failed to navigate to {url}: {e}")
        return None, False

    if not has_response:
        print(f"  No response for {url}")
        return None, True

    if "text/html" not in content_type.lower():
        print(f"  Not HTML for {url}: {content_type}")
        return None, True

    print(f"  HTML page: {url}")
    if settle_seconds > 0:
        await asyncio.sleep(settle_seconds)
    await pre_page_waiting(page, url)

    try:
        await save_rendered_html(page, html_path)
        print(f"  Saved rendered HTML → {html_path.name}")
    except Exception as e:
        print(f"  Failed to save rendered HTML for {url}: {e}")

    try:
        await inject_axe(page)
    except Exception as e:
        print(f"  Failed to inject axe-core on {url}: {e}")
        return None, False

    try:
        results = await run_axe(page)
    except Exception as e:
        print(f"  Failed to run axe on {url}: {e}")
        return None, False

    results["url"] = url
    save_report(results, output_path)
    violations = sum(len(v.get("nodes", [])) for v in results.get("violations", []))
    print(f"  {url} — {violations} violations → {output_path.name}")
    return results, False


async def update_domain_status_locked(csv_lock: asyncio.Lock, domain: str, **kwargs) -> None:
    async with csv_lock:
        update_domain_status(INPUT_CSV, domain, **kwargs)


async def process_domain(domain: str, args, llm, update_csv_status: bool, csv_lock: asyncio.Lock) -> None:
    domain_dir = OUTPUT_DIR / safe_dir_name(domain)
    domain_dir.mkdir(parents=True, exist_ok=True)
    all_reports: list[dict] = []
    first_status: str | None = None

    if args.no_nav_landing_page:
        print(f"[{domain}] --no-nav-landing-page set, analyzing landing page only")
        profile = make_browser_profile(args)
        browser_session = BrowserSession(browser_profile=profile)

        try:
            await browser_session.start()
            page = await browser_session.new_page(f"https://{domain}")
            report, not_rendered = await analyze_page(
                page,
                f"https://{domain}",
                domain_dir / "page-0_home.json",
                html_path=domain_dir / "0.html",
                settle_seconds=args.settle_seconds,
                timeout_ms=args.page_timeout_ms,
            )
            if not_rendered and update_csv_status:
                await update_domain_status_locked(csv_lock, domain, first_status=NOT_RENDERED_STATUS)
            if not_rendered:
                first_status = NOT_RENDERED_STATUS
            if report:
                all_reports.append(report)
                first_status = "yes"
                if update_csv_status:
                    await update_domain_status_locked(csv_lock, domain, first=True)
        finally:
            try:
                await browser_session.close()
            except Exception:
                pass

        summary = save_domain_summary(all_reports, domain_dir / "summary.json", first_status)
        print(f"[{domain}] Done — {summary['total_pages']} pages, {summary['total_violations']} total violations")
        return

    password = get_saved_password(domain)
    if args.signup or not password:
        if not password:
            print(f"[{domain}] No saved password found, signing up")
        else:
            print(f"[{domain}] --signup flag set, signing up")
        password = make_password(domain)
        should_signup = True
    else:
        print(f"[{domain}] Logging in{' and extracting navigation links' if not args.no_nav else ''}")
        should_signup = False

    if should_signup:
        task = await (signup_only_task if args.no_nav else signup_and_extract_task)(domain)
    else:
        task = await (login_only_task if args.no_nav else login_and_extract_task)(domain)

    profile = make_browser_profile(args)
    custom_tools = Tools()

    class GeneratePasswordParams(BaseModel):
        domain: str

    @custom_tools.registry.action(
        description="Generate a new cryptographically secure random password for the given domain and save it to .env.passwords. Use this ONLY when the signup form rejects the current password (e.g., too weak, too short, missing special characters, does not meet complexity requirements).",
        param_model=GeneratePasswordParams,
    )
    def generate_new_password(params: GeneratePasswordParams) -> str:
        password = generate_random_password()
        save_password(params.domain, password)
        return f"Generated and saved new password for {params.domain}: {password}"

    agent = Agent(task=task, llm=llm, browser_profile=profile, tools=custom_tools)

    try:
        result = await agent.run()
        output = result.final_result or ""

        if should_signup:
            # The LLM may have called generate_new_password via tool call.
            # Check what password ended up in .env.passwords and save that.
            final_password = get_saved_password(domain)
            if final_password and final_password != password:
                print(f"[{domain}] LLM generated a new password via tool call, saved to .env.passwords")
            elif not signup_looks_broken(output, result):
                save_password(domain, password)
                print(f"[{domain}] Saved generated password to .env.passwords")
            else:
                print(f"[{domain}] Signup may have failed, not saving password")

        if args.no_nav:
            print(f"[{domain}] --no-nav set, analyzing current page only")
            page = await agent.browser_session.get_current_page()
            if page:
                current_url = await page.get_url()
                report, not_rendered = await analyze_page(
                    page,
                    current_url,
                    domain_dir / "page-0_home.json",
                    html_path=domain_dir / "0.html",
                    settle_seconds=args.settle_seconds,
                    timeout_ms=args.page_timeout_ms,
                )
                if not_rendered and update_csv_status:
                    await update_domain_status_locked(csv_lock, domain, first_status=NOT_RENDERED_STATUS)
                if not_rendered:
                    first_status = NOT_RENDERED_STATUS
                if report:
                    all_reports.append(report)
                    first_status = "yes"
                    if update_csv_status:
                        await update_domain_status_locked(csv_lock, domain, first=True)
        else:
            urls = parse_urls_from_output(output, domain)
            if not urls:
                print(f"[{domain}] No navigation links found, analyzing current page only")
                page = await agent.browser_session.get_current_page()
                if page:
                    current_url = await page.get_url()
                    report, not_rendered = await analyze_page(
                        page,
                        current_url,
                        domain_dir / "page-0_home.json",
                        html_path=domain_dir / "0.html",
                        settle_seconds=args.settle_seconds,
                        timeout_ms=args.page_timeout_ms,
                    )
                    if not_rendered and update_csv_status:
                        await update_domain_status_locked(csv_lock, domain, first_status=NOT_RENDERED_STATUS)
                    if not_rendered:
                        first_status = NOT_RENDERED_STATUS
                    if report:
                        all_reports.append(report)
                        first_status = "yes"
                        if update_csv_status:
                            await update_domain_status_locked(csv_lock, domain, first=True)
            else:
                print(f"[{domain}] Found {len(urls)} pages to analyze")
                page = await agent.browser_session.get_current_page()
                if not page:
                    print(f"[{domain}] No page available after login")
                    return

                for i, url in enumerate(urls):
                    slug = slug_from_url(url)
                    out_path = domain_dir / f"page-{i}_{slug}.json"
                    report, not_rendered = await analyze_page(
                        page,
                        url,
                        out_path,
                        html_path=domain_dir / f"{i}.html",
                        settle_seconds=args.settle_seconds,
                        timeout_ms=args.page_timeout_ms,
                    )
                    if i == 0 and not_rendered and update_csv_status:
                        await update_domain_status_locked(csv_lock, domain, first_status=NOT_RENDERED_STATUS)
                    if i == 0 and not_rendered:
                        first_status = NOT_RENDERED_STATUS
                    if report:
                        all_reports.append(report)
                        if i == 0:
                            first_status = "yes"
                            if update_csv_status:
                                await update_domain_status_locked(csv_lock, domain, first=True)

                if len(all_reports) == len(urls) and update_csv_status:
                    await update_domain_status_locked(csv_lock, domain, full=True)

    finally:
        try:
            await agent.close()
        except Exception:
            pass

    summary = save_domain_summary(all_reports, domain_dir / "summary.json", first_status)
    print(f"[{domain}] Done — {summary['total_pages']} pages, {summary['total_violations']} total violations")


async def main():
    args = parser.parse_args()
    if args.no_nav_landing_page and (args.no_nav or args.signup):
        parser.error("--no-nav-landing-page cannot be combined with --no-nav or --signup")
    if args.start_row < 1:
        parser.error("--start-row/--source-row must be 1 or greater")
    if args.end_row is not None and args.end_row < args.start_row:
        parser.error("--end-row/--destination-row must be greater than or equal to --start-row/--source-row")
    if args.site and (args.start_row != 1 or args.end_row is not None):
        parser.error("row range flags cannot be combined with --site")
    if args.workers < 1:
        parser.error("--workers must be 1 or greater")
    if args.foreground and not args.headful:
        parser.error("--foreground can only be used with --headful")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be 0 or greater")
    if args.page_timeout_ms < 1000:
        parser.error("--page-timeout-ms must be at least 1000")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = None
    if not args.no_nav_landing_page:
        llm = ChatOpenRouter(
            model="kimi-k2.6",
            base_url=os.getenv("OPENCODE_GO_URL"),
            api_key=os.getenv("AI_API_KEY"),
        )

    if args.site:
        domains = [args.site]
        update_csv_status = False
    else:
        completion_field = "scrapped_full"
        if args.no_nav_landing_page or args.no_nav:
            completion_field = "scrapped_first"
        domains = read_domains(
            INPUT_CSV,
            start_row=args.start_row,
            end_row=args.end_row,
            skip_completed=args.skip_completed,
            completion_field=completion_field,
        )
        update_csv_status = not args.no_csv_status
        selected_range = f"rows {args.start_row} to {args.end_row}" if args.end_row is not None else f"rows {args.start_row} to end"
        print(f"[domains.csv] Loaded {len(domains)} domains from {selected_range}")
        if args.skip_completed:
            print(f"[domains.csv] Skipped rows already marked {completion_field}=yes")

    semaphore = asyncio.Semaphore(args.workers)
    csv_lock = asyncio.Lock()

    async def run_domain(domain: str) -> None:
        async with semaphore:
            try:
                if args.skip_existing_output and (OUTPUT_DIR / safe_dir_name(domain) / "summary.json").exists():
                    print(f"[{domain}] Skipping existing output")
                    return
                await process_domain(domain, args, llm, update_csv_status, csv_lock)
            except Exception as e:
                print(f"[{domain}] Failed: {e}")

    if args.workers == 1:
        for domain in domains:
            await run_domain(domain)
    else:
        print(f"[parallel] Processing {len(domains)} domains with {args.workers} workers")
        await asyncio.gather(*(run_domain(domain) for domain in domains))


if __name__ == "__main__":
    asyncio.run(main())
