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
from browser_use.llm.openrouter.chat import ChatOpenRouter
from browser_use.tools.service import Tools
from dotenv import load_dotenv

from auth import generate_random_password, get_saved_password, login, make_password, save_password, signup
from axe import inject_axe, run_axe, save_report, slug_from_url, summarize_reports

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "temp.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "axe-core"

MAX_PAGES = 15

load_dotenv(BASE_DIR / ".env")


parser = argparse.ArgumentParser(description="Run axe-core accessibility audits on websites.")
parser.add_argument("--no-nav", action="store_true", help="Skip navigation link extraction; only audit the current page after login.")
parser.add_argument("--site", type=str, help="Run on a single site directly (bypasses CSV).")
parser.add_argument("--signup", action="store_true", help="Sign up for a new account instead of logging in. Password will be saved to .env.passwords.")


def read_domains(csv_path: Path) -> list[str]:
    domains: list[str] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 1:
                continue
            domain = row[0].strip()
            if domain == "domain" or not domain:
                continue
            domains.append(domain)
    return domains


def safe_dir_name(domain: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", domain).strip("._")


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


async def analyze_page(page, url: str, output_path: Path) -> dict | None:
    try:
        await page.goto(url)
    except Exception as e:
        print(f"  Failed to navigate to {url}: {e}")
        return None

    await asyncio.sleep(2)

    try:
        await inject_axe(page)
    except Exception as e:
        print(f"  Failed to inject axe-core on {url}: {e}")
        return None

    try:
        results = await run_axe(page)
    except Exception as e:
        print(f"  Failed to run axe on {url}: {e}")
        return None

    results["url"] = url
    save_report(results, output_path)
    violations = sum(len(v.get("nodes", [])) for v in results.get("violations", []))
    print(f"  {url} — {violations} violations → {output_path.name}")
    return results


async def main():
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = ChatOpenRouter(
        model="kimi-k2.6",
        base_url=os.getenv("OPENCODE_GO_URL"),
        api_key=os.getenv("AI_API_KEY"),
    )

    if args.site:
        domains = [args.site]
    else:
        domains = read_domains(INPUT_CSV)

    for domain in domains:
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

        domain_dir = OUTPUT_DIR / safe_dir_name(domain)
        domain_dir.mkdir(parents=True, exist_ok=True)

        if should_signup:
            task = await (signup_only_task if args.no_nav else signup_and_extract_task)(domain)
        else:
            task = await (login_only_task if args.no_nav else login_and_extract_task)(domain)

        profile = BrowserProfile(keep_alive=True)
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

        all_reports: list[dict] = []
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
                    report = await analyze_page(page, current_url, domain_dir / "page-0_home.json")
                    if report:
                        all_reports.append(report)
            else:
                urls = parse_urls_from_output(output, domain)
                if not urls:
                    print(f"[{domain}] No navigation links found, analyzing current page only")
                    page = await agent.browser_session.get_current_page()
                    if page:
                        current_url = await page.get_url()
                        report = await analyze_page(page, current_url, domain_dir / "page-0_home.json")
                        if report:
                            all_reports.append(report)
                else:
                    print(f"[{domain}] Found {len(urls)} pages to analyze")
                    page = await agent.browser_session.get_current_page()
                    if not page:
                        print(f"[{domain}] No page available after login")
                        continue

                    for i, url in enumerate(urls):
                        slug = slug_from_url(url)
                        out_path = domain_dir / f"page-{i}_{slug}.json"
                        report = await analyze_page(page, url, out_path)
                        if report:
                            all_reports.append(report)

        finally:
            try:
                await agent.close()
            except Exception:
                pass

        summary = summarize_reports(all_reports)
        save_report(summary, domain_dir / "summary.json")
        print(f"[{domain}] Done — {summary['total_pages']} pages, {summary['total_violations']} total violations")


if __name__ == "__main__":
    asyncio.run(main())
