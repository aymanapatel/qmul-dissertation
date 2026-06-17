import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AXE_MIN_JS_PATH = BASE_DIR / "axe-core.min.js"

# Lazy-loaded minified axe-core source
_axe_source: str | None = None


def _get_axe_source() -> str:
    global _axe_source
    if _axe_source is None:
        if not AXE_MIN_JS_PATH.exists():
            raise FileNotFoundError(
                f"axe-core.min.js not found at {AXE_MIN_JS_PATH}. "
                "Download it with: curl -L https://unpkg.com/axe-core@4.10.0/axe.min.js -o axe-core.min.js"
            )
        _axe_source = AXE_MIN_JS_PATH.read_text(encoding="utf-8")
    return _axe_source


def _build_inject_script(axe_source: str) -> str:
    # Escape backticks and backslashes for the JS template literal
    escaped = axe_source.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    return f"""() => {{
    return new Promise((resolve, reject) => {{
        if (typeof window.axe !== 'undefined') {{
            resolve('already loaded');
            return;
        }}
        try {{
            eval(`{escaped}`);
            resolve('loaded');
        }} catch (err) {{
            reject(new Error('Failed to inject axe-core: ' + err.message));
        }}
    }});
}}"""


RUN_SCRIPT = """() => {
    return axe.run(document, {
        runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] },
        resultTypes: ['violations', 'passes', 'incomplete', 'inapplicable']
    }).then(results => JSON.stringify(results));
}"""


async def inject_axe(page) -> None:
    source = _get_axe_source()
    script = _build_inject_script(source)
    result = await page.evaluate(script)
    if result not in ("loaded", "already loaded"):
        raise RuntimeError(f"axe-core injection failed: {result}")


async def run_axe(page) -> dict:
    raw = await page.evaluate(RUN_SCRIPT)
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def save_report(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def slug_from_url(url: str) -> str:
    from urllib.parse import urlparse

    path = urlparse(url).path.strip("/")
    if not path:
        return "home"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", path).strip("-")
    return slug[:60] or "page"


def summarize_reports(reports: list[dict]) -> dict:
    summary = {"total_pages": len(reports), "total_violations": 0, "by_rule": {}, "by_page": []}
    for i, report in enumerate(reports):
        violations = report.get("violations", [])
        count = sum(len(v.get("nodes", [])) for v in violations)
        summary["total_violations"] += count
        summary["by_page"].append({"page_index": i, "url": report.get("url", ""), "violations": count})
        for v in violations:
            rule_id = v.get("id", "unknown")
            summary["by_rule"][rule_id] = summary["by_rule"].get(rule_id, 0) + len(v.get("nodes", []))
    return summary
