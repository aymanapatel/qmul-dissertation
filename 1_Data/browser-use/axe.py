import asyncio
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


ACCEPT_COOKIES_SCRIPT = """() => {
    const directSelectors = [
        '#onetrust-accept-btn-handler',
        '#onetrust-pc-btn-handler + button',
        '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
        '#didomi-notice-agree-button',
        '#truste-consent-button',
        '#acceptCookie',
        '#accept-cookies',
        '#acceptCookies',
        '#cookie-accept',
        '#cookies-accept',
        '#cookieAccept',
        '#cookie_action_close_header',
        '[data-testid="uc-accept-all-button"]',
        '[data-testid="accept-all"]',
        '[data-testid="cookie-accept-all"]',
        '[data-test="accept-all"]',
        '[data-cy="accept-all"]',
        '[aria-label="Accept all"]',
        '[aria-label="Accept cookies"]'
    ];

    const positivePatterns = [
        /^(accept|agree|allow|consent|ok|got it|continue)$/i,
        /^(accept|agree|allow)\\s+(all|cookies|all cookies)$/i,
        /^i\\s+(accept|agree|consent)$/i,
        /^(yes|sure),?\\s*(i\\s*)?(accept|agree)$/i,
        /^allow\\s+all$/i
    ];
    const negativePattern = /\\b(reject|decline|deny|disagree|manage|settings|preferences|customi[sz]e|options|more|learn|necessary only|essential only)\\b/i;
    const consentContextPattern = /\\b(cookie|cookies|consent|privacy|gdpr|ccpa|tracking|personalized|personalised)\\b/i;

    const normalize = (value) => (value || '').replace(/\\s+/g, ' ').trim();

    const isVisible = (el) => {
        if (!el || typeof el.getBoundingClientRect !== 'function') return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) return false;
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && rect.bottom >= 0 && rect.right >= 0 && rect.top <= window.innerHeight && rect.left <= window.innerWidth;
    };

    const clickElement = (el, source, score) => {
        el.scrollIntoView({ block: 'center', inline: 'center' });
        el.click();
        return {
            clicked: true,
            source,
            score,
            text: normalize(el.innerText || el.textContent || el.value || el.getAttribute('aria-label') || '').slice(0, 120)
        };
    };

    for (const selector of directSelectors) {
        const el = document.querySelector(selector);
        if (isVisible(el)) {
            return clickElement(el, selector, 100);
        }
    }

    const roots = [];
    const visitRoot = (root) => {
        roots.push(root);
        const all = root.querySelectorAll ? root.querySelectorAll('*') : [];
        for (const el of all) {
            if (el.shadowRoot) visitRoot(el.shadowRoot);
        }
    };
    visitRoot(document);

    const candidateSelector = [
        'button',
        'a[href]',
        'input[type="button"]',
        'input[type="submit"]',
        '[role="button"]'
    ].join(',');

    const candidates = [];
    for (const root of roots) {
        for (const el of root.querySelectorAll(candidateSelector)) {
            if (!isVisible(el)) continue;

            const text = normalize([
                el.innerText,
                el.textContent,
                el.value,
                el.getAttribute('aria-label'),
                el.getAttribute('title')
            ].filter(Boolean).join(' '));
            const metadata = normalize([
                el.id,
                el.className && typeof el.className === 'string' ? el.className : '',
                el.getAttribute('name'),
                el.getAttribute('data-testid'),
                el.getAttribute('data-test'),
                el.getAttribute('data-cy')
            ].filter(Boolean).join(' '));
            const combined = normalize(`${text} ${metadata}`);

            if (!combined || negativePattern.test(combined)) continue;
            if (!positivePatterns.some((pattern) => pattern.test(text)) && !/\\baccept(-|_)?all\\b/i.test(metadata)) continue;

            let score = 10;
            if (positivePatterns.some((pattern) => pattern.test(text))) score += 40;
            if (/\\ball\\b/i.test(text)) score += 20;
            if (/\\bcookies?\\b/i.test(text)) score += 10;
            if (/\\baccept(-|_)?all\\b/i.test(metadata)) score += 25;

            const ancestorText = normalize(el.closest('[role="dialog"], dialog, aside, section, div')?.innerText || '');
            if (consentContextPattern.test(ancestorText)) score += 30;

            let current = el;
            while (current && current !== document.documentElement) {
                const style = window.getComputedStyle(current);
                if (style.position === 'fixed' || style.position === 'sticky') {
                    score += 15;
                    break;
                }
                current = current.parentElement;
            }

            candidates.push({ el, score });
        }
    }

    candidates.sort((a, b) => b.score - a.score);
    if (candidates[0] && candidates[0].score >= 50) {
        return clickElement(candidates[0].el, 'heuristic', candidates[0].score);
    }

    return { clicked: false };
}"""


async def accept_cookies(page, attempts: int = 3) -> list[dict]:
    accepted: list[dict] = []
    for _ in range(attempts):
        result = await page.evaluate(ACCEPT_COOKIES_SCRIPT)
        if not isinstance(result, dict) or not result.get("clicked"):
            break
        accepted.append(result)
        await asyncio.sleep(0.75)
    return accepted


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
