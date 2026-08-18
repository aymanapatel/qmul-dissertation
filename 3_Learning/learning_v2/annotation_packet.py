"""Create a blinded, evidence-complete detection annotation packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import random
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .rules import rule_metadata


# Chromium cannot reliably encode arbitrarily tall pages as one bitmap.  Keep
# individual tiles comfortably below its practical image-size limit.
MAX_SINGLE_SCREENSHOT_HEIGHT = 30_000
SCREENSHOT_TILE_HEIGHT = 8_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_id(site_id: str, criterion_id: str, seed: int) -> str:
    return "case-" + hashlib.sha256(f"{seed}|{site_id}|{criterion_id}".encode()).hexdigest()[:20]


def _criterion_records(registry_path: Path, rule_ids: list[str]) -> dict[str, dict[str, Any]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["criteria"]
    criterion_ids = sorted({criterion for rule in rule_ids for criterion in rule_metadata(rule)["wcag_ids"]})
    missing = set(criterion_ids) - set(registry)
    if missing:
        raise ValueError(f"Registry is missing criteria: {sorted(missing)}")
    return {criterion_id: registry[criterion_id] for criterion_id in criterion_ids}


def _rater_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "criterion_id": case["criterion_id"],
        "criterion_name": case["criterion_name"],
        "level": case["level"],
        "required_evidence": case["required_evidence"],
        "evidence_paths": case["evidence_paths"],
        "status": None,
        "applicable_exception": None,
        "evidence_notes": "",
        "confidence": None,
    }


def _capture_url(axe_report: Path, expected_site: str) -> str:
    report = json.loads(axe_report.read_text(encoding="utf-8"))
    value = str(report.get("url", ""))
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"Saved axe report has an invalid public page URL: {value!r}")
    if parsed.hostname.lower().rstrip(".") != expected_site.lower().rstrip("."):
        raise ValueError(f"Saved page host {parsed.hostname!r} does not match frozen site {expected_site!r}")
    return value


def _with_base(source_html: str, capture_url: str) -> str:
    """Add a capture URL base without changing the frozen page body."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(source_html, "lxml")
    if soup.html is None:
        root = soup.new_tag("html")
        root.extend(list(soup.contents))
        soup.append(root)
    if soup.head is None:
        soup.html.insert(0, soup.new_tag("head"))
    for old in soup.head.find_all("base"):
        old.decompose()
    soup.head.insert(0, soup.new_tag("base", href=capture_url))
    return str(soup)


def _canonical_markup(value: str) -> str:
    """Canonicalize HTML for strict shell/target integrity comparison."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(value, "lxml")
    for node in list(soup.find_all(string=lambda text: text is not None and not text.strip())):
        node.extract()
    return str(soup)


def _is_app_shell(source_html: str) -> bool:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(source_html, "lxml")
    body = soup.body
    if body is None:
        return False
    elements = body.find_all(True)
    return len(elements) <= 2 and not body.get_text(" ", strip=True) and any(
        element.get("id") in {"app", "root", "__next"} for element in elements
    )


def _saved_target_fragments(axe_report: Path) -> list[tuple[str, str]]:
    report = json.loads(axe_report.read_text(encoding="utf-8"))
    result = []
    for violation in report.get("violations", []):
        for node in violation.get("nodes", []):
            targets = node.get("target", [])
            selector = targets[0] if targets and isinstance(targets[0], str) else None
            fragment = str(node.get("html", ""))
            if selector and fragment:
                result.append((selector, fragment))
    return result


VISUAL_EVIDENCE_SCRIPT = r"""() => {
  function rgba(value) {
    const match = String(value || '').match(/rgba?\(([^)]+)\)/);
    if (!match) return [0, 0, 0, 0];
    const parts = match[1].split(',').map(v => Number.parseFloat(v.trim()));
    return [parts[0] || 0, parts[1] || 0, parts[2] || 0, parts.length > 3 && Number.isFinite(parts[3]) ? parts[3] : 1];
  }
  function composite(fg, bg) {
    const a = fg[3] + bg[3] * (1 - fg[3]);
    if (a <= 0) return [0, 0, 0, 0];
    return [(fg[0]*fg[3]+bg[0]*bg[3]*(1-fg[3]))/a,(fg[1]*fg[3]+bg[1]*bg[3]*(1-fg[3]))/a,(fg[2]*fg[3]+bg[2]*bg[3]*(1-fg[3]))/a,a];
  }
  function background(element) {
    const chain = []; let current = element;
    while (current && current.nodeType === 1) { chain.push(rgba(getComputedStyle(current).backgroundColor)); current = current.parentElement; }
    let result = [255,255,255,1];
    for (let i=chain.length-1;i>=0;i--) result = composite(chain[i], result);
    return result;
  }
  function luminance(rgb) {
    const c = rgb.slice(0,3).map(v => { const x=v/255; return x<=0.04045 ? x/12.92 : Math.pow((x+0.055)/1.055,2.4); });
    return .2126*c[0]+.7152*c[1]+.0722*c[2];
  }
  function ratio(a,b) { const x=luminance(a), y=luminance(b); return (Math.max(x,y)+.05)/(Math.min(x,y)+.05); }
  function selector(element) {
    if (element.id) return `${element.tagName.toLowerCase()}#${CSS.escape(element.id)}`;
    const parts=[]; let current=element;
    while (current && current.nodeType===1 && parts.length<8) {
      const tag=current.tagName.toLowerCase(); let n=1, sibling=current.previousElementSibling;
      while (sibling) { if (sibling.tagName===current.tagName) n++; sibling=sibling.previousElementSibling; }
      parts.unshift(`${tag}:nth-of-type(${n})`); current=current.parentElement;
    }
    return parts.join(' > ');
  }
  const all = [...document.querySelectorAll('*')], rows = [];
  for (const element of all) {
    const directText=[...element.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent.trim()).filter(Boolean).join(' ');
    if (!directText) continue;
    const style=getComputedStyle(element), rect=element.getBoundingClientRect();
    const visible=rect.width>0 && rect.height>0 && style.display!=='none' && style.visibility!=='hidden' && Number.parseFloat(style.opacity)>0;
    if (!visible) continue;
    const bg=background(element), raw=rgba(style.color), fg=raw[3]<1 ? composite(raw,bg) : raw;
    const fontSize=Number.parseFloat(style.fontSize), fontWeight=Number.parseFloat(style.fontWeight);
    const large=fontSize>=24 || (fontSize>=18.6667 && fontWeight>=700), required=large?3:4.5, measured=ratio(fg,bg);
    const complex=style.backgroundImage && style.backgroundImage!=='none';
    rows.push({selector:selector(element),tag:element.tagName.toLowerCase(),text:directText.slice(0,500),
      foreground_rgb:fg.slice(0,3).map(Math.round),background_rgb:bg.slice(0,3).map(Math.round),
      contrast_ratio:measured,required_contrast_ratio:required,passes_numeric_threshold:measured>=required,
      font_size_px:fontSize,font_weight:fontWeight,large_text:large,background_image:style.backgroundImage,
      requires_manual_background_review:Boolean(complex),bounds:{x:rect.x,y:rect.y,width:rect.width,height:rect.height}});
  }
  return {schema_version:1,viewport:{width:innerWidth,height:innerHeight},element_count:all.length,
    visible_text_sample_count:rows.length,samples:rows};
}"""


def _viewer_html(*, site_id: str, capture_url: str, captured_at: str, screenshot_mode: str) -> str:
    tiled_note = (
        '<p>This unusually long page is shown in ordered <a href="rendered_tiles/manifest.json">full-page screenshot tiles</a>; '
        '<code>rendered.png</code> is the first tile.</p>'
        if screenshot_mode == "tiled_full_page" else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Styled evidence — {html.escape(site_id)}</title><style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0;color:#17202a;background:#e9edf2}}header{{position:sticky;top:0;background:#fff;padding:1rem;border-bottom:1px solid #aab4c0;z-index:1}}
header p{{margin:.25rem 0}}a{{margin-right:1rem}}main{{max-width:1280px;margin:auto;background:white}}img{{display:block;width:100%;height:auto}}
</style></head><body><header><strong>Styled frozen-page evidence: {html.escape(site_id)}</strong>
<p>DOM from the frozen scan; CSS, fonts and images were hydrated from <code>{html.escape(capture_url)}</code> at {html.escape(captured_at)}.</p>
<p><a href="rendered.mhtml">Open interactive styled archive in Chrome</a><a href="rendered_dom.html">Inspect frozen rendered DOM</a><a href="computed_visual.json">Inspect computed contrast evidence</a><a href="live_ax.json">Inspect accessibility tree</a></p></header>
{tiled_note}
<main><img src="rendered.png" alt="Full-page styled screenshot of the frozen page"></main></body></html>"""


def _capture_screenshot_artifact(page: Any, screenshot_path: Path, dimensions: dict[str, Any]) -> tuple[str, list[Path]]:
    """Write a complete visual capture without crashing on extremely tall pages."""
    width = max(1, int(dimensions.get("width", 1)))
    height = max(1, int(dimensions.get("height", 1)))
    if height <= MAX_SINGLE_SCREENSHOT_HEIGHT:
        try:
            page.screenshot(path=str(screenshot_path), full_page=True, animations="disabled")
            return "full_page", [screenshot_path]
        except Exception:
            # A normal page can still hit a browser-specific bitmap limit.  A
            # viewport preview is retained instead of losing all visual evidence.
            page.screenshot(path=str(screenshot_path), full_page=False, animations="disabled")
            return "viewport_fallback", [screenshot_path]

    tile_dir = screenshot_path.parent / "rendered_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_paths: list[Path] = []
    original_viewport = page.viewport_size
    original_styles = page.evaluate("""() => ({
      html: document.documentElement.getAttribute('style'),
      body: document.body ? document.body.getAttribute('style') : null
    })""")
    try:
        # Some captured pages place their content outside the document scroller.
        # Give the root a temporary explicit height so each screenshot tile can
        # be reached by scrolling without modifying the saved rendered DOM.
        page.evaluate("""height => {
          document.documentElement.style.setProperty('height', `${height}px`, 'important');
          if (document.body) document.body.style.setProperty('min-height', `${height}px`, 'important');
        }""", height)
        page.set_viewport_size({"width": width, "height": min(SCREENSHOT_TILE_HEIGHT, height)})
        for index, top in enumerate(range(0, height, SCREENSHOT_TILE_HEIGHT)):
            tile = tile_dir / f"tile-{index:04d}.jpg"
            page.evaluate("top => window.scrollTo(0, top)", top)
            # JPEG tiles avoid retaining hundreds of megabytes of PNG data for
            # extreme pages while leaving visual text and contrast inspectable.
            page.screenshot(path=str(tile), full_page=False, animations="disabled", scale="css", type="jpeg", quality=70)
            tile_paths.append(tile)
        page.evaluate("top => window.scrollTo(0, top)", 0)
        page.screenshot(path=str(screenshot_path), full_page=False, animations="disabled", scale="css")
    finally:
        page.evaluate("""styles => {
          const restore = (element, value) => value === null ? element.removeAttribute('style') : element.setAttribute('style', value);
          restore(document.documentElement, styles.html);
          if (document.body) restore(document.body, styles.body);
          window.scrollTo(0, 0);
        }""", original_styles)
        if original_viewport:
            page.set_viewport_size(original_viewport)
    # Keep the documented single-file path available as a quick preview while
    # the tile manifest gives raters the complete page.
    manifest = tile_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "screenshot_mode": "tiled_full_page",
        "document_dimensions": {"width": width, "height": height},
        "tile_height": SCREENSHOT_TILE_HEIGHT,
        "tiles": [{"path": path.name, "top": index * SCREENSHOT_TILE_HEIGHT}
                  for index, path in enumerate(tile_paths)],
    }, indent=2), encoding="utf-8")
    return "tiled_full_page", [screenshot_path, manifest, *tile_paths]


def _use_same_session_sidecars(
    *, source: Path, destination: Path, site_id: str, capture_url: str, captured_at: str,
    rendered_path: Path, archive_path: Path, screenshot_path: Path, ax_path: Path,
    visual_path: Path, viewer_path: Path, resource_path: Path, style_state: dict[str, Any],
    successful_stylesheets: list[dict[str, Any]], failed_stylesheets: list[dict[str, str]],
    request_failures: list[dict[str, str]], dimensions: dict[str, Any],
) -> dict[str, Any] | None:
    """Use collector-side evidence when later CSS hydration is unavailable.

    This preserves the original same-session screenshot, accessibility tree and
    computed visual snapshot rather than pretending an unstyled local replay is
    equivalent to the captured page.
    """
    sidecars = {
        "screenshot": source.with_name("0.png"),
        "ax": source.with_name("0.ax.json"),
        "visual": source.with_name("0.visual.json"),
    }
    if not all(path.is_file() and path.stat().st_size > 0 for path in sidecars.values()):
        return None
    shutil.copy2(sidecars["screenshot"], screenshot_path)
    shutil.copy2(sidecars["ax"], ax_path)
    shutil.copy2(sidecars["visual"], visual_path)
    viewer_path.write_text(
        _viewer_html(site_id=site_id, capture_url=capture_url, captured_at=captured_at, screenshot_mode="same_session_sidecar"),
        encoding="utf-8",
    )
    resource_path.write_text(json.dumps({
        "capture_url": capture_url,
        "captured_at": captured_at,
        "capture_contract": "same_session_collector_sidecars_v1",
        "fallback_reason": "frozen DOM CSS hydration loaded no external stylesheet",
        "style_state": style_state,
        "successful_stylesheet_count": len(successful_stylesheets),
        "failed_stylesheet_requests": failed_stylesheets,
        "request_failure_count": len(request_failures),
        "request_failures": request_failures[:100],
        "sidecar_sources": {name: str(path) for name, path in sidecars.items()},
    }, indent=2), encoding="utf-8")
    source_copy = destination / "source.html"
    return {
        "capture_status": "complete",
        "capture_contract": "same_session_collector_sidecars_v1",
        "capture_url": capture_url,
        "captured_at": captured_at,
        "viewport": None,
        "document_dimensions": dimensions,
        "screenshot_mode": "same_session_collector",
        "style_state": style_state,
        "successful_stylesheet_count": len(successful_stylesheets),
        "failed_stylesheet_count": len(failed_stylesheets),
        "hashes": {str(path.relative_to(destination)): _sha256(path) for path in (
            source_copy, viewer_path, rendered_path, archive_path, screenshot_path,
            ax_path, visual_path, resource_path,
        ) if path.is_file()},
    }


def _capture_live_verified_shell(
    source: Path, axe_report: Path, site_id: str, destination: Path, timeout_ms: int,
) -> dict[str, Any]:
    """Rehydrate a JS app shell only after strict shell and axe-target matching."""
    from playwright.sync_api import sync_playwright

    source_html = source.read_text(encoding="utf-8", errors="replace")
    if not _is_app_shell(source_html):
        raise RuntimeError("Live rehydration is allowed only for a frozen empty app shell")
    capture_url = _capture_url(axe_report, site_id)
    expected_targets = _saved_target_fragments(axe_report)
    if not expected_targets:
        raise RuntimeError("Live shell rehydration has no saved target fragments to verify")
    rendered_path = destination / "rendered_dom.html"
    viewer_path = destination / "rendered.html"
    archive_path = destination / "rendered.mhtml"
    screenshot_path = destination / "rendered.png"
    ax_path = destination / "live_ax.json"
    visual_path = destination / "computed_visual.json"
    resource_path = destination / "resource_audit.json"
    responses: list[dict[str, Any]] = []
    request_failures: list[dict[str, str]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720}, reduced_motion="reduce",
            ignore_https_errors=True,
        )
        page = context.new_page()
        page.on("response", lambda response: responses.append({
            "url": response.url, "status": response.status,
            "resource_type": response.request.resource_type,
        }))
        page.on("requestfailed", lambda request: request_failures.append({
            "url": request.url, "resource_type": request.resource_type,
            "error": str(request.failure or "request_failed"),
        }))
        response = page.goto(capture_url, wait_until="load", timeout=timeout_ms)
        if response is None or not (200 <= response.status < 400):
            raise RuntimeError(f"Live shell document failed to load: {response.status if response else 'no response'}")
        live_initial_html = response.text()
        if _canonical_markup(source_html) != _canonical_markup(live_initial_html):
            raise RuntimeError("Live initial document no longer matches the frozen app shell")
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 20_000))
        except Exception:
            pass
        matched = 0
        for selector, expected_html in expected_targets:
            try:
                actual = page.locator(selector).evaluate_all("elements => elements.map(element => element.outerHTML)")
            except Exception as exc:
                raise RuntimeError(f"Saved target selector is invalid on live shell: {selector}: {exc}") from exc
            expected = _canonical_markup(expected_html)
            if any(_canonical_markup(value) == expected for value in actual):
                matched += 1
        if matched != len(expected_targets):
            raise RuntimeError(f"Live shell matched only {matched}/{len(expected_targets)} saved target fragments")
        page.add_style_tag(content="*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}")
        rendered_path.write_text(page.content(), encoding="utf-8")
        session = page.context.new_cdp_session(page)
        ax_path.write_text(json.dumps(session.send("Accessibility.getFullAXTree"), indent=2), encoding="utf-8")
        visual_payload = page.evaluate(VISUAL_EVIDENCE_SCRIPT)
        visual_path.write_text(json.dumps(visual_payload, indent=2), encoding="utf-8")
        dimensions = page.evaluate("""() => ({
          width: Math.max(document.documentElement.scrollWidth, document.body?.scrollWidth || 0),
          height: Math.max(document.documentElement.scrollHeight, document.body?.scrollHeight || 0,
            document.documentElement.offsetHeight, document.body?.offsetHeight || 0)
        })""")
        screenshot_mode, screenshot_assets = _capture_screenshot_artifact(page, screenshot_path, dimensions)
        style_state = page.evaluate("""() => ({
          externalStylesheetLinks: document.querySelectorAll('link[rel="stylesheet"]').length,
          stylesheetObjects: document.styleSheets.length,
          inlineStyleElements: document.querySelectorAll('style').length,
          loadedImages: [...document.images].filter(image => image.complete && image.naturalWidth > 0).length,
          failedImages: [...document.images].filter(image => image.complete && image.naturalWidth === 0).length
        })""")
        archive_path.write_text(
            str(session.send("Page.captureSnapshot", {"format": "mhtml"})["data"]),
            encoding="utf-8",
        )
        browser.close()
    stylesheet_responses = [item for item in responses if item["resource_type"] == "stylesheet"]
    successful_stylesheets = [item for item in stylesheet_responses if 200 <= int(item["status"]) < 400]
    failed_stylesheets = [item for item in request_failures if item["resource_type"] == "stylesheet"]
    if style_state["externalStylesheetLinks"] and not successful_stylesheets:
        raise RuntimeError("Verified live shell loaded no external stylesheet")
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    viewer_path.write_text(
        _viewer_html(site_id=site_id, capture_url=capture_url, captured_at=captured_at, screenshot_mode=screenshot_mode),
        encoding="utf-8",
    )
    resource_path.write_text(json.dumps({
        "capture_url": capture_url, "captured_at": captured_at,
        "network_policy": "verified live app-shell hydration; reduced motion and animations frozen",
        "integrity": {
            "frozen_initial_document_matches_live": True,
            "saved_target_fragment_count": len(expected_targets),
            "matched_target_fragment_count": matched,
        },
        "style_state": style_state,
        "screenshot_mode": screenshot_mode,
        "successful_stylesheet_count": len(successful_stylesheets),
        "stylesheet_responses": stylesheet_responses,
        "failed_stylesheet_requests": failed_stylesheets,
        "request_failure_count": len(request_failures),
        "request_failures": request_failures[:100],
    }, indent=2), encoding="utf-8")
    return {
        "capture_status": "complete",
        "capture_contract": "verified_live_app_shell_hydration_v2",
        "capture_url": capture_url, "captured_at": captured_at,
        "viewport": {"width": 1280, "height": 720},
        "document_dimensions": dimensions, "screenshot_mode": screenshot_mode,
        "style_state": style_state, "successful_stylesheet_count": len(successful_stylesheets),
        "failed_stylesheet_count": len(failed_stylesheets),
        "integrity": {"shell_match": True, "saved_targets": len(expected_targets), "matched_targets": matched},
        "hashes": {str(path.relative_to(destination)): _sha256(path) for path in (
            destination / "source.html", viewer_path, rendered_path, archive_path,
            ax_path, visual_path, resource_path, *screenshot_assets,
        )},
    }


def _capture_site(source: Path, axe_report: Path, site_id: str, destination: Path, timeout_ms: int) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    destination.mkdir(parents=True, exist_ok=True)
    source_copy = destination / "source.html"
    shutil.copy2(source, source_copy)
    capture_url = _capture_url(axe_report, site_id)
    source_html = source.read_text(encoding="utf-8", errors="replace")
    hydrated_html = _with_base(source_html, capture_url)
    rendered_path = destination / "rendered_dom.html"
    viewer_path = destination / "rendered.html"
    archive_path = destination / "rendered.mhtml"
    screenshot_path = destination / "rendered.png"
    ax_path = destination / "live_ax.json"
    visual_path = destination / "computed_visual.json"
    resource_path = destination / "resource_audit.json"
    responses: list[dict[str, Any]] = []
    request_failures: list[dict[str, str]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720}, reduced_motion="reduce",
            java_script_enabled=False,
        )
        page = context.new_page()

        def route_resource(route):
            if route.request.resource_type in {"script", "xhr", "fetch", "websocket", "eventsource", "media"}:
                route.abort()
            else:
                route.continue_()

        page.route("**/*", route_resource)
        page.on("response", lambda response: responses.append({
            "url": response.url, "status": response.status,
            "resource_type": response.request.resource_type,
        }))
        page.on("requestfailed", lambda request: request_failures.append({
            "url": request.url, "resource_type": request.resource_type,
            "error": str(request.failure or "request_failed"),
        }))
        page.set_content(hydrated_html, wait_until="load", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15_000))
        except Exception:
            pass
        rendered_path.write_text(page.content(), encoding="utf-8")
        session = page.context.new_cdp_session(page)
        ax_payload = session.send("Accessibility.getFullAXTree")
        ax_path.write_text(json.dumps(ax_payload, indent=2), encoding="utf-8")
        visual_payload = page.evaluate(VISUAL_EVIDENCE_SCRIPT)
        visual_path.write_text(json.dumps(visual_payload, indent=2), encoding="utf-8")
        dimensions = page.evaluate("""() => ({
          width: Math.max(document.documentElement.scrollWidth, document.body?.scrollWidth || 0),
          height: Math.max(document.documentElement.scrollHeight, document.body?.scrollHeight || 0,
            document.documentElement.offsetHeight, document.body?.offsetHeight || 0)
        })""")
        screenshot_mode, screenshot_assets = _capture_screenshot_artifact(page, screenshot_path, dimensions)
        style_state = page.evaluate("""() => ({
          externalStylesheetLinks: document.querySelectorAll('link[rel="stylesheet"]').length,
          stylesheetObjects: document.styleSheets.length,
          inlineStyleElements: document.querySelectorAll('style').length,
          loadedImages: [...document.images].filter(image => image.complete && image.naturalWidth > 0).length,
          failedImages: [...document.images].filter(image => image.complete && image.naturalWidth === 0).length
        })""")
        archive_path.write_text(
            str(session.send("Page.captureSnapshot", {"format": "mhtml"})["data"]),
            encoding="utf-8",
        )
        browser.close()
    stylesheet_responses = [item for item in responses if item["resource_type"] == "stylesheet"]
    successful_stylesheets = [item for item in stylesheet_responses if 200 <= int(item["status"]) < 400]
    failed_stylesheets = [item for item in request_failures if item["resource_type"] == "stylesheet"]
    if style_state["externalStylesheetLinks"] and not successful_stylesheets:
        if _is_app_shell(source_html):
            return _capture_live_verified_shell(source, axe_report, site_id, destination, timeout_ms)
        sidecar_capture = _use_same_session_sidecars(
            source=source, destination=destination, site_id=site_id, capture_url=capture_url,
            captured_at=dt.datetime.now(dt.timezone.utc).isoformat(), rendered_path=rendered_path,
            archive_path=archive_path, screenshot_path=screenshot_path, ax_path=ax_path,
            visual_path=visual_path, viewer_path=viewer_path, resource_path=resource_path,
            style_state=style_state, successful_stylesheets=successful_stylesheets,
            failed_stylesheets=failed_stylesheets, request_failures=request_failures,
            dimensions=dimensions,
        )
        if sidecar_capture is not None:
            return sidecar_capture
        raise RuntimeError(
            f"Visual evidence invalid: {style_state['externalStylesheetLinks']} external stylesheets declared but none loaded"
        )
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    viewer_path.write_text(
        _viewer_html(site_id=site_id, capture_url=capture_url, captured_at=captured_at, screenshot_mode=screenshot_mode),
        encoding="utf-8",
    )
    resource_path.write_text(json.dumps({
        "capture_url": capture_url, "captured_at": captured_at,
        "network_policy": "scripts/xhr/fetch/websocket/eventsource/media blocked; styles/fonts/images allowed",
        "style_state": style_state,
        "screenshot_mode": screenshot_mode,
        "successful_stylesheet_count": len(successful_stylesheets),
        "stylesheet_responses": stylesheet_responses,
        "failed_stylesheet_requests": failed_stylesheets,
        "request_failure_count": len(request_failures),
        "request_failures": request_failures[:100],
    }, indent=2), encoding="utf-8")
    return {
        "capture_status": "complete",
        "capture_contract": "frozen_dom_live_resource_hydration_v2",
        "capture_url": capture_url,
        "captured_at": captured_at,
        "viewport": {"width": 1280, "height": 720},
        "document_dimensions": dimensions,
        "screenshot_mode": screenshot_mode,
        "style_state": style_state,
        "successful_stylesheet_count": len(successful_stylesheets),
        "failed_stylesheet_count": len(failed_stylesheets),
        "hashes": {str(path.relative_to(destination)): _sha256(path) for path in (
            source_copy, viewer_path, rendered_path, archive_path,
            ax_path, visual_path, resource_path, *screenshot_assets,
        )},
    }


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    partition = str(getattr(args, "partition", "test"))
    if partition not in {"val", "test"}:
        raise ValueError("Annotation partition must be val or test")
    sites = list(split.get(partition, []))
    if not sites:
        raise ValueError(f"Frozen split has no {partition} sites")
    criteria = _criterion_records(args.registry, args.rule_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    captures = {}
    failures = []
    for index, site_id in enumerate(sites, 1):
        print(f"[{index}/{len(sites)}] capture {site_id}", flush=True)
        source = args.corpus_dir / site_id / "0.html"
        destination = args.output_dir / "evidence" / site_id
        attempt_errors: list[str] = []
        capture_attempts = max(1, int(getattr(args, "capture_attempts", 2)))
        for attempt in range(1, capture_attempts + 1):
            try:
                if not source.is_file():
                    raise FileNotFoundError(source)
                axe_report = args.corpus_dir / site_id / "page-0_home.json"
                if not axe_report.is_file():
                    raise FileNotFoundError(axe_report)
                capture = _capture_site(source, axe_report, site_id, destination, args.network_timeout_ms)
                capture["capture_attempts"] = attempt
                capture["prior_attempt_errors"] = attempt_errors
                captures[site_id] = capture
                if attempt > 1:
                    print(f"[{index}/{len(sites)}] capture recovered {site_id} on attempt {attempt}", flush=True)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:2000]
                attempt_errors.append(error)
                print(f"[{index}/{len(sites)}] WARNING capture attempt {attempt}/{capture_attempts} failed for {site_id}: {error}", flush=True)
        else:
            captures[site_id] = {
                "capture_status": "failed", "error": attempt_errors[-1],
                "capture_attempts": capture_attempts, "attempt_errors": attempt_errors,
            }
            failures.append(site_id)
    if failures and not args.allow_capture_failures:
        raise RuntimeError(f"Annotation packet capture failed for {len(failures)} sites: {failures[:3]}")
    cases = []
    for site_id in sites:
        for criterion_id, criterion in criteria.items():
            case_id = _case_id(site_id, criterion_id, args.seed)
            relative = Path("evidence") / site_id
            cases.append({
                "case_id": case_id,
                "site_id": site_id,
                "criterion_id": criterion_id,
                "criterion_name": criterion["name"],
                "level": criterion.get("level"),
                "required_evidence": criterion.get("required_evidence", []),
                "evidence_paths": {
                    "source_html": str(relative / "source.html"),
                    "rendered_html": str(relative / "rendered.html"),
                    "rendered_dom": str(relative / "rendered_dom.html"),
                    "rendered_archive": str(relative / "rendered.mhtml"),
                    "rendered_screenshot": str(relative / "rendered.png"),
                    "rendered_screenshot_tiles": str(relative / "rendered_tiles" / "manifest.json"),
                    "live_accessibility_tree": str(relative / "live_ax.json"),
                    "computed_visual": str(relative / "computed_visual.json"),
                    "resource_audit": str(relative / "resource_audit.json"),
                },
                "capture_status": captures[site_id]["capture_status"],
            })
    coordinator = args.output_dir / "coordinator"
    raters = args.output_dir / "rater_packets"
    coordinator.mkdir(exist_ok=True); raters.mkdir(exist_ok=True)
    identity_path = coordinator / "identity_map.json"
    identity_path.write_text(json.dumps({"schema_version": 1, "cases": cases}, indent=2), encoding="utf-8")
    # ayman: rater is used here for producing two independent, blinded pass/fail judgements; the selected partition is validation or test, never training.
    for rater_index in (1, 2):
        rows = [_rater_case(case) for case in cases]
        random.Random(args.seed + rater_index).shuffle(rows)
        (raters / f"rater_{rater_index}.json").write_text(json.dumps({
            "schema_version": 1,
            "rater_id": f"annotator-{rater_index}",
            "blinded_to": ["axe labels", "model predictions", "fusion outputs", "repair conditions"],
            "allowed_statuses": ["pass", "fail"],
            "cases": rows,
        }, indent=2), encoding="utf-8")
    adjudication = {
        "schema_version": 1,
        "annotation_protocol": "dual_independent_then_adjudicated",
        "pairs": [{
            "case_id": case["case_id"], "site_id": case["site_id"], "criterion_id": case["criterion_id"],
            "status": None, "annotator_ids": ["annotator-1", "annotator-2"],
            "adjudicated": False, "evidence": "",
        } for case in cases],
    }
    adjudication_path = coordinator / "independent_detection_truth.json"
    adjudication_path.write_text(json.dumps(adjudication, indent=2), encoding="utf-8")
    instructions_path = args.output_dir / "ANNOTATION_INSTRUCTIONS.md"
    instructions_path.write_text(
        "# Detection annotation instructions\n\n"
        "Each rater works independently from their own JSON file and must not inspect `coordinator/`, axe reports, model predictions, or the other rater's answers. "
        "Open `rendered.html` for the correctly styled visual evidence viewer; use `rendered.mhtml` in Chrome when an interactive archived view is needed. "
        "When `rendered_screenshot_tiles` exists, inspect every ordered tile listed by its manifest; `rendered.png` is only a preview for that long page. "
        "Review `source.html`/`rendered_dom.html` for markup, `computed_visual.json` for measured contrast evidence, and `live_ax.json` for the browser accessibility tree. "
        "Set `status` to `pass` or `fail`, record applicable exceptions and evidence, and give confidence from 1 to 5. "
        "Do not infer conformance from missing evidence; report capture problems to the coordinator. "
        "After both sheets are frozen, the coordinator adjudicates disagreements in `coordinator/independent_detection_truth.json`.\n",
        encoding="utf-8",
    )
    artifacts = [path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "packet_manifest.json"]
    manifest = {
        "schema_version": 1,
        "status": "blinded_detection_annotation_packet",
        "split": str(args.split.resolve()), "split_sha256": _sha256(args.split), "split_hash": split.get("split_hash"),
        "registry": str(args.registry.resolve()), "registry_sha256": _sha256(args.registry),
        "partition": partition,
        "rule_ids": args.rule_ids, "criteria": sorted(criteria), "site_count": len(sites), "case_count": len(cases),
        "capture_failures": failures,
        "capture_contract": "frozen_dom_live_resource_hydration_v2",
        "captures": captures,
        "blinding": {"rater_packets_exclude_site_identity": False, "exclude_axe_and_model_outputs": True},
        "artifacts": {str(path.relative_to(args.output_dir)): _sha256(path) for path in sorted(artifacts)},
    }
    (args.output_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _cohen_kappa(left: list[str], right: list[str]) -> float:
    # Cohen's kappa reports rater agreement beyond the agreement expected by
    # chance from each rater's pass/fail label distribution.
    if not left:
        return 0.0
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = tuple(sorted(set(left) | set(right)))
    expected = sum((left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _rater_cases(payload: Any, annotator_index: int) -> list[dict[str, Any]]:
    """Read either the generated packet wrapper or a frozen completed sheet.

    Raters may return the packet's ``{"cases": [...]}`` JSON object or the
    extracted top-level case array. Both preserve the same case records; this
    function deliberately does not coerce decisions or alter evidence.
    """

    if isinstance(payload, dict):
        rows = payload.get("cases")
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = None
    if not isinstance(rows, list) or not all(isinstance(item, dict) for item in rows):
        raise ValueError(
            f"annotator-{annotator_index} sheet must be a case array or an object containing 'cases'"
        )
    return rows


def finalize_packet(packet_dir: Path, output: Path) -> dict[str, Any]:
    identity_rows = json.loads((packet_dir / "coordinator/identity_map.json").read_text(encoding="utf-8"))["cases"]
    identities = {item["case_id"]: item for item in identity_rows}
    # ayman: rater is used here for building the reference labels: agreements are retained and disagreements must be resolved by the adjudicator.
    rater_rows = {}
    for index in (1, 2):
        payload = json.loads((packet_dir / f"rater_packets/rater_{index}.json").read_text(encoding="utf-8"))
        rows = {item["case_id"]: item for item in _rater_cases(payload, index)}
        if set(rows) != set(identities):
            raise ValueError(f"annotator-{index} sheet does not match the frozen case universe")
        for case_id, row in rows.items():
            if row.get("status") not in {"pass", "fail", "needs_human_review"}:
                raise ValueError(f"annotator-{index} has an incomplete status for {case_id}")
            confidence = row.get("confidence")
            if not isinstance(confidence, (int, float)) or not 1 <= confidence <= 5:
                raise ValueError(f"annotator-{index} confidence must be 1..5 for {case_id}")
        rater_rows[index] = rows
    ordered_ids = sorted(identities)
    left = [rater_rows[1][case_id]["status"] for case_id in ordered_ids]
    right = [rater_rows[2][case_id]["status"] for case_id in ordered_ids]
    coordination_required = [
        case_id for case_id in ordered_ids
        if (
            rater_rows[1][case_id]["status"] != rater_rows[2][case_id]["status"]
            or rater_rows[1][case_id]["status"] == "needs_human_review"
            or rater_rows[2][case_id]["status"] == "needs_human_review"
        )
    ]
    adjudication_path = packet_dir / "coordinator/independent_detection_truth.json"
    adjudication_rows = {
        item["case_id"]: item
        for item in json.loads(adjudication_path.read_text(encoding="utf-8"))["pairs"]
    }
    final_rows = []
    for case_id in ordered_ids:
        identity = identities[case_id]
        if case_id in coordination_required:
            adjudicated = adjudication_rows.get(case_id, {})
            if adjudicated.get("status") not in {"pass", "fail"} or not adjudicated.get("adjudicated") or not str(adjudicated.get("evidence", "")).strip():
                raise ValueError(f"Disagreement or non-binary rating requires completed adjudication with evidence: {case_id}")
            status = adjudicated["status"]
            evidence = str(adjudicated["evidence"])
        else:
            status = rater_rows[1][case_id]["status"]
            evidence = "Independent annotators agreed. " + " | ".join(filter(None, (
                str(rater_rows[1][case_id].get("evidence_notes", "")).strip(),
                str(rater_rows[2][case_id].get("evidence_notes", "")).strip(),
            )))
        final_rows.append({
            "site_id": identity["site_id"], "criterion_id": identity["criterion_id"],
            "status": status, "annotator_ids": ["annotator-1", "annotator-2"],
            "adjudicated": True, "evidence": evidence,
        })
    result = {
        "schema_version": 1,
        "annotation_protocol": "dual_independent_then_adjudicated",
        "blinded": True,
        "agreement": {
            "pair_count": len(ordered_ids),
            "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(ordered_ids),
            "cohen_kappa": _cohen_kappa(left, right),
            "disagreement_count": sum(
                rater_rows[1][case_id]["status"] != rater_rows[2][case_id]["status"]
                for case_id in ordered_ids
            ),
            "coordination_required_count": len(coordination_required),
        },
        "pairs": final_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rule-ids", nargs="+", required=True)
    parser.add_argument("--partition", choices=("val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--network-timeout-ms", type=int, default=60_000)
    parser.add_argument("--capture-attempts", type=int, default=2, help="Attempts per site before recording a capture failure.")
    parser.add_argument("--allow-capture-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = build_packet(parse_args())
    print(json.dumps({key: report[key] for key in ("status", "site_count", "case_count", "capture_failures")}, indent=2))


if __name__ == "__main__":
    main()
