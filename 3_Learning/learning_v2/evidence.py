"""Evidence ingestion from saved axe-core HTML/report pairs.

The corpus is treated as immutable input. Optional rendered capture operates on
the saved local HTML and never revisits the live website.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup, Tag

from .contracts import EvidenceNode, NodeIdentity, ScanArtifact


COLLECTOR_VERSION = "snapshot-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_site_dirs(corpus_dir: Path) -> list[Path]:
    return [
        directory for directory in sorted(corpus_dir.iterdir())
        if directory.is_dir() and (directory / "0.html").is_file() and (directory / "page-0_home.json").is_file()
    ]


def _attrs(tag: Tag) -> dict[str, Any]:
    result = {}
    for name, value in tag.attrs.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
        elif isinstance(value, list):
            result[name] = [str(part) for part in value]
        else:
            result[name] = str(value)
    return result


def _css_path(tag: Tag) -> str:
    parts = []
    current: Tag | None = tag
    while isinstance(current, Tag) and current.name != "[document]":
        if current.get("id"):
            parts.append(f"{current.name}#{current.get('id')}")
            break
        sibling_index = 1
        for sibling in current.previous_siblings:
            if isinstance(sibling, Tag) and sibling.name == current.name:
                sibling_index += 1
        parts.append(f"{current.name}:nth-of-type({sibling_index})")
        current = current.parent if isinstance(current.parent, Tag) else None
    return " > ".join(reversed(parts))


def _accessible_name(tag: Tag) -> str:
    for name in ("aria-label", "title", "alt"):
        value = tag.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    text = tag.get_text(" ", strip=True)
    return text[:300]


def collect_static_evidence(site_dir: Path) -> ScanArtifact:
    html_path = site_dir / "0.html"; axe_path = site_dir / "page-0_home.json"
    html = html_path.read_text(encoding="utf-8", errors="replace")
    report = json.loads(axe_path.read_text(encoding="utf-8"))
    soup = BeautifulSoup(html, "lxml")
    tags = list(soup.find_all(True)); tag_to_id = {id(tag): index for index, tag in enumerate(tags)}
    nodes = []
    for index, tag in enumerate(tags):
        parent = tag.parent if isinstance(tag.parent, Tag) else None
        nodes.append(EvidenceNode(
            identity=NodeIdentity(index, _css_path(tag), tag.name, tag_to_id.get(id(parent)) if parent else None),
            attributes=_attrs(tag), text=tag.get_text(" ", strip=True)[:500],
            accessible_name=_accessible_name(tag), role=str(tag.get("role", "")),
        ))
    return ScanArtifact(
        site_id=site_dir.name, source_url=str(report.get("url", "")),
        html_path=str(html_path), axe_path=str(axe_path),
        html_sha256=sha256_file(html_path), axe_sha256=sha256_file(axe_path),
        collector_version=COLLECTOR_VERSION, nodes=nodes,
    )


def capture_local_browser_evidence(html_path: Path, screenshot_path: Path | None = None) -> dict[str, Any]:
    """Capture computed layout and Chromium's accessibility tree for saved HTML."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(html_path.resolve().as_uri(), wait_until="domcontentloaded")
        layout = page.locator("*").evaluate_all(
            """els => { const path = el => { const parts=[]; let current=el; while(current && current.nodeType===1) {
              const tag=current.tagName.toLowerCase(); if(current.id) { parts.unshift(`${tag}#${current.id}`); break; }
              let n=1, sibling=current.previousElementSibling; while(sibling) { if(sibling.tagName===current.tagName) n++; sibling=sibling.previousElementSibling; }
              parts.unshift(`${tag}:nth-of-type(${n})`); current=current.parentElement;
            } return parts.join(' > '); }; return els.map((el, i) => { const r=el.getBoundingClientRect(); const s=getComputedStyle(el); return {
              node_id:i, css_path:path(el), tag:el.tagName.toLowerCase(), x:r.x, y:r.y, width:r.width, height:r.height,
              display:s.display, visibility:s.visibility, opacity:s.opacity, color:s.color,
              backgroundColor:s.backgroundColor, outline:s.outline, zIndex:s.zIndex
            }}); }"""
        )
        session = page.context.new_cdp_session(page)
        ax_tree = session.send("Accessibility.getFullAXTree").get("nodes", [])
        page.evaluate("document.body && document.body.focus()")
        focus_sequence = []
        seen_focus = set()
        for order in range(100):
            page.keyboard.press("Tab")
            state = page.evaluate(
                """order => { const el=document.activeElement; if(!el || el===document.body) return null;
                const path = el => { const parts=[]; let current=el; while(current && current.nodeType===1) {
                  const tag=current.tagName.toLowerCase(); if(current.id) { parts.unshift(`${tag}#${current.id}`); break; }
                  let n=1, sibling=current.previousElementSibling; while(sibling) { if(sibling.tagName===current.tagName) n++; sibling=sibling.previousElementSibling; }
                  parts.unshift(`${tag}:nth-of-type(${n})`); current=current.parentElement;
                } return parts.join(' > '); }; const r=el.getBoundingClientRect(); const s=getComputedStyle(el); return {
                  order, css_path:path(el), tag:el.tagName.toLowerCase(), role:el.getAttribute('role')||'',
                  accessible_label:el.getAttribute('aria-label')||el.innerText||el.getAttribute('title')||'',
                  x:r.x,y:r.y,width:r.width,height:r.height,outline:s.outline
                }; }""",
                order,
            )
            if state is None:
                continue
            key = state["css_path"]
            if key in seen_focus:
                break
            seen_focus.add(key); focus_sequence.append(state)
        if screenshot_path:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_path), full_page=True)
        browser.close()
    return {
        "viewport": {"width": 1280, "height": 720}, "layout": layout,
        "accessibility_tree": ax_tree, "focus_sequence": focus_sequence,
    }


def write_evidence(artifact: ScanArtifact, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")


def collect_many(site_dirs: Iterable[Path], output_dir: Path, *, rendered: bool = False) -> dict[str, int]:
    completed = failed = 0
    for site_dir in site_dirs:
        try:
            artifact = collect_static_evidence(site_dir)
            if rendered:
                try:
                    artifact.browser_evidence = capture_local_browser_evidence(
                        site_dir / "0.html", output_dir / site_dir.name / "screenshot.png"
                    )
                except Exception as exc:  # capture failures must be explicit
                    artifact.collection_failures.append(f"browser_capture: {type(exc).__name__}: {exc}")
            write_evidence(artifact, output_dir / site_dir.name / "evidence.json")
            completed += 1
        except Exception:
            failed += 1
    return {"completed": completed, "failed": failed}
