"""Isolated browser and deterministic validation for bounded Phase 9 repairs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image, ImageChops

from learning_v2.baselines import deterministic_findings

from .contracts import RepairProposal, ValidationResult
from .patches import PatchApplicationError, apply_typed_patch


SEMANTIC_OR_CONTEXTUAL_RULES = frozenset({
    "button-name", "document-title", "html-has-lang", "html-lang-valid", "image-alt",
    "label", "link-name", "select-name",
})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:120] or "attempt"


def _finding_pairs(findings: list[Any]) -> set[tuple[str, str]]:
    return {
        (finding.rule_id, finding.node.css_path if finding.node else "page")
        for finding in findings
    }


def _normalise_axe(report: dict[str, Any]) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    violations: list[dict[str, Any]] = []
    pairs: set[tuple[str, str]] = set()
    for violation in report.get("violations", []):
        rule = str(violation.get("id", "unknown"))
        nodes = []
        for node in violation.get("nodes", []):
            target = node.get("target", [])
            selector = str(target[0]) if target else "page"
            pairs.add((rule, selector))
            nodes.append({
                "target": target,
                "html": str(node.get("html", ""))[:3000],
                "failure_summary": str(node.get("failureSummary", ""))[:3000],
            })
        violations.append({"id": rule, "impact": violation.get("impact"), "nodes": nodes})
    return violations, pairs


def _browser_snapshot(html_path: Path, axe_js: Path, screenshot_path: Path) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    axe_source = axe_js.read_text(encoding="utf-8")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720}, reduced_motion="reduce")

        def route_request(route):
            if route.request.url.startswith(("http://", "https://")):
                route.abort()
            else:
                route.continue_()

        page.route("**/*", route_request)
        page.goto(html_path.resolve().as_uri(), wait_until="domcontentloaded")
        page.add_script_tag(content=axe_source)
        axe_report = page.evaluate("""async () => await axe.run(document, {
          resultTypes: ['violations'],
          rules: {'region': {enabled: false}}
        })""")
        functional = page.evaluate("""() => ({
          title: document.title,
          htmlLang: document.documentElement.lang,
          links: [...document.querySelectorAll('a[href]')].map(e => e.getAttribute('href')),
          forms: [...document.forms].map(e => e.getAttribute('action')),
          controls: [...document.querySelectorAll('button,input,select,textarea')].map(e => ({
            tag:e.tagName.toLowerCase(), type:e.getAttribute('type'), id:e.id, name:e.getAttribute('name')
          })),
          scripts: document.scripts.length,
          modals: document.querySelectorAll('[role=dialog],[aria-modal=true],dialog').length,
          liveRegions: document.querySelectorAll('[aria-live],[role=alert],[role=status]').length
        })""")
        page.evaluate("document.body && document.body.focus()")
        focus_sequence: list[str] = []
        for _ in range(100):
            page.keyboard.press("Tab")
            selector = page.evaluate("""() => { const el=document.activeElement;
              if(!el || el===document.body) return null; const parts=[]; let current=el;
              while(current && current.nodeType===1) { const tag=current.tagName.toLowerCase();
                if(current.id) { parts.unshift(`${tag}#${CSS.escape(current.id)}`); break; }
                let n=1,s=current.previousElementSibling; while(s){if(s.tagName===current.tagName)n++;s=s.previousElementSibling;}
                parts.unshift(`${tag}:nth-of-type(${n})`); current=current.parentElement; }
              return parts.join(' > '); }""")
            if not selector:
                continue
            if selector in focus_sequence:
                break
            focus_sequence.append(selector)
        session = page.context.new_cdp_session(page)
        raw_ax = session.send("Accessibility.getFullAXTree").get("nodes", [])
        ax_tree = []
        for node in raw_ax:
            if node.get("ignored"):
                continue
            role = (node.get("role") or {}).get("value", "")
            name = (node.get("name") or {}).get("value", "")
            properties = {
                item.get("name", ""): (item.get("value") or {}).get("value")
                for item in node.get("properties", [])
                if item.get("name") in {"disabled", "expanded", "focusable", "focused", "modal", "pressed", "required"}
            }
            ax_tree.append({"role": role, "name": name, "properties": properties})
        page.screenshot(path=str(screenshot_path), full_page=True, animations="disabled")
        browser.close()
    violations, axe_pairs = _normalise_axe(axe_report)
    return {
        "axe_violations": violations,
        "axe_pairs": sorted([list(item) for item in axe_pairs]),
        "accessibility_tree": ax_tree,
        "focus_sequence": focus_sequence,
        "functional": functional,
        "interaction_coverage": {
            "keyboard_focus_replay": True,
            "hover_replay": False,
            "modal_replay": functional["modals"] == 0,
            "live_region_replay": functional["liveRegions"] == 0,
            "note": "Modal/live-region replay is required when those mechanisms are present.",
        },
    }


def _visual_diff(before_path: Path, after_path: Path) -> dict[str, Any]:
    before = Image.open(before_path).convert("RGBA")
    after = Image.open(after_path).convert("RGBA")
    if before.size != after.size:
        return {"same_dimensions": False, "before_size": list(before.size), "after_size": list(after.size), "changed_pixel_fraction": 1.0}
    difference = ImageChops.difference(before, after).convert("L")
    histogram = difference.histogram()
    total = before.size[0] * before.size[1]
    changed = total - histogram[0]
    return {
        "same_dimensions": True,
        "before_size": list(before.size),
        "after_size": list(after.size),
        "changed_pixel_fraction": changed / total if total else 0.0,
        "difference_bbox": list(difference.getbbox()) if difference.getbbox() else None,
    }


def _visual_change_independently_verified(proposal: RepairProposal, original_finding: dict[str, Any], oracle_match: bool) -> bool:
    if not oracle_match:
        return False
    if original_finding.get("visual_verified", False):
        return True
    # Visible text insertion is expected to change pixels. It may bypass the
    # generic review threshold only when the exact operation matched hidden
    # truth and the natural-language value was independently verified.
    return bool(
        original_finding.get("semantic_verified", False)
        and any(operation.operation in {"insert_label_before", "replace_text"} for operation in proposal.operations)
    )


def _list_diff(before: list[Any], after: list[Any]) -> dict[str, Any]:
    before_json = [json.dumps(item, sort_keys=True) for item in before]
    after_json = [json.dumps(item, sort_keys=True) for item in after]
    return {
        "changed": before_json != after_json,
        "removed": [json.loads(item) for item in before_json if item not in after_json],
        "added": [json.loads(item) for item in after_json if item not in before_json],
    }


def _target_present(pairs: set[tuple[str, str]], rule_id: str, selector: str) -> bool:
    if (rule_id, selector) in pairs:
        return True
    return any(rule == rule_id for rule, _ in pairs)


def validate_repair(
    *,
    source_path: Path,
    proposal: RepairProposal,
    original_finding: dict[str, Any],
    output_dir: Path,
    axe_js: Path,
    run_browser: bool = True,
) -> ValidationResult:
    """Apply and validate a repair in a preserved attempt directory."""
    attempt_id = _safe_name(proposal.proposal_id)
    attempt_dir = output_dir / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    before_path = attempt_dir / "before.html"
    after_path = attempt_dir / "after.html"
    before_png = attempt_dir / "before.png"
    after_png = attempt_dir / "after.png"
    result_path = attempt_dir / "validation.json"

    source_bytes = source_path.read_bytes()
    source_hash = _sha256_bytes(source_bytes)
    source_html = source_bytes.decode("utf-8", errors="replace")
    shutil.copy2(source_path, before_path)
    patch_evidence: dict[str, Any] = {}
    rejection_reasons: list[str] = []
    human_review_reasons = list(proposal.human_review_reasons)
    collection_failures: list[str] = []
    try:
        patched_html, patch_evidence = apply_typed_patch(source_html, proposal)
    except PatchApplicationError as exc:
        patched_html = source_html
        patch_evidence = {"applied": False, "error": str(exc), "operations": []}
        rejection_reasons.append(f"patch_application_failed: {exc}")
    after_path.write_text(patched_html, encoding="utf-8")
    after_hash = _sha256_bytes(after_path.read_bytes())
    source_unchanged = _sha256_bytes(source_path.read_bytes()) == source_hash
    if not source_unchanged:
        rejection_reasons.append("immutable_source_changed")

    try:
        parsed = BeautifulSoup(after_path.read_text(encoding="utf-8"), "lxml")
        if not parsed.find("html"):
            rejection_reasons.append("patched_document_has_no_html_root")
        for operation in proposal.operations:
            try:
                if len(parsed.select(operation.selector)) != 1:
                    rejection_reasons.append(f"intended_node_not_uniquely_resolved: {operation.selector}")
            except Exception as exc:
                rejection_reasons.append(f"patched_selector_invalid: {operation.selector}: {exc}")
    except Exception as exc:
        rejection_reasons.append(f"parse_failed: {type(exc).__name__}: {exc}")

    before_static = _finding_pairs(deterministic_findings(before_path, site_id=str(original_finding.get("site_id", "phase9"))))
    after_static = _finding_pairs(deterministic_findings(after_path, site_id=str(original_finding.get("site_id", "phase9"))))
    before_browser: dict[str, Any] = {}; after_browser: dict[str, Any] = {}
    if run_browser:
        for label, html_path, png_path in (("before", before_path, before_png), ("after", after_path, after_png)):
            try:
                value = _browser_snapshot(html_path, axe_js, png_path)
                if label == "before": before_browser = value
                else: after_browser = value
            except Exception as exc:
                collection_failures.append(f"{label}_browser_capture: {type(exc).__name__}: {exc}")
    else:
        collection_failures.append("browser_validation_skipped")

    before_axe = {tuple(item) for item in before_browser.get("axe_pairs", [])}
    after_axe = {tuple(item) for item in after_browser.get("axe_pairs", [])}
    combined_before = before_static | before_axe
    combined_after = after_static | after_axe
    rule_id = str(original_finding.get("rule_id", ""))
    targets = original_finding.get("evidence", {}).get("target", [])
    selector = str(targets[0]) if targets else "page"
    was_present = _target_present(combined_before, rule_id, selector)
    still_present = _target_present(combined_after, rule_id, selector)
    target_resolved = was_present and not still_present
    if proposal.decision == "leave_unchanged":
        human_review_reasons.append("generator_left_finding_unchanged")
    elif not was_present:
        rejection_reasons.append("originating_detector_did_not_reproduce_before_patch")
    elif proposal.decision == "requires_human_review" and not proposal.operations:
        human_review_reasons.append("generator_requested_human_review_without_applying_a_patch")
    elif not target_resolved:
        rejection_reasons.append("target_finding_not_resolved")

    new_pairs = combined_after - combined_before
    new_regressions = [f"{rule}@{target}" for rule, target in sorted(new_pairs)]
    if new_regressions:
        rejection_reasons.append("new_in_scope_accessibility_regressions")
    oracle_operations = original_finding.get("oracle_operations") or []
    actual_operations = [operation.model_dump(mode="json") for operation in proposal.operations]
    oracle_match = bool(oracle_operations) and actual_operations == oracle_operations
    if original_finding.get("status") != "verified_fail":
        human_review_reasons.append("input_finding_is_not_independently_verified")
    if rule_id in SEMANTIC_OR_CONTEXTUAL_RULES and not original_finding.get("semantic_verified", False):
        human_review_reasons.append("semantic_or_contextual_correctness_not_independently_verified")
    for operation in proposal.operations:
        if operation.operation in {"replace_text", "insert_label_before"}:
            if not (oracle_match and original_finding.get("semantic_verified", False)):
                human_review_reasons.append("generated_natural_language_requires_review")
        if operation.operation == "set_attribute" and operation.attribute_name in {"alt", "aria-label", "lang", "role", "title"}:
            if not (oracle_match and original_finding.get("semantic_verified", False)):
                human_review_reasons.append("generated_semantic_attribute_requires_review")
    if collection_failures:
        human_review_reasons.append("browser_validation_incomplete")

    ax_diff = _list_diff(
        before_browser.get("accessibility_tree", []), after_browser.get("accessibility_tree", []),
    )
    interaction_diff = {
        "focus_sequence": _list_diff(before_browser.get("focus_sequence", []), after_browser.get("focus_sequence", [])),
        "before_coverage": before_browser.get("interaction_coverage", {}),
        "after_coverage": after_browser.get("interaction_coverage", {}),
    }
    if interaction_diff["focus_sequence"]["changed"]:
        human_review_reasons.append("keyboard_focus_sequence_changed")
    functional_before = before_browser.get("functional", {})
    functional_after = after_browser.get("functional", {})
    functional_diff = {
        "changed": functional_before != functional_after,
        "before": functional_before,
        "after": functional_after,
    }
    if functional_before and functional_after:
        for protected in ("links", "forms", "scripts"):
            if functional_before.get(protected) != functional_after.get(protected):
                rejection_reasons.append(f"functional_regression: {protected}_changed")
    visual_diff = _visual_diff(before_png, after_png) if before_png.exists() and after_png.exists() else {"available": False}
    if float(visual_diff.get("changed_pixel_fraction", 0.0)) > 0.001 and not _visual_change_independently_verified(proposal, original_finding, oracle_match):
        human_review_reasons.append("material_visual_difference_requires_review")
    if any(operation.operation == "set_style_property" for operation in proposal.operations):
        if not (oracle_match and original_finding.get("visual_verified", False)):
            human_review_reasons.append("visual_css_change_requires_review")
    after_functional = after_browser.get("functional", {})
    if after_functional.get("modals", 0) or after_functional.get("liveRegions", 0):
        human_review_reasons.append("page_specific_modal_or_live_region_replay_required")

    rejection_reasons = list(dict.fromkeys(rejection_reasons))
    human_review_reasons = list(dict.fromkeys(human_review_reasons))
    if rejection_reasons:
        outcome = "rejected"
    elif proposal.requires_human_review or human_review_reasons:
        outcome = "requires_human_review"
    else:
        outcome = "accepted"
    detector_results = {
        "originating_rule": rule_id,
        "originating_selector": selector,
        "reproduced_before": was_present,
        "present_after": still_present,
        "deterministic_before": [list(item) for item in sorted(before_static)],
        "deterministic_after": [list(item) for item in sorted(after_static)],
        "axe_before": before_browser.get("axe_violations", []),
        "axe_after": after_browser.get("axe_violations", []),
    }
    result = ValidationResult(
        attempt_id=attempt_id,
        outcome=outcome,
        target_resolved=target_resolved,
        new_regressions=new_regressions,
        human_review_reasons=human_review_reasons,
        rejection_reasons=rejection_reasons,
        before_sha256=source_hash,
        after_sha256=after_hash,
        source_unchanged=source_unchanged,
        detector_results=detector_results,
        accessibility_tree_diff=ax_diff,
        interaction_diff=interaction_diff,
        visual_diff=visual_diff,
        functional_diff=functional_diff,
        patch_evidence=patch_evidence,
        collection_failures=collection_failures,
        artifact_paths={
            "before_html": str(before_path), "after_html": str(after_path),
            "before_screenshot": str(before_png) if before_png.exists() else "",
            "after_screenshot": str(after_png) if after_png.exists() else "",
            "validation": str(result_path),
        },
        specialist_scope=[
            "originating detector", "axe-core", "deterministic HTML specialists",
            "Chromium accessibility tree", "keyboard focus replay", "visual regression", "functional DOM regression",
        ],
    )
    result.patch_evidence["oracle_match"] = oracle_match
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
