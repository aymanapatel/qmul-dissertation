"""Run conservative deterministic repair templates through the Phase 9 sandbox.

This is the non-LLM repair baseline required by Plan_v3.  Templates may use the
finding evidence and saved HTML, but never the hidden repair oracle.  The oracle
is merged only after proposal creation so that exact-match evaluation cannot
leak into the repair decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from .repair.contracts import RepairOperation, RepairProposal
from .repair.policy import proposal_policy_errors
from .repair.validators import validate_repair


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(element: Tag | None) -> str:
    return " ".join(element.get_text(" ", strip=True).split()) if element else ""


def _target(item: dict[str, Any], soup: BeautifulSoup) -> tuple[str, Tag | None]:
    targets = item.get("original_finding", {}).get("evidence", {}).get("target", [])
    selector = str(targets[0]) if targets else ""
    try:
        matches = soup.select(selector) if selector else []
    except Exception:
        matches = []
    return selector, matches[0] if len(matches) == 1 and isinstance(matches[0], Tag) else None


def _hex_rgb(value: str) -> tuple[int, int, int] | None:
    value = value.strip().lower()
    if re.fullmatch(r"#[0-9a-f]{3}", value):
        value = "#" + "".join(character * 2 for character in value[1:])
    if not re.fullmatch(r"#[0-9a-f]{6}", value):
        return None
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def _luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb:
        channel = value / 255
        channels.append(channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    first, second = sorted((_luminance(left), _luminance(right)), reverse=True)
    return (first + 0.05) / (second + 0.05)


def _operation_for(item: dict[str, Any], source_html: str) -> tuple[RepairOperation | None, str]:
    """Return a conservative template operation without consulting repair truth."""
    finding = item.get("original_finding", {})
    rule_id = str(finding.get("rule_id", ""))
    evidence = finding.get("evidence", {})
    summary = str(evidence.get("failure_summary", ""))
    soup = BeautifulSoup(source_html, "lxml")
    selector, element = _target(item, soup)
    if not selector or element is None:
        return None, "The saved selector is not uniquely resolvable."

    if rule_id == "html-has-lang" and element.name == "html":
        languages = {
            "english": "en", "spanish": "es", "french": "fr", "german": "de",
            "italian": "it", "portuguese": "pt", "japanese": "ja",
        }
        lowered = summary.lower()
        for language, code in languages.items():
            if re.search(rf"\b(independently verified|written|content|prose|page)\b[^.]*\b{language}\b", lowered):
                return RepairOperation(operation="set_attribute", selector=selector, attribute_name="lang", css_property=None, new_value=code), "Set the language declared explicitly by independently supplied finding evidence."

    if rule_id == "image-alt" and element.name == "img":
        figure = element.find_parent("figure")
        caption = _text(figure.find("figcaption")) if figure else ""
        if caption:
            return RepairOperation(operation="set_attribute", selector=selector, attribute_name="alt", css_property=None, new_value=caption), "Reuse the visible figure caption as the text alternative."

    if rule_id == "link-name" and element.name == "a":
        path = urlparse(str(element.get("href", ""))).path.rstrip("/")
        slug = path.rsplit("/", 1)[-1] if path else ""
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,60}", slug):
            name = slug.replace("-", " ").replace("_", " ").strip().title()
            return RepairOperation(operation="set_attribute", selector=selector, attribute_name="aria-label", css_property=None, new_value=name), "Derive a bounded link name from its stable local URL slug."

    if rule_id == "button-name" and element.name == "button":
        previous = element.find_previous_sibling()
        context = _text(previous if isinstance(previous, Tag) else None)
        match = re.search(r"\bto\s+([A-Za-z][A-Za-z ]{1,60}?)[.!?]?\s*$", context, re.IGNORECASE)
        if match:
            name = match.group(1).strip().capitalize()
            return RepairOperation(operation="set_attribute", selector=selector, attribute_name="aria-label", css_property=None, new_value=name), "Derive the button name from the immediately preceding explicit purpose statement."

    if rule_id == "label" and element.name in {"input", "select", "textarea"}:
        previous = element.find_previous_sibling()
        context = _text(previous if isinstance(previous, Tag) else None)
        match = re.fullmatch(r"(?:enter|select|choose)\s+(?:your\s+)?(.+?)[.!?]?", context, re.IGNORECASE)
        if match:
            label = match.group(1).strip().capitalize()
            return RepairOperation(operation="insert_label_before", selector=selector, attribute_name=None, css_property=None, new_value=label), "Convert the immediately preceding explicit field instruction into a visible programmatic label."

    if rule_id == "color-contrast":
        style = str(element.get("style", ""))
        background_match = re.search(r"(?:^|;)\s*background-color\s*:\s*(#[0-9A-Fa-f]{3,6})", style)
        background = _hex_rgb(background_match.group(1)) if background_match else None
        if background:
            black = (0, 0, 0); white = (255, 255, 255)
            colour = "#000000" if _contrast(black, background) >= _contrast(white, background) else "#ffffff"
            if max(_contrast(black, background), _contrast(white, background)) >= 4.5:
                return RepairOperation(operation="set_style_property", selector=selector, attribute_name=None, css_property="color", new_value=colour), "Choose black or white deterministically to maximise WCAG contrast against the fixed inline background."

    return None, "No conservative deterministic template could establish a bounded value."


def build_proposal(item: dict[str, Any], source_html: str) -> RepairProposal:
    """Build one deterministic proposal; this function has no oracle argument by design."""
    finding = item.get("original_finding", {})
    operation, rationale = _operation_for(item, source_html)
    query_id = str(item.get("query_id", ""))
    proposal_id = "det-" + hashlib.sha256(f"{query_id}|{operation}".encode()).hexdigest()[:20]
    if operation is None:
        return RepairProposal(
            schema_version=1, proposal_id=proposal_id, query_id=query_id,
            finding_id=str(finding.get("finding_id", "")), decision="requires_human_review",
            operations=[], rationale=rationale,
            expected_resolution="The original finding remains unchanged until a reviewer supplies the contextual value.",
            cited_record_ids=[], uncertainty="Required contextual information was not deterministically available.",
            inspected_visual_elements=[],
            requires_human_review=True, human_review_reasons=["no_safe_deterministic_template"],
            validation_steps=["Review the saved DOM and rendered evidence."], confidence=0.0,
        )
    return RepairProposal(
        schema_version=1, proposal_id=proposal_id, query_id=query_id,
        finding_id=str(finding.get("finding_id", "")), decision="propose",
        operations=[operation], rationale=rationale,
        expected_resolution=f"Resolve {finding.get('rule_id', 'the originating rule')} at the saved target.",
        cited_record_ids=[], uncertainty="The template is limited to explicit local evidence and must pass the complete sandbox gate.",
        inspected_visual_elements=[],
        requires_human_review=False, human_review_reasons=[],
        validation_steps=["Apply the typed operation to an isolated copy.", "Rerun the originating detector and regression suite."],
        confidence=0.9,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = json.loads(args.generator_inputs.read_text(encoding="utf-8"))
    selected = [item for item in inputs if item.get("condition") == args.source_condition][:args.max_proposals]
    if not selected:
        raise ValueError(f"No generator inputs found for source condition {args.source_condition!r}")
    truth = {}
    if args.repair_truth:
        payload = json.loads(args.repair_truth.read_text(encoding="utf-8"))
        truth = {str(item["query_id"]): item for item in payload.get("cases", [])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for index, item in enumerate(selected):
        attempt_started = time.perf_counter()
        finding = item.get("original_finding", {})
        site_id = str(finding.get("site_id", ""))
        source = args.corpus_dir / site_id / "0.html"
        attempt: dict[str, Any] = {
            "query_id": item.get("query_id"), "site_id": site_id,
            "finding_id": finding.get("finding_id"), "condition": "deterministic_template",
        }
        if not source.is_file():
            attempt.update(status="generation_or_input_failed", error_category="missing_source", error=f"Missing source snapshot: {source}")
        else:
            proposal = build_proposal(item, source.read_text(encoding="utf-8", errors="replace"))
            proposal_path = args.output_dir / "proposals" / f"{index:04d}-{proposal.proposal_id}.json"
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_path.write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
            attempt["generation"] = {
                "response_id": None, "model": "deterministic-template-v1",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
                "refusal": None, "proposal_path": str(proposal_path),
            }
            policy_item = {**item, "condition": "deterministic_template", "citations": []}
            policy_errors = proposal_policy_errors(proposal, policy_item)
            if policy_errors:
                attempt.update(status="rejected", policy_errors=policy_errors)
            elif proposal.decision != "propose":
                attempt.update(status="requires_human_review")
            else:
                validation_finding = dict(finding)
                hidden = truth.get(str(item.get("query_id")))
                if hidden:
                    validation_finding.update({key: hidden[key] for key in ("status", "semantic_verified", "visual_verified", "oracle_operations") if key in hidden})
                validation = validate_repair(
                    source_path=source, proposal=proposal, original_finding=validation_finding,
                    output_dir=args.output_dir / "attempts", axe_js=args.axe_js,
                    run_browser=not args.skip_browser,
                )
                attempt.update(status=validation.outcome, validation=validation.model_dump(mode="json"))
        attempt["duration_seconds"] = time.perf_counter() - attempt_started
        attempts.append(attempt)
    counts = Counter(str(item["status"]) for item in attempts)
    report = {
        "schema_version": 1, "phase": 9, "status": "deterministic_repair_validation_run",
        "model": "deterministic-template-v1", "api": None, "api_mode": None, "endpoint": None,
        "structured_output_contract": "accessibility_system.repair.contracts.RepairProposal",
        "condition": "deterministic_template", "attempt_count": len(attempts),
        "run_duration_seconds": time.perf_counter() - started,
        "outcome_counts": dict(sorted(counts.items())), "attempts": attempts,
        "oracle_separation": "Hidden repair truth is supplied only after deterministic proposal creation.",
        "acceptance_policy": {"target_must_be_resolved": True, "new_in_scope_regressions_allowed": 0},
    }
    report_path = args.output_dir / "phase_9_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1, "phase": 9, "condition": "deterministic_template",
        "inputs": {
            "generator_inputs": {"path": str(args.generator_inputs), "sha256": _sha256(args.generator_inputs)},
            "repair_truth": {"path": str(args.repair_truth), "sha256": _sha256(args.repair_truth)} if args.repair_truth else None,
            "axe_js": {"path": str(args.axe_js), "sha256": _sha256(args.axe_js)},
        },
        "config": {"source_condition": args.source_condition, "max_proposals": args.max_proposals, "skip_browser": args.skip_browser},
        "outputs": {"phase_9_report.json": _sha256(report_path)},
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-inputs", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--axe-js", type=Path, required=True)
    parser.add_argument("--repair-truth", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-condition", choices=("no_rag", "flat_vector_rag", "graph_constrained_rag"), default="no_rag")
    parser.add_argument("--max-proposals", type=int, default=100)
    parser.add_argument("--skip-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "attempts": report["attempt_count"], "outcomes": report["outcome_counts"]}, indent=2))


if __name__ == "__main__":
    main()
