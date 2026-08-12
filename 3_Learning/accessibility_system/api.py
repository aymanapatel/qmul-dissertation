"""HTTP API for accessibility scans, RAG inspection, and Phase 9 repair runs."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import socket
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .phase9 import _configure_logging, run as run_phase9
from .repair.generator import OpenAIRepairGenerator, SYSTEM_PROMPT, build_prompt_messages
from learning_v2.live_inference import (
    _visual_evidence_payload,
    capture_aligned_page,
    run_live_specialists,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "learning_v2/artifacts_3107_0015/phase_8/generator_inputs.json"
DEFAULT_CORPUS = ROOT.parent / "2_Data/browser-use/outputs/axe-core"
DEFAULT_AXE = ROOT.parent / "2_Data/browser-use/axe-core.min.js"
DEFAULT_RUNS = ROOT / "learning_v2/artifacts_3107_0015/api_runs"
LOGGER = logging.getLogger("accessibility_system.api")


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScanRequest(StrictRequest):
    urls: list[HttpUrl] = Field(min_length=1, max_length=20)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)


class RepairRunRequest(StrictRequest):
    condition: Literal["no_rag", "flat_vector_rag", "graph_constrained_rag"] = "graph_constrained_rag"
    query_ids: list[str] = Field(default_factory=list, max_length=100)
    max_proposals: int = Field(default=10, ge=1, le=100)
    model: str | None = Field(default=None, max_length=200)
    api_mode: Literal["responses", "chat_completions"] = "chat_completions"
    base_url: str | None = Field(default=None, max_length=500)
    skip_browser: bool = False
    generation_retries: int = Field(default=1, ge=0, le=3)


class SuggestionAuditRequest(StrictRequest):
    """One public page to scan and send to the structured LLM adapter."""

    url: HttpUrl
    max_suggestions: int = Field(default=5, ge=1, le=8)
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_public_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute HTTP(S) URLs are supported")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not supported")
    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or default_port)}
    except socket.gaierror as exc:
        raise ValueError(f"Host could not be resolved: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(f"Private, loopback, and reserved targets are blocked: {parsed.hostname}")


def _normalise_violations(report: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for violation in report.get("violations", []):
        output.append({
            "id": violation.get("id"),
            "impact": violation.get("impact"),
            "description": violation.get("description"),
            "help": violation.get("help"),
            "help_url": violation.get("helpUrl"),
            "tags": violation.get("tags", []),
            "nodes": [{
                "target": node.get("target", []),
                "html": str(node.get("html", ""))[:3000],
                "failure_summary": str(node.get("failureSummary", ""))[:3000],
            } for node in violation.get("nodes", [])],
        })
    return output


def scan_sites(
    urls: list[str], axe_js: Path, timeout_seconds: float,
    screenshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Scan public URLs with axe-core in one isolated Chromium session."""
    from playwright.sync_api import sync_playwright

    if not axe_js.is_file():
        raise FileNotFoundError(f"axe-core bundle not found: {axe_js}")
    axe_source = axe_js.read_text(encoding="utf-8")
    sites: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 720}, reduced_motion="reduce")

        def guard_request(route) -> None:
            try:
                _safe_public_url(route.request.url)
            except ValueError:
                route.abort("blockedbyclient")
            else:
                route.continue_()

        context.route("**/*", guard_request)
        for url in urls:
            started = _now()
            try:
                _safe_public_url(url)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=int(timeout_seconds * 1000))
                page.add_script_tag(content=axe_source)
                report = page.evaluate("""async () => await axe.run(document, {
                  resultTypes: ['violations'], rules: {'region': {enabled: false}}
                })""")
                violations = _normalise_violations(report)
                screenshot_artifact = None
                if screenshot_dir is not None:
                    evidence_dir = screenshot_dir / "live_page"
                    capture_aligned_page(page, evidence_dir, report)
                    screenshot_artifact = "live_page/page.png"
                sites.append({
                    "url": url, "final_url": page.url, "status": "completed", "started_at": started,
                    "completed_at": _now(), "violation_count": len(violations),
                    "affected_node_count": sum(len(item["nodes"]) for item in violations),
                    "violations_by_impact": dict(Counter(str(item["impact"] or "unknown") for item in violations)),
                    "violations": violations, "screenshot_artifact": screenshot_artifact,
                })
            except Exception as exc:
                sites.append({
                    "url": url, "status": "failed", "started_at": started,
                    "completed_at": _now(), "error": f"{type(exc).__name__}: {exc}"[:2000],
                })
            finally:
                if "page" in locals():
                    page.close()
                    del page
        browser.close()
    return {
        "schema_version": 1, "status": "completed", "site_count": len(sites),
        "completed_count": sum(item["status"] == "completed" for item in sites), "sites": sites,
    }


IMPACT_ORDER = {"critical": 0, "serious": 1, "moderate": 2, "minor": 3, None: 4}


def _suggestion_inputs(site: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Create bounded LLM inputs from the highest-impact axe node findings."""

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for violation in site.get("violations", []):
        for node in violation.get("nodes", []):
            candidates.append((violation, node))
    candidates.sort(key=lambda item: IMPACT_ORDER.get(item[0].get("impact"), 4))
    hostname = urlparse(str(site.get("final_url") or site.get("url") or "")).hostname or "unknown"
    values = []
    for index, (violation, node) in enumerate(candidates[:limit], 1):
        rule_id = str(violation.get("id") or "unknown-rule")
        target = [str(item) for item in node.get("target", []) if str(item).strip()]
        identity = f"{hostname}:{rule_id}:{index}:{'|'.join(target)}"
        finding_id = "live-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        query_id = f"suggest-{finding_id}"
        values.append({
            "query_id": query_id,
            "condition": "no_rag",
            "safe_action": "suggest_only_do_not_apply",
            "prompt": (
                "Give one bounded, actionable accessibility remediation suggestion for this observed "
                "finding. Do not claim the suggestion has been applied or validated. If the correct "
                "semantic value cannot be known from the evidence, require human review."
            ),
            "citations": [],
            "original_finding": {
                "finding_id": finding_id,
                "site_id": hostname,
                "rule_id": rule_id,
                "impact": violation.get("impact"),
                "help": violation.get("help"),
                "help_url": violation.get("help_url"),
                "evidence": {
                    "target": target,
                    "html": str(node.get("html", ""))[:3000],
                    "failure_summary": str(node.get("failure_summary", ""))[:3000],
                },
            },
        })
    return values


def _specialist_suggestion_inputs(
    site: dict[str, Any], specialist_report: dict[str, Any], limit: int,
) -> list[dict[str, Any]]:
    """Convert specialist and same-session visual findings into bounded LLM inputs."""

    axe_by_rule = {str(item.get("id")): item for item in site.get("violations", [])}
    hostname = urlparse(str(site.get("final_url") or site.get("url") or "")).hostname or "unknown"
    values = []
    visual_report = specialist_report.get("visual_evidence", {})
    visual_source = str(visual_report.get("source") or "same-session-rendered-visual-capture")
    visual_elements = [
        item for item in visual_report.get("elements", [])
        if isinstance(item, dict)
    ] if isinstance(visual_report, dict) else []
    visual_by_selector = {
        str(item.get("selector")): item for item in visual_elements
        if str(item.get("selector", ""))
    }

    contrast_failures = [
        item for item in visual_report.get("contrast_failures", [])
        if isinstance(item, dict) and str(item.get("selector", ""))
    ] if isinstance(visual_report, dict) else []
    contrast_candidates = []
    for item in contrast_failures:
        context = item.get("repair_context", {})
        contrast_candidates.extend(
            candidate for candidate in context.get("bounded_candidates", [])
            if isinstance(candidate, dict)
        )
    if contrast_failures and contrast_candidates and limit > 0:
        selectors = [str(item["selector"]) for item in contrast_failures]
        identity = f"{hostname}:color-contrast:1.4.3:{'|'.join(selectors)}:measured-rendered"
        finding_id = "visual-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        axe = axe_by_rule.get("color-contrast", {})
        measurements = ", ".join(
            f"{item['selector']}={float(item.get('visual', {}).get('contrast_ratio', 0)):.3f}:1"
            f" (required {float(item.get('visual', {}).get('required_contrast_ratio', 0)):.1f}:1)"
            for item in contrast_failures
        )
        measured_evidence = {
            "evidence_kind": "measured_visual",
            "graph_view": "rendered-visual",
            "architecture": "browser-measurement",
            "detector_id": "axe-core+same-session-rendered-geometry:color-contrast",
            "probability": None,
            "threshold": None,
            "routing_status": "fail",
            "routing_confidence": 1.0,
            "axe_used_for_prediction": False,
            "evidence": {
                "selector": selectors[0],
                "selectors": selectors,
                "visual": contrast_failures[0].get("visual", {}),
            },
        }
        values.append({
            "query_id": f"suggest-{finding_id}",
            "condition": "no_rag",
            "safe_action": "suggest_only_do_not_apply",
            "prompt": (
                "Give one bounded, actionable remediation proposal for all supplied color-contrast "
                "targets. Inspect visual_tool_result and copy every visual observation into "
                "inspected_visual_elements. Copy every computed bounded candidate operation exactly, "
                "so each evidenced failing selector receives a colour fix. These are measured browser "
                "and axe findings, not trained-model predictions. Do not claim that a suggestion has "
                "been applied or validated."
            ),
            "citations": [],
            "original_finding": {
                "finding_id": finding_id,
                "site_id": hostname,
                "rule_id": "color-contrast",
                "criterion_id": "1.4.3",
                "impact": axe.get("impact"),
                "help": axe.get("help") or "Elements must meet minimum color contrast ratio thresholds",
                "help_url": axe.get("help_url"),
                "detector": measured_evidence["detector_id"],
                "confidence": 1.0,
                "routing_status": "fail",
                "evidence": {
                    "target": selectors,
                    "html": "",
                    "text": " | ".join(str(item.get("text", "")) for item in contrast_failures),
                    "tag": "multiple",
                    "attributes": {},
                    "visual": contrast_failures[0].get("visual", {}),
                    "visual_source": visual_source,
                    "visual_elements": contrast_failures,
                    "repair_context": {
                        "rule_id": "color-contrast",
                        "target": None,
                        "targets": [item.get("repair_context", {}).get("target") for item in contrast_failures],
                        "accessible_name_signals": {},
                        "nearby": {},
                        "current_state": {
                            "meets_requirement": False,
                            "measurements": measurements,
                        },
                        "bounded_candidates": contrast_candidates,
                        "candidate_policy": "Copy every computed candidate operation exactly.",
                    },
                    "failure_summary": (
                        f"Same-session rendered measurements and axe-core identify {len(contrast_failures)} "
                        f"color-contrast failures: {measurements}. The exact replacement colours in the "
                        "bounded candidates were computed against the captured backgrounds. This is "
                        "measured evidence, not a trained-model prediction."
                    ),
                    "graph_view": "rendered-visual",
                    "architecture": "browser-measurement",
                    "axe_used_for_prediction": False,
                },
            },
            "model_evidence": measured_evidence,
        })

    raw_findings = [
        item for item in specialist_report.get("findings", []) if isinstance(item, dict)
    ]
    primary_findings: list[dict[str, Any]] = []
    repeated_findings: list[dict[str, Any]] = []
    primary_keys: set[tuple[str, str]] = set()
    for finding in raw_findings:
        evidence = finding.get("evidence", {})
        key = (str(finding.get("rule_id", "")), str(evidence.get("selector") or "html"))
        if key in primary_keys:
            repeated_findings.append(finding)
        else:
            primary_keys.add(key)
            primary_findings.append(finding)

    seen: set[tuple[str, str, str, str]] = set()
    for finding in [*primary_findings, *repeated_findings]:
        if len(values) >= limit:
            break
        rule_id = str(finding["rule_id"])
        evidence = finding.get("evidence", {})
        selector = str(evidence.get("selector") or "html")
        issue_key = (
            rule_id, selector,
            str(finding.get("graph_view", "")),
            str(finding.get("architecture", "")),
        )
        if issue_key in seen:
            continue
        seen.add(issue_key)
        identity = (
            f"{hostname}:{rule_id}:{finding.get('criterion_id')}:{selector}:"
            f"{finding.get('graph_view')}:{finding.get('architecture')}"
        )
        finding_id = "gnn-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        axe = axe_by_rule.get(rule_id, {})
        probability = float(finding["probability"])
        threshold = float(finding["threshold"])
        values.append({
            "query_id": f"suggest-{finding_id}",
            "condition": "no_rag",
            "safe_action": "suggest_only_do_not_apply",
            "prompt": (
                "Give one bounded, actionable remediation suggestion for this accessibility finding "
                "identified by the trained graph specialist. Use the supplied selector, model evidence, "
                "nearby page context, accessible-name signals, and provenance-labelled bounded repair "
                "candidates. Copy a candidate operation exactly rather than inventing a value. A computed "
                "candidate may be proposed directly; a contextual semantic candidate may be proposed only "
                "with requires_human_review=true. For image-alt, use author-supplied semantic context such as "
                "a caption, referenced description, adjacent descriptive text, accessible-name text on the "
                "previous or next sibling, local-container prose, matching existing alt, or named parent link. "
                "Inspect repair_context.nearby explicitly. If no exact candidate exists, synthesise a concise "
                "purpose-oriented alt from the nearest heading, local ancestors, siblings, and image position; "
                "do not invent unseen visual details. For image-alt return a proposal without human review. "
                "Never turn a filename, path, asset ID, or hash into alt text. For other rules, if no bounded "
                "candidate is supported, require human review with no operations. Do not claim the suggestion "
                "has been applied or validated."
            ),
            "citations": [],
            "original_finding": {
                "finding_id": finding_id,
                "site_id": hostname,
                "rule_id": rule_id,
                "criterion_id": finding.get("criterion_id"),
                "impact": axe.get("impact"),
                "help": axe.get("help") or rule_id.replace("-", " ").title(),
                "help_url": axe.get("help_url"),
                "detector": finding["detector_id"],
                "confidence": probability,
                "routing_status": finding["routing_status"],
                "evidence": {
                    "target": [selector],
                    "html": str(evidence.get("html", ""))[:3000],
                    "text": str(evidence.get("text", ""))[:500],
                    "tag": str(evidence.get("tag", ""))[:100],
                    "attributes": evidence.get("attributes", {}),
                    "visual": evidence.get("visual"),
                    "visual_source": visual_source,
                    "visual_elements": (
                        [visual_by_selector[selector]] if selector in visual_by_selector else []
                    ),
                    "repair_context": evidence.get("repair_context", {}),
                    "failure_summary": (
                        f"The frozen {finding['architecture']} {finding['graph_view']} specialist "
                        f"predicted {rule_id} with probability {probability:.6f}, above its "
                        f"validation-frozen threshold {threshold:.6f}. The routed status is "
                        f"{finding['routing_status']}."
                    ),
                    "model_probability": probability,
                    "model_threshold": threshold,
                    "graph_view": finding["graph_view"],
                    "architecture": finding["architecture"],
                    "axe_used_for_prediction": False,
                },
            },
            "model_evidence": finding,
        })
        if len(values) >= limit:
            break
    return values


def generate_live_suggestions(
    scan: dict[str, Any], *, max_suggestions: int,
    generator_factory=OpenAIRepairGenerator,
    specialist_report: dict[str, Any] | None = None,
    progress=None,
) -> dict[str, Any]:
    """Call the configured LLM once per selected finding and preserve failures."""

    site = scan["sites"][0]
    if site.get("status") != "completed":
        return {"status": "scan_failed", "site": site, "suggestions": []}
    generator = generator_factory()
    suggestions = []
    generator_inputs = (
        _specialist_suggestion_inputs(site, specialist_report, max_suggestions)
        if specialist_report is not None
        else _suggestion_inputs(site, max_suggestions)
    )
    for index, generator_input in enumerate(generator_inputs, 1):
        finding = generator_input["original_finding"]
        event_id = f"call_llm_{index:02d}"
        label = (
            f"Call LLM for {finding.get('rule_id')} · "
            f"{finding.get('evidence', {}).get('architecture', 'detector')}"
        )
        if progress is not None:
            progress(event_id, "running", label, {
                "finding_id": finding["finding_id"], "rule_id": finding.get("rule_id"),
                "architecture": finding.get("evidence", {}).get("architecture"),
                "graph_view": finding.get("evidence", {}).get("graph_view"),
                "index": index, "total": len(generator_inputs),
            })
        _, user_payload = build_prompt_messages(generator_input)
        fallback_trace = {
            "method": "POST",
            "endpoint": str(getattr(generator, "endpoint", "configured OpenAI-compatible endpoint")),
            "api_mode": str(getattr(generator, "api_mode", "structured_output")),
            "model": str(getattr(generator, "model", "configured model")),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_payload,
            "response_format": "RepairProposal (strict structured output)",
        }
        try:
            generated = generator.generate(generator_input)
            proposal = generated.proposal.model_dump(mode="json")
            request_trace = dict(getattr(generated, "request_trace", {}) or fallback_trace)
            suggestions.append({
                "finding_id": finding["finding_id"],
                "rule_id": finding["rule_id"],
                "impact": finding.get("impact"),
                "target": finding["evidence"]["target"],
                "help": finding.get("help"),
                "decision": proposal["decision"],
                "rationale": proposal["rationale"],
                "expected_resolution": proposal["expected_resolution"],
                "operations": proposal["operations"],
                "inspected_visual_elements": proposal["inspected_visual_elements"],
                "confidence": proposal["confidence"],
                "requires_human_review": proposal["requires_human_review"],
                "human_review_reasons": proposal["human_review_reasons"],
                "validation_steps": proposal["validation_steps"],
                "model_evidence": generator_input.get("model_evidence"),
                "model": generated.model,
                "response_id": generated.response_id,
                "usage": generated.usage,
                "api_trace": {
                    "request": request_trace,
                    "response": {
                        "response_id": generated.response_id,
                        "model": generated.model,
                        "usage": generated.usage,
                        "structured_output": proposal,
                    },
                },
                "generation_status": "completed",
            })
            if progress is not None:
                progress(event_id, "completed", label, {
                    "finding_id": finding["finding_id"], "rule_id": finding.get("rule_id"),
                    "architecture": finding.get("evidence", {}).get("architecture"),
                    "graph_view": finding.get("evidence", {}).get("graph_view"),
                    "model": generated.model, "response_id": generated.response_id,
                    "index": index, "total": len(generator_inputs),
                })
        except Exception as exc:
            LOGGER.exception("suggestion_generation_failed finding_id=%s", finding["finding_id"])
            safe_error = f"{type(exc).__name__}: {exc}"[:1000]
            suggestions.append({
                "finding_id": finding["finding_id"], "rule_id": finding["rule_id"],
                "impact": finding.get("impact"), "target": finding["evidence"]["target"],
                "help": finding.get("help"), "generation_status": "failed",
                "model_evidence": generator_input.get("model_evidence"),
                "api_trace": {"request": fallback_trace, "response": {"error": safe_error}},
                "error": safe_error,
            })
            if progress is not None:
                progress(event_id, "failed", label, {
                    "finding_id": finding["finding_id"], "rule_id": finding.get("rule_id"),
                    "architecture": finding.get("evidence", {}).get("architecture"),
                    "graph_view": finding.get("evidence", {}).get("graph_view"),
                    "error": safe_error, "index": index, "total": len(generator_inputs),
                })
    return {
        "schema_version": 2,
        "status": "completed" if all(item["generation_status"] == "completed" for item in suggestions) else "partial",
        "source_url": site.get("url"),
        "final_url": site.get("final_url"),
        "screenshot_artifact": site.get("screenshot_artifact"),
        "violation_count": site.get("violation_count", 0),
        "affected_node_count": site.get("affected_node_count", 0),
        "violations_by_impact": site.get("violations_by_impact", {}),
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
        "visual_evidence": (
            specialist_report.get("visual_evidence") if specialist_report is not None else None
        ),
        "specialist": {
            "architectures": specialist_report.get("architectures", []),
            "training_artifacts": specialist_report.get("training_artifacts"),
            "fusion_policy": specialist_report.get("fusion_policy"),
            "model_runs": specialist_report.get("model_runs", []),
            "finding_count": len(specialist_report.get("findings", [])),
        } if specialist_report is not None else None,
        "safety": "Suggestions only. No source HTML was changed and no operation was applied.",
    }


class JobStore:
    def __init__(self, root: Path, workers: int = 2) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="a11y-api")
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}

    def submit(self, kind: str, function, *args) -> dict[str, Any]:
        job_id = f"{kind}-{uuid.uuid4().hex[:16]}"
        run_dir = self.root / job_id
        job = {
            "job_id": job_id, "kind": kind, "status": "queued", "created_at": _now(),
            "run_dir": str(run_dir),
            "progress": {"current_stage": "queued", "label": "Queued", "events": []},
        }
        self._write(job)
        self.pool.submit(self._execute, job_id, function, run_dir, *args)
        return self.public(job_id)

    def _write(self, job: dict[str, Any]) -> None:
        with self.lock:
            self.jobs[job["job_id"]] = job
            _atomic_json(Path(job["run_dir"]) / "job.json", job)

    def _execute(self, job_id: str, function, run_dir: Path, *args) -> None:
        job = json.loads(json.dumps(self.jobs[job_id]))
        job.update(status="running", started_at=_now())
        self._write(job)
        try:
            result = function(run_dir, self.reporter(job_id), *args)
            _atomic_json(run_dir / "result.json", result)
            job = json.loads(json.dumps(self.jobs[job_id]))
            job.update(status="completed", completed_at=_now(), result_path=str(run_dir / "result.json"))
        except Exception as exc:
            LOGGER.exception("job_failed job_id=%s", job_id)
            job = json.loads(json.dumps(self.jobs[job_id]))
            safe_error = f"{type(exc).__name__}: {exc}"[:4000]
            running = next(
                (item for item in reversed(job.get("progress", {}).get("events", [])) if item.get("status") == "running"),
                None,
            )
            if running is not None:
                self.reporter(job_id)(
                    str(running["event_id"]), "failed", str(running["label"]),
                    {**dict(running.get("details", {})), "error": safe_error},
                )
                job = json.loads(json.dumps(self.jobs[job_id]))
            job.update(status="failed", completed_at=_now(), error=safe_error)
        self._write(job)

    def reporter(self, job_id: str):
        """Return a thread-safe stage reporter for one background job."""

        def report(event_id: str, status: str, label: str, details: dict[str, Any] | None = None) -> None:
            if status not in {"running", "completed", "failed", "skipped"}:
                raise ValueError(f"Unsupported progress status: {status}")
            job = json.loads(json.dumps(self.jobs[job_id]))
            progress = job.setdefault("progress", {"events": []})
            events = progress.setdefault("events", [])
            now = _now()
            event = next((item for item in events if item["event_id"] == event_id), None)
            if event is None:
                event = {
                    "event_id": event_id, "label": label,
                    "status": status, "started_at": now,
                }
                events.append(event)
            event.update(label=label, status=status, details=details or {})
            if status in {"completed", "failed", "skipped"}:
                event["completed_at"] = now
                started = datetime.fromisoformat(event["started_at"])
                completed = datetime.fromisoformat(now)
                event["duration_ms"] = max(0, round((completed - started).total_seconds() * 1000))
            progress.update(
                current_stage=event_id,
                label=label,
                completed=sum(item["status"] in {"completed", "failed", "skipped"} for item in events),
                total=len(events),
            )
            self._write(job)

        return report

    def public(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            path = self.root / job_id / "job.json"
            if not path.is_file():
                raise KeyError(job_id)
            job = json.loads(path.read_text(encoding="utf-8"))
        public = {
            key: value for key, value in job.items()
            if key not in {"run_dir", "result_path"}
        }
        public["links"] = {
            "self": f"/v1/jobs/{job_id}",
            "result": f"/v1/jobs/{job_id}/result",
            "artifacts": f"/v1/jobs/{job_id}/artifacts/{{artifact_path}}",
        }
        return public

    def list(self) -> list[dict[str, Any]]:
        discovered = {path.parent.name for path in self.root.glob("*/job.json")}
        discovered.update(self.jobs)
        values = [self.public(item) for item in discovered]
        return sorted(values, key=lambda item: item["created_at"], reverse=True)


def create_app(
    *, generator_inputs: Path = DEFAULT_INPUTS, corpus_dir: Path = DEFAULT_CORPUS,
    axe_js: Path = DEFAULT_AXE, runs_dir: Path = DEFAULT_RUNS,
    suggestion_scanner=scan_sites, suggestion_generator_factory=OpenAIRepairGenerator,
    suggestion_specialist_runner=run_live_specialists,
) -> FastAPI:
    app = FastAPI(
        title="Accessibility Research API", version="1.0.0",
        description="Run axe scans, inspect Phase 8 RAG evidence, and execute structured Phase 9 repairs.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    store = JobStore(runs_dir)
    app.state.generator_inputs = generator_inputs.resolve()
    app.state.corpus_dir = corpus_dir.resolve()
    app.state.axe_js = axe_js.resolve()
    app.state.store = store

    def load_inputs() -> list[dict[str, Any]]:
        path = app.state.generator_inputs
        if not path.is_file():
            raise HTTPException(status_code=503, detail=f"Generator inputs not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok", "generator_inputs_available": app.state.generator_inputs.is_file(),
            "corpus_available": app.state.corpus_dir.is_dir(), "axe_available": app.state.axe_js.is_file(),
        }

    @app.post("/v1/scans", status_code=202)
    def create_scan(request: ScanRequest) -> dict[str, Any]:
        urls = [str(url) for url in request.urls]
        for url in urls:
            try:
                _safe_public_url(url)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        def task(run_dir: Path, progress, values: list[str], timeout: float) -> dict[str, Any]:
            progress("capture_page", "running", "Capture and scan pages", {"site_count": len(values)})
            result = scan_sites(values, app.state.axe_js, timeout)
            progress("capture_page", "completed", "Capture and scan pages", {
                "site_count": len(values), "completed_count": result.get("completed_count", 0),
            })
            result["run_id"] = run_dir.name
            return result

        return store.submit("scan", task, urls, request.timeout_seconds)

    @app.post("/v1/suggestion-audits", status_code=202)
    def create_suggestion_audit(request: SuggestionAuditRequest) -> dict[str, Any]:
        """Scan exactly one public URL and request structured LLM suggestions."""

        url = str(request.url)
        try:
            _safe_public_url(url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        def task(run_dir: Path, progress, source_url: str, maximum: int, timeout: float) -> dict[str, Any]:
            progress("capture_page", "running", "Capture page and aligned evidence", {"url": source_url})
            scan = suggestion_scanner([source_url], app.state.axe_js, timeout, run_dir)
            specialist_report = None
            if scan.get("sites") and scan["sites"][0].get("status") == "completed":
                site = scan["sites"][0]
                progress("capture_page", "completed", "Capture page and aligned evidence", {
                    "source_url": source_url, "final_url": site.get("final_url"),
                    "violation_count": site.get("violation_count", 0),
                    "affected_node_count": site.get("affected_node_count", 0),
                    "screenshot_artifact": site.get("screenshot_artifact"),
                })
                specialist_report = suggestion_specialist_runner(
                    run_dir / "live_page", run_dir / "learning_v2_graphs",
                    progress=progress,
                )
            else:
                error = str(scan.get("sites", [{}])[0].get("error", "Page capture failed"))
                progress("capture_page", "failed", "Capture page and aligned evidence", {"error": error})
            result = generate_live_suggestions(
                scan, max_suggestions=maximum,
                generator_factory=suggestion_generator_factory,
                specialist_report=specialist_report,
                progress=progress,
            )
            progress("finalise_result", "running", "Finalise audit result", {
                "suggestion_count": result.get("suggestion_count", 0),
            })
            result["run_id"] = run_dir.name
            if result.get("screenshot_artifact"):
                result["screenshot_url"] = (
                    f"/v1/jobs/{run_dir.name}/artifacts/{result['screenshot_artifact']}"
                )
            result["application_api"] = {
                "submit": {
                    "method": "POST", "endpoint": "/v1/suggestion-audits",
                    "request_body": {
                        "url": source_url, "max_suggestions": maximum,
                        "timeout_seconds": timeout,
                    },
                    "response_body": {
                        "job_id": run_dir.name, "kind": "suggest", "status": "queued",
                    },
                },
                "poll": {
                    "method": "GET", "endpoint": f"/v1/jobs/{run_dir.name}",
                    "response": "Live job state and progress events",
                },
                "result": {
                    "method": "GET", "endpoint": f"/v1/jobs/{run_dir.name}/result",
                    "response": "Schema-v2 structured suggestion audit",
                },
            }
            progress("finalise_result", "completed", "Finalise audit result", {
                "status": result.get("status"), "suggestion_count": result.get("suggestion_count", 0),
            })
            return result

        return store.submit(
            "suggest", task, url, request.max_suggestions, request.timeout_seconds,
        )

    @app.get("/v1/rag/inputs")
    def rag_inputs(
        condition: str | None = None, site_id: str | None = None,
        rule_id: str | None = None, limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        values = load_inputs()
        if condition:
            values = [item for item in values if item.get("condition") == condition]
        if site_id:
            values = [item for item in values if item.get("original_finding", {}).get("site_id") == site_id]
        if rule_id:
            values = [item for item in values if item.get("original_finding", {}).get("rule_id") == rule_id]
        return {"count": min(len(values), limit), "total_matching": len(values), "items": values[:limit]}

    @app.get("/v1/rag/inputs/{query_id}")
    def rag_input(query_id: str) -> dict[str, Any]:
        values = [item for item in load_inputs() if item.get("query_id") == query_id]
        if not values:
            raise HTTPException(status_code=404, detail="Query ID not found")
        return {"query_id": query_id, "conditions": values}

    @app.post("/v1/repairs", status_code=202)
    def create_repair_run(request: RepairRunRequest) -> dict[str, Any]:
        values = [item for item in load_inputs() if item.get("condition") == request.condition]
        if request.query_ids:
            requested = set(request.query_ids)
            values = [item for item in values if item.get("query_id") in requested]
            missing = sorted(requested - {str(item.get("query_id")) for item in values})
            if missing:
                raise HTTPException(status_code=404, detail={"missing_query_ids": missing})
        values = values[:request.max_proposals]
        if not values:
            raise HTTPException(status_code=422, detail="No matching generator inputs")

        def task(run_dir: Path, progress, selected: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
            progress("run_repair_study", "running", "Run structured repair study", {
                "proposal_count": len(selected), "condition": config["condition"],
            })
            inputs_path = run_dir / "generator_inputs.json"
            _atomic_json(inputs_path, selected)
            output_dir = run_dir / "phase_9"
            args = SimpleNamespace(
                generator_inputs=inputs_path, corpus_dir=app.state.corpus_dir, axe_js=app.state.axe_js,
                output_dir=output_dir, condition=config["condition"], model=config["model"],
                api_mode=config["api_mode"], base_url=config["base_url"],
                max_proposals=len(selected), skip_browser=config["skip_browser"],
                generation_retries=config["generation_retries"], repair_truth=None,
            )
            _configure_logging(output_dir, "INFO")
            result = run_phase9(args)
            progress("run_repair_study", "completed", "Run structured repair study", {
                "proposal_count": len(selected), "condition": config["condition"],
            })
            return result

        return store.submit("repair", task, values, request.model_dump())

    @app.get("/v1/jobs")
    def list_jobs() -> dict[str, Any]:
        jobs = store.list()
        return {"count": len(jobs), "jobs": jobs}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return store.public(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/v1/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> dict[str, Any]:
        try:
            job = store.public(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail={"status": job["status"], "error": job.get("error")})
        run_dir = store.root / job_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        live_page = run_dir / "live_page"
        if result.get("visual_evidence") is None and all(
            (live_page / name).is_file()
            for name in ("0.html", "0.visual.json", "page-0_home.json")
        ):
            result["visual_evidence"] = _visual_evidence_payload(live_page)
        return result

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
    def get_artifact(job_id: str, artifact_path: str):
        try:
            store.public(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        run_dir = (store.root / job_id).resolve()
        target = (run_dir / artifact_path).resolve()
        if not target.is_relative_to(run_dir) or not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(target)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("accessibility_system.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
