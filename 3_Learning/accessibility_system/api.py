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
from .repair.generator import OpenAIRepairGenerator
from learning_v2.live_inference import capture_aligned_page, run_live_specialists


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
    """Convert trained-model findings into bounded LLM generator inputs."""

    axe_by_rule = {str(item.get("id")): item for item in site.get("violations", [])}
    hostname = urlparse(str(site.get("final_url") or site.get("url") or "")).hostname or "unknown"
    values = []
    seen: set[tuple[str, str]] = set()
    for finding in specialist_report.get("findings", []):
        rule_id = str(finding["rule_id"])
        evidence = finding.get("evidence", {})
        selector = str(evidence.get("selector") or "html")
        issue_key = (rule_id, selector)
        if issue_key in seen:
            continue
        seen.add(issue_key)
        identity = f"{hostname}:{rule_id}:{finding.get('criterion_id')}:{selector}:{len(values) + 1}"
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
                "identified by the trained graph specialist. Use the supplied selector and model "
                "evidence. Do not claim the suggestion has been applied or validated. If the correct "
                "semantic value cannot be known from the evidence, require human review."
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
                    "visual": evidence.get("visual"),
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
    for generator_input in generator_inputs:
        finding = generator_input["original_finding"]
        try:
            generated = generator.generate(generator_input)
            proposal = generated.proposal.model_dump(mode="json")
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
                "confidence": proposal["confidence"],
                "requires_human_review": proposal["requires_human_review"],
                "human_review_reasons": proposal["human_review_reasons"],
                "validation_steps": proposal["validation_steps"],
                "model_evidence": generator_input.get("model_evidence"),
                "model": generated.model,
                "response_id": generated.response_id,
                "usage": generated.usage,
                "generation_status": "completed",
            })
        except Exception as exc:
            LOGGER.exception("suggestion_generation_failed finding_id=%s", finding["finding_id"])
            suggestions.append({
                "finding_id": finding["finding_id"], "rule_id": finding["rule_id"],
                "impact": finding.get("impact"), "target": finding["evidence"]["target"],
                "help": finding.get("help"), "generation_status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            })
    return {
        "schema_version": 1,
        "status": "completed" if all(item["generation_status"] == "completed" for item in suggestions) else "partial",
        "source_url": site.get("url"),
        "final_url": site.get("final_url"),
        "screenshot_artifact": site.get("screenshot_artifact"),
        "violation_count": site.get("violation_count", 0),
        "affected_node_count": site.get("affected_node_count", 0),
        "violations_by_impact": site.get("violations_by_impact", {}),
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
        "specialist": {
            "architecture": specialist_report.get("architecture"),
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
        job = {"job_id": job_id, "kind": kind, "status": "queued", "created_at": _now(), "run_dir": str(run_dir)}
        self._write(job)
        self.pool.submit(self._execute, job_id, function, run_dir, *args)
        return self.public(job_id)

    def _write(self, job: dict[str, Any]) -> None:
        with self.lock:
            self.jobs[job["job_id"]] = job
            _atomic_json(Path(job["run_dir"]) / "job.json", job)

    def _execute(self, job_id: str, function, run_dir: Path, *args) -> None:
        job = self.jobs[job_id]
        job.update(status="running", started_at=_now())
        self._write(job)
        try:
            result = function(run_dir, *args)
            _atomic_json(run_dir / "result.json", result)
            job.update(status="completed", completed_at=_now(), result_path=str(run_dir / "result.json"))
        except Exception as exc:
            LOGGER.exception("job_failed job_id=%s", job_id)
            job.update(status="failed", completed_at=_now(), error=f"{type(exc).__name__}: {exc}"[:4000])
        self._write(job)

    def public(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            path = self.root / job_id / "job.json"
            if not path.is_file():
                raise KeyError(job_id)
            job = json.loads(path.read_text(encoding="utf-8"))
        return {key: value for key, value in job.items() if key != "run_dir"}

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

        def task(run_dir: Path, values: list[str], timeout: float) -> dict[str, Any]:
            result = scan_sites(values, app.state.axe_js, timeout)
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

        def task(run_dir: Path, source_url: str, maximum: int, timeout: float) -> dict[str, Any]:
            scan = suggestion_scanner([source_url], app.state.axe_js, timeout, run_dir)
            specialist_report = None
            if scan.get("sites") and scan["sites"][0].get("status") == "completed":
                specialist_report = suggestion_specialist_runner(
                    run_dir / "live_page", run_dir / "learning_v2_graphs",
                )
            result = generate_live_suggestions(
                scan, max_suggestions=maximum,
                generator_factory=suggestion_generator_factory,
                specialist_report=specialist_report,
            )
            result["run_id"] = run_dir.name
            if result.get("screenshot_artifact"):
                result["screenshot_url"] = (
                    f"/v1/jobs/{run_dir.name}/artifacts/{result['screenshot_artifact']}"
                )
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

        def task(run_dir: Path, selected: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
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
            return run_phase9(args)

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
        return json.loads((store.root / job_id / "result.json").read_text(encoding="utf-8"))

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
