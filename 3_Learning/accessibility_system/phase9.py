"""Generate OpenAI-structured repairs and validate each attempt in an isolated sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .repair.generator import OpenAIRepairGenerator
from .repair.policy import proposal_policy_errors
from .repair.validators import validate_repair


LOGGER = logging.getLogger("accessibility_system.phase9")
FATAL_ERROR_CATEGORIES = frozenset({
    "authentication_failed", "connection_failed", "model_or_endpoint_not_found",
    "permission_denied", "rate_limit_or_quota_exceeded", "request_or_schema_rejected",
})


def _configure_logging(output_dir: Path, level: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "phase_9.log"
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler); LOGGER.addHandler(console_handler)
    LOGGER.propagate = False
    return log_path


def _error_category(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "AuthenticationError": return "authentication_failed"
    if name in {"APIConnectionError", "APITimeoutError"}: return "connection_failed"
    if name == "NotFoundError": return "model_or_endpoint_not_found"
    if name == "RateLimitError": return "rate_limit_or_quota_exceeded"
    if name in {"BadRequestError", "UnprocessableEntityError"}: return "request_or_schema_rejected"
    if name == "PermissionDeniedError": return "permission_denied"
    if name == "PatchApplicationError": return "patch_application_failed"
    if name in {"ValidationError", "JSONDecodeError"}: return "structured_output_or_input_invalid"
    return "unexpected_error"


def _safe_error(exc: Exception) -> str:
    category = _error_category(exc)
    if category == "authentication_failed":
        return "AuthenticationError: endpoint rejected the configured API key (credential redacted)"
    message = str(exc)
    message = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED_API_KEY]", message)
    message = re.sub(r"(?i)(api[_ -]?key[^:]*:\s*)[^,}\s]+", r"\1[REDACTED]", message)
    return f"{type(exc).__name__}: {message}"[:4000]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reason_summary(reasons: list[str], *, limit: int = 1000) -> str:
    """Produce one safe, concise log field from structured rejection reasons."""
    cleaned = [" ".join(str(reason).split()) for reason in reasons if str(reason).strip()]
    if not cleaned:
        return "unspecified"
    message = "; ".join(cleaned)
    return message if len(message) <= limit else f"{message[:limit - 3]}..."


def run(args: argparse.Namespace, *, generator: Any | None = None) -> dict[str, Any]:
    run_started = time.perf_counter()
    inputs = json.loads(args.generator_inputs.read_text(encoding="utf-8"))
    repair_truth_path = getattr(args, "repair_truth", None)
    repair_truth = {}
    if repair_truth_path:
        truth_payload = json.loads(Path(repair_truth_path).read_text(encoding="utf-8"))
        repair_truth = {str(item["query_id"]): item for item in truth_payload.get("cases", [])}
    selected = [item for item in inputs if item.get("condition") == args.condition][:args.max_proposals]
    if not selected:
        raise ValueError(f"No generator inputs found for condition {args.condition!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_url = getattr(args, "base_url", None)
    api_mode = getattr(args, "api_mode", "responses")
    requested_model = getattr(args, "model", None)
    generator = generator or OpenAIRepairGenerator(
        model=requested_model, base_url=base_url, api_mode=api_mode,
        request_timeout_seconds=getattr(args, "request_timeout_seconds", None),
        http_logger=LOGGER,
        log_http_bodies=bool(getattr(args, "log_http_bodies", False)),
    )
    effective_model = str(getattr(generator, "model", requested_model or "provider_default"))
    LOGGER.info(
        "run_start model=%s api_mode=%s endpoint=%s condition=%s selected=%d browser=%s",
        effective_model, api_mode, base_url or "official_openai", args.condition, len(selected), not args.skip_browser,
    )
    attempts: list[dict[str, Any]] = []
    seen_proposal_ids: set[str] = set()
    fatal_error: dict[str, str] | None = None
    for index, item in enumerate(selected, start=1):
        attempt_started = time.perf_counter()
        stop_after_attempt = False
        finding = item.get("original_finding", {})
        site_id = str(finding.get("site_id", ""))
        source_path = args.corpus_dir / site_id / "0.html"
        attempt: dict[str, Any] = {
            "query_id": item.get("query_id"), "site_id": site_id,
            "finding_id": finding.get("finding_id"), "condition": args.condition,
        }
        if fatal_error:
            attempt.update(
                status="skipped_after_fatal_error", error_category=fatal_error["category"],
                error=f"Skipped after prior fatal error: {fatal_error['message']}",
            )
            LOGGER.warning("attempt_skipped index=%d query_id=%s cause=%s", index, item.get("query_id"), fatal_error["category"])
            attempts.append(attempt)
            continue
        LOGGER.info("attempt_start index=%d query_id=%s site_id=%s rule_id=%s", index, item.get("query_id"), site_id, finding.get("rule_id"))
        if not source_path.is_file():
            attempt.update(status="generation_or_input_failed", error_category="missing_source", error=f"Missing source snapshot: {source_path}")
            LOGGER.error("attempt_failed index=%d category=missing_source source=%s", index, source_path)
            attempts.append(attempt)
            continue
        try:
            generation_retries = max(0, int(getattr(args, "generation_retries", 1)))
            generation = None
            generation_input = item
            for generation_index in range(generation_retries + 1):
                try:
                    generation = generator.generate(generation_input)
                    attempt["generation_attempts"] = generation_index + 1
                    break
                except Exception as generation_error:
                    if _error_category(generation_error) != "structured_output_or_input_invalid" or generation_index >= generation_retries:
                        raise
                    generation_input = {
                        **item,
                        "_schema_retry_feedback": _safe_error(generation_error),
                    }
                    LOGGER.warning(
                        "generation_retry index=%d retry=%d category=structured_output_or_input_invalid",
                        index, generation_index + 1,
                    )
            if generation is None:  # pragma: no cover - loop either returns or raises
                raise RuntimeError("Structured generation produced no result")
            proposal = generation.proposal
            policy_errors = proposal_policy_errors(proposal, item)
            if proposal.proposal_id in seen_proposal_ids:
                policy_errors.append(f"duplicate proposal_id: {proposal.proposal_id}")
            seen_proposal_ids.add(proposal.proposal_id)
            proposal_path = args.output_dir / "proposals" / f"{len(attempts):04d}-{proposal.proposal_id}.json"
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_path.write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
            attempt["generation"] = {
                "response_id": generation.response_id,
                "model": generation.model,
                "usage": generation.usage,
                "refusal": generation.refusal,
                "proposal_path": str(proposal_path),
            }
            if policy_errors:
                attempt.update(status="rejected", policy_errors=policy_errors)
                LOGGER.warning(
                    "attempt_rejected index=%d category=grounding_policy reasons=%s",
                    index, _reason_summary(policy_errors),
                )
            else:
                validation_finding = dict(finding)
                truth = repair_truth.get(str(item.get("query_id")))
                if truth:
                    validation_finding.update({
                        key: truth[key] for key in ("status", "semantic_verified", "visual_verified", "oracle_operations") if key in truth
                    })
                validation = validate_repair(
                    source_path=source_path,
                    proposal=proposal,
                    original_finding=validation_finding,
                    output_dir=args.output_dir / "attempts",
                    axe_js=args.axe_js,
                    run_browser=not args.skip_browser,
                )
                attempt.update(status=validation.outcome, validation=validation.model_dump(mode="json"))
                if validation.outcome == "rejected":
                    rejection_reasons = list(validation.rejection_reasons) or list(validation.human_review_reasons)
                    LOGGER.warning(
                        "attempt_rejected index=%d category=validation reasons=%s",
                        index, _reason_summary(rejection_reasons),
                    )
                else:
                    LOGGER.info(
                        "attempt_complete index=%d outcome=%s target_resolved=%s regressions=%d review_reasons=%d",
                        index, validation.outcome, validation.target_resolved, len(validation.new_regressions), len(validation.human_review_reasons),
                    )
                    stop_after_attempt = (
                        validation.outcome == "accepted" and bool(getattr(args, "stop_on_accepted", False))
                    ) or bool(
                        getattr(args, "stop_on_non_rejected", False)
                    )
        except Exception as exc:
            category = _error_category(exc); safe_error = _safe_error(exc)
            status = "generation_authentication_failed" if category == "authentication_failed" else "generation_or_input_failed"
            attempt.update(status=status, error_category=category, error=safe_error)
            LOGGER.error("attempt_failed index=%d category=%s error=%s", index, category, safe_error)
            if category in FATAL_ERROR_CATEGORIES:
                fatal_error = {"category": category, "message": safe_error}
        attempt["duration_seconds"] = time.perf_counter() - attempt_started
        attempts.append(attempt)
        if stop_after_attempt:
            LOGGER.info(
                "run_stop index=%d outcome=%s reason=%s",
                index, attempt["status"],
                "first_accepted_outcome" if attempt["status"] == "accepted" and getattr(args, "stop_on_accepted", False) else "first_non_rejected_outcome",
            )
            break

    counts = Counter(str(item["status"]) for item in attempts)
    report = {
        "schema_version": 1,
        "phase": 9,
        "status": "structured_repair_validation_run",
        "model": effective_model,
        "api": "OpenAI-compatible structured output API",
        "api_mode": api_mode,
        "endpoint": base_url or "official_openai",
        "structured_output_contract": "accessibility_system.repair.contracts.RepairProposal",
        "condition": args.condition,
        "attempt_count": len(attempts),
        "run_duration_seconds": time.perf_counter() - run_started,
        "outcome_counts": dict(sorted(counts.items())),
        "attempts": attempts,
        "acceptance_policy": {
            "target_must_be_resolved": True,
            "new_in_scope_regressions_allowed": 0,
            "semantic_syntax_only_acceptance_allowed": False,
            "weak_label_repairs_require_human_review": True,
        },
        "limitations": [
            "The saved Phase 8 real-page findings are axe-derived weak labels; they cannot be automatically accepted without independent verification.",
            "Learned Phase 5 graph specialists are not rerun on modified HTML because there is no canonical modified-page graph regeneration path; axe and deterministic specialists are the executable in-scope detectors.",
            "Modal, live-region, and workflow-specific replay requires a page-specific interaction script when those mechanisms are present.",
        ],
    }
    report_path = args.output_dir / "phase_9_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("run_complete outcomes=%s report=%s", dict(sorted(counts.items())), report_path)
    for handler in LOGGER.handlers:
        handler.flush()
    output_hashes = {
        str(path.relative_to(args.output_dir)): _sha256(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "phase": 9,
        "inputs": {
            "generator_inputs": str(args.generator_inputs),
            "generator_inputs_sha256": _sha256(args.generator_inputs),
            "corpus_dir": str(args.corpus_dir),
            "axe_js": str(args.axe_js),
            "axe_js_sha256": _sha256(args.axe_js),
            "repair_truth": str(repair_truth_path) if repair_truth_path else None,
            "repair_truth_sha256": _sha256(Path(repair_truth_path)) if repair_truth_path else None,
        },
        "config": {
            "model": effective_model, "condition": args.condition,
            "api_mode": api_mode, "base_url": base_url,
            "max_proposals": args.max_proposals, "skip_browser": args.skip_browser,
            "generation_retries": max(0, int(getattr(args, "generation_retries", 1))),
            "stop_on_non_rejected": bool(getattr(args, "stop_on_non_rejected", False)),
            "stop_on_accepted": bool(getattr(args, "stop_on_accepted", False)),
            "request_timeout_seconds": getattr(args, "request_timeout_seconds", None),
            "log_http_bodies": bool(getattr(args, "log_http_bodies", False)),
        },
        "outputs": output_hashes,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-inputs", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--axe-js", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("no_rag", "flat_vector_rag", "graph_constrained_rag"), default="graph_constrained_rag")
    parser.add_argument("--model")
    parser.add_argument("--api-mode", choices=("responses", "chat_completions"), default="responses")
    parser.add_argument("--base-url", help="OpenAI-compatible API root, for example https://provider.example/v1; never include /chat/completions")
    parser.add_argument("--max-proposals", type=int, default=10)
    parser.add_argument("--generation-retries", type=int, default=1, help="Retry schema-invalid provider output; authentication and other fatal errors are never retried")
    parser.add_argument(
        "--request-timeout-seconds", type=float, default=90.0,
        help="Maximum time for one provider request; prevents a stalled endpoint from blocking the bounded run",
    )
    parser.add_argument("--repair-truth", type=Path, help="Independent oracle file used only by validation and never sent to the generator")
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument(
        "--log-http-bodies", action="store_true",
        help="Include redacted, truncated HTTP JSON request/response bodies in phase_9.log; metadata is always logged",
    )
    parser.add_argument(
        "--stop-on-non-rejected", action="store_true",
        help="Stop after the first accepted or requires_human_review validation outcome; generation failures never stop the run",
    )
    parser.add_argument(
        "--stop-on-accepted", action="store_true",
        help="Stop only after the first accepted validation outcome; rejected and human-review outcomes continue",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = _configure_logging(args.output_dir, args.log_level)
    report = run(args)
    print(json.dumps({"status": report["status"], "attempts": report["attempt_count"], "outcomes": report["outcome_counts"]}, indent=2))
    print(f"Phase 9 log: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
