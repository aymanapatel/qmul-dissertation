"""Evaluate matched Phase 9 repair conditions without hiding failed attempts.

The unit of analysis is a proposed repair/query.  Automatic acceptance is kept
separate from generation, patch applicability, target resolution, and human
contextual correctness.  This prevents a syntactically valid LLM response from
being counted as a successful accessibility repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


OUTCOMES = ("accepted", "rejected", "requires_human_review")
RATING_FIELDS = ("contextual_correctness", "safety", "helpfulness")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float | None:
    entries = list(values)
    return sum(entries) / len(entries) if entries else None


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class ConditionRun:
    condition: str
    report_path: Path
    report: dict[str, Any]


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must be CONDITION=/path/to/phase_9_report.json")
    condition, raw_path = value.split("=", 1)
    if not condition.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Run condition and report path must be non-empty")
    return condition.strip(), Path(raw_path)


def _load_runs(values: list[tuple[str, Path]]) -> list[ConditionRun]:
    runs: list[ConditionRun] = []
    seen_conditions: set[str] = set()
    for condition, path in values:
        if condition in seen_conditions:
            raise ValueError(f"Duplicate repair condition: {condition}")
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        attempts = report.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"Phase 9 report has no attempts list: {path}")
        query_ids = [str(item.get("query_id", "")) for item in attempts]
        if not all(query_ids) or len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Each attempt must have a unique non-empty query_id: {path}")
        declared = str(report.get("condition", condition))
        if declared != condition:
            raise ValueError(f"Condition label {condition!r} disagrees with report value {declared!r}")
        seen_conditions.add(condition)
        runs.append(ConditionRun(condition, path.resolve(), report))
    if not runs:
        raise ValueError("At least one Phase 9 run is required")
    return runs


def _proposal_path(run: ConditionRun, attempt: dict[str, Any]) -> Path | None:
    raw = (attempt.get("generation") or {}).get("proposal_path")
    if not raw:
        return None
    candidate = Path(str(raw))
    options = [candidate]
    if not candidate.is_absolute():
        options.extend((run.report_path.parent / candidate, run.report_path.parent / "proposals" / candidate.name))
    for option in options:
        if option.is_file():
            return option.resolve()
    return None


def _load_proposal(run: ConditionRun, attempt: dict[str, Any]) -> dict[str, Any] | None:
    path = _proposal_path(run, attempt)
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def _usage_value(usage: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _attempt_record(run: ConditionRun, attempt: dict[str, Any], supplied: dict[tuple[str, str], set[str]], oracle_queries: set[str]) -> dict[str, Any]:
    proposal = _load_proposal(run, attempt)
    validation = attempt.get("validation") or {}
    generation = attempt.get("generation") or {}
    usage = generation.get("usage") or {}
    status = str(attempt.get("status", "unknown"))
    decision = str((proposal or {}).get("decision", "not_generated"))
    cited = {str(value) for value in (proposal or {}).get("cited_record_ids", [])}
    query_id = str(attempt["query_id"])
    allowed = supplied.get((run.condition, query_id), set())
    regressions = list(validation.get("new_regressions") or [])
    patch = validation.get("patch_evidence") or {}
    oracle_available = str(attempt["query_id"]) in oracle_queries or "oracle_match" in patch
    policy_errors = list(attempt.get("policy_errors") or [])
    applied = bool(patch.get("applied", False))
    target_resolved = bool(validation.get("target_resolved", False))
    regression_free = bool(validation) and not regressions
    parsed = bool(validation) and not any(
        token in str(reason).lower()
        for reason in validation.get("rejection_reasons", [])
        for token in ("parse", "patch application", "selector resolved")
    )
    return {
        "condition": run.condition,
        "query_id": query_id,
        "site_id": str(attempt.get("site_id", "")),
        "status": status,
        "generated": bool(generation),
        "proposal_available": proposal is not None,
        "decision": decision,
        "proposed": decision == "propose",
        "policy_passed": proposal is not None and not policy_errors,
        "sandbox_ran": bool(validation),
        "patch_applied": applied,
        "parse_success": parsed,
        "target_resolved": target_resolved,
        "regression_free": regression_free,
        "validated_repair": target_resolved and regression_free,
        "oracle_available": oracle_available,
        "oracle_match": bool(patch.get("oracle_match", False)),
        "accepted": status == "accepted",
        "rejected": status == "rejected",
        "requires_human_review": status == "requires_human_review",
        "validation_gate_prevented_auto_acceptance": decision == "propose" and status != "accepted",
        # Counterfactual paired gate ablation: without automatic validation,
        # every schema- and policy-valid proposal would be provisionally
        # accepted.  Reusing the exact same generated proposal avoids an LLM
        # sampling confound between the with/without-gate conditions.
        "ungated_provisional_acceptance": decision == "propose" and proposal is not None and not policy_errors,
        "new_regression_count": len(regressions),
        "citation_count": len(cited),
        "supplied_citation_count": len(allowed),
        "citation_valid": not allowed or cited <= allowed,
        "citation_utilisation": _ratio(len(cited & allowed), len(allowed)) if allowed else None,
        "duration_seconds": float(attempt.get("duration_seconds", 0.0) or 0.0),
        "prompt_tokens": _usage_value(usage, "prompt_tokens", "input_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens", "output_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
        "cost": _usage_value(usage, "cost"),
        "error_category": attempt.get("error_category"),
    }


def _load_supplied_citations(path: Path | None) -> dict[tuple[str, str], set[str]]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], set[str]] = {}
    for item in rows:
        key = (str(item.get("condition", "")), str(item.get("query_id", "")))
        result[key] = {str(citation.get("record_id")) for citation in item.get("citations", [])}
    return result


def _condition_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    proposed = [item for item in records if item["proposed"]]
    sandboxed = [item for item in records if item["sandbox_ran"]]
    applied = [item for item in records if item["patch_applied"]]
    citeable = [item for item in proposed if item["supplied_citation_count"]]
    oracle_records = [item for item in records if item["oracle_available"]]
    oracle_tp = sum(item["accepted"] and item["oracle_match"] for item in oracle_records)
    oracle_fp = sum(item["accepted"] and not item["oracle_match"] for item in oracle_records)
    ungated_tp = sum(item["ungated_provisional_acceptance"] and item["oracle_match"] for item in oracle_records)
    ungated_fp = sum(item["ungated_provisional_acceptance"] and not item["oracle_match"] for item in oracle_records)
    counts = {outcome: sum(item[outcome if outcome != "requires_human_review" else "requires_human_review"] for item in records) for outcome in OUTCOMES}
    return {
        "unit": "repair_query",
        "attempt_count": count,
        "site_count": len({item["site_id"] for item in records}),
        "generation_success_rate": _ratio(sum(item["generated"] for item in records), count),
        "generation_failure_rate": _ratio(sum(not item["generated"] for item in records), count),
        "structured_proposal_rate": _ratio(sum(item["proposal_available"] for item in records), count),
        "propose_rate": _ratio(len(proposed), count),
        "grounding_policy_pass_rate": _ratio(sum(item["policy_passed"] for item in records), count),
        "sandbox_execution_rate": _ratio(len(sandboxed), count),
        "patch_applicability_rate": _ratio(len(applied), len(proposed)),
        "parse_success_rate": _ratio(sum(item["parse_success"] for item in sandboxed), len(sandboxed)),
        "target_resolution_rate": _ratio(sum(item["target_resolved"] for item in sandboxed), len(sandboxed)),
        "regression_free_rate": _ratio(sum(item["regression_free"] for item in sandboxed), len(sandboxed)),
        "validated_repair_rate": _ratio(sum(item["validated_repair"] for item in records), count),
        "regression_free_accepted_rate": _ratio(sum(item["accepted"] for item in records), count),
        "rejection_rate": _ratio(counts["rejected"], count),
        "human_review_rate": _ratio(counts["requires_human_review"], count),
        "validation_gate_intervention_rate": _ratio(sum(item["validation_gate_prevented_auto_acceptance"] for item in proposed), len(proposed)),
        "new_regression_rate_per_applied_patch": _ratio(sum(item["new_regression_count"] > 0 for item in applied), len(applied)),
        "citation_validity_rate": _ratio(sum(item["citation_valid"] for item in citeable), len(citeable)) if citeable else None,
        "mean_retrieved_citation_utilisation": _mean(item["citation_utilisation"] for item in citeable if item["citation_utilisation"] is not None),
        "outcome_counts": counts,
        "total_prompt_tokens": sum(item["prompt_tokens"] for item in records),
        "total_completion_tokens": sum(item["completion_tokens"] for item in records),
        "total_tokens": sum(item["total_tokens"] for item in records),
        "total_cost": sum(item["cost"] for item in records),
        "mean_latency_seconds": _mean(item["duration_seconds"] for item in records if item["duration_seconds"] > 0),
        "oracle_evaluable": bool(oracle_records),
        "oracle_exact_acceptance_precision": _ratio(oracle_tp, oracle_tp + oracle_fp) if oracle_records else None,
        "oracle_exact_acceptance_recall": _ratio(oracle_tp, len(oracle_records)) if oracle_records else None,
        "oracle_exact_match_rate": _ratio(sum(item["oracle_match"] for item in oracle_records), len(oracle_records)) if oracle_records else None,
        "validation_gate_ablation": {
            "design": "paired counterfactual on the identical generated proposal",
            "without_validation": {
                "provisional_acceptance_rate": _ratio(sum(item["ungated_provisional_acceptance"] for item in records), count),
                "oracle_exact_acceptance_precision": _ratio(ungated_tp, ungated_tp + ungated_fp) if oracle_records else None,
                "oracle_exact_acceptance_recall": _ratio(ungated_tp, len(oracle_records)) if oracle_records else None,
                "known_regression_acceptance_rate": _ratio(
                    sum(item["ungated_provisional_acceptance"] and item["new_regression_count"] > 0 for item in records),
                    sum(item["ungated_provisional_acceptance"] for item in records),
                ),
            },
            "with_validation": {
                "automatic_acceptance_rate": _ratio(sum(item["accepted"] for item in records), count),
                "oracle_exact_acceptance_precision": _ratio(oracle_tp, oracle_tp + oracle_fp) if oracle_records else None,
                "oracle_exact_acceptance_recall": _ratio(oracle_tp, len(oracle_records)) if oracle_records else None,
                "known_regression_acceptance_rate": _ratio(
                    sum(item["accepted"] and item["new_regression_count"] > 0 for item in records),
                    sum(item["accepted"] for item in records),
                ),
            },
            "prevented_known_false_acceptance_count": sum(
                item["ungated_provisional_acceptance"] and not item["accepted"]
                and (not item["oracle_match"] or item["new_regression_count"] > 0)
                for item in records
            ),
            "prevented_or_deferred_oracle_match_count": sum(
                item["ungated_provisional_acceptance"] and not item["accepted"]
                and item["oracle_match"] and item["new_regression_count"] == 0
                for item in records
            ),
        },
        "failure_taxonomy": dict(sorted((key, sum(item["error_category"] == key for item in records)) for key in {item["error_category"] for item in records if item["error_category"]})),
    }


def _bootstrap_difference(left: list[dict[str, Any]], right: list[dict[str, Any]], field: str, samples: int, seed: int) -> list[float]:
    left_map = {item["query_id"]: item for item in left}; right_map = {item["query_id"]: item for item in right}
    common = sorted(set(left_map) & set(right_map))
    if not common:
        return [0.0, 0.0]
    by_site: dict[str, list[str]] = defaultdict(list)
    for query_id in common:
        by_site[left_map[query_id]["site_id"]].append(query_id)
    sites = sorted(by_site); rng = random.Random(seed); differences = []
    for _ in range(samples):
        chosen = [rng.choice(sites) for _ in sites]
        query_ids = [query for site in chosen for query in by_site[site]]
        differences.append(_mean(float(left_map[q][field]) - float(right_map[q][field]) for q in query_ids) or 0.0)
    return [_percentile(differences, 0.025), _percentile(differences, 0.975)]


def _mcnemar_exact(left: list[dict[str, Any]], right: list[dict[str, Any]], field: str) -> dict[str, Any]:
    left_map = {item["query_id"]: item for item in left}; right_map = {item["query_id"]: item for item in right}
    common = set(left_map) & set(right_map)
    left_only = sum(bool(left_map[q][field]) and not bool(right_map[q][field]) for q in common)
    right_only = sum(bool(right_map[q][field]) and not bool(left_map[q][field]) for q in common)
    discordant = left_only + right_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(left_only, right_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {"common_queries": len(common), "left_only": left_only, "right_only": right_only, "exact_p_value": p_value}


def _load_ratings(path: Path | None, known: set[tuple[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("ratings", payload) if isinstance(payload, dict) else payload
    metadata = {key: payload.get(key) for key in ("schema_version", "blinded", "adjudicated") if key in payload} if isinstance(payload, dict) else {}
    if not isinstance(rows, list):
        raise ValueError("Human ratings must be a list or an object with a ratings list")
    seen = set(); result = []
    for item in rows:
        key = (str(item.get("condition", "")), str(item.get("query_id", "")), str(item.get("rater_id", "")))
        if not all(key) or key in seen:
            raise ValueError("Each human rating needs a unique condition/query_id/rater_id")
        if key[:2] not in known:
            raise ValueError(f"Human rating refers to an unknown condition/query: {key[:2]}")
        for field in RATING_FIELDS:
            value = item.get(field)
            if not isinstance(value, (int, float)) or not 1 <= value <= 5:
                raise ValueError(f"{field} must be in the range 1..5")
        if not isinstance(item.get("acceptable"), bool):
            raise ValueError("Human rating acceptable must be boolean")
        seen.add(key); result.append(dict(item))
    return result, metadata


def _cohen_kappa(left: list[bool], right: list[bool]) -> float:
    if not left:
        return 0.0
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    lp = sum(left) / len(left); rp = sum(right) / len(right)
    expected = lp * rp + (1 - lp) * (1 - rp)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _human_metrics(ratings: list[dict[str, Any]], records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    if not ratings:
        return {"available": False, "rated_query_condition_count": 0, "automatic_acceptance_precision": None, "automatic_acceptance_recall": None}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rating in ratings:
        grouped[(str(rating["condition"]), str(rating["query_id"]))].append(rating)
    majority = {}
    unresolved = []
    for key, rows in grouped.items():
        positive = sum(item["acceptable"] for item in rows)
        if positive * 2 == len(rows):
            unresolved.append(key)
        else:
            majority[key] = positive * 2 > len(rows)
    record_map = {(item["condition"], item["query_id"]): item for item in records}
    auto_positive = {key for key, item in record_map.items() if item["accepted"] and key in majority}
    human_positive = {key for key, value in majority.items() if value}
    tp = len(auto_positive & human_positive); fp = len(auto_positive - human_positive); fn = len(human_positive - auto_positive)
    raters = sorted({str(item["rater_id"]) for item in ratings}); kappas = []
    for index, left_rater in enumerate(raters):
        for right_rater in raters[index + 1:]:
            left = {(str(item["condition"]), str(item["query_id"])): bool(item["acceptable"]) for item in ratings if str(item["rater_id"]) == left_rater}
            right = {(str(item["condition"]), str(item["query_id"])): bool(item["acceptable"]) for item in ratings if str(item["rater_id"]) == right_rater}
            common = sorted(set(left) & set(right))
            if common:
                kappas.append(_cohen_kappa([left[k] for k in common], [right[k] for k in common]))
    per_condition = {}
    for condition in sorted({item["condition"] for item in records}):
        condition_keys = {key for key in majority if key[0] == condition}
        condition_auto = {key for key in auto_positive if key[0] == condition}
        condition_human = {key for key in human_positive if key[0] == condition}
        condition_tp = len(condition_auto & condition_human)
        condition_fp = len(condition_auto - condition_human)
        condition_fn = len(condition_human - condition_auto)
        condition_ratings = [item for item in ratings if str(item["condition"]) == condition]
        per_condition[condition] = {
            "rated_query_count": len(condition_keys),
            "majority_acceptable_rate": _ratio(len(condition_human), len(condition_keys)),
            "automatic_acceptance_precision": _ratio(condition_tp, condition_tp + condition_fp),
            "automatic_acceptance_recall": _ratio(condition_tp, condition_tp + condition_fn),
            **{f"mean_{field}": _mean(float(item[field]) for item in condition_ratings) for field in RATING_FIELDS},
        }
    return {
        "available": True,
        "blinded": bool(metadata.get("blinded", False)),
        "adjudicated": bool(metadata.get("adjudicated", False)),
        "rating_count": len(ratings),
        "rater_count": len(raters),
        "rated_query_condition_count": len(grouped),
        "resolved_majority_count": len(majority),
        "unresolved_tie_count": len(unresolved),
        "coverage": _ratio(len(grouped), len(records)),
        **{f"mean_{field}": _mean(float(item[field]) for item in ratings) for field in RATING_FIELDS},
        "majority_acceptable_rate": _ratio(len(human_positive), len(grouped)),
        "automatic_acceptance_precision": _ratio(tp, tp + fp),
        "automatic_acceptance_recall": _ratio(tp, tp + fn),
        "acceptance_confusion": {"tp": tp, "fp": fp, "fn": fn},
        "mean_pairwise_cohen_kappa": _mean(kappas),
        "per_condition": per_condition,
    }


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    runs = _load_runs(args.run)
    supplied = _load_supplied_citations(args.generator_inputs)
    repair_truth_path = getattr(args, "repair_truth", None)
    oracle_queries = set()
    if repair_truth_path:
        oracle_payload = json.loads(Path(repair_truth_path).read_text(encoding="utf-8"))
        oracle_queries = {str(item["query_id"]) for item in oracle_payload.get("cases", [])}
    records_by_condition = {
        run.condition: [_attempt_record(run, item, supplied, oracle_queries) for item in run.report["attempts"]]
        for run in runs
    }
    all_records = [item for records in records_by_condition.values() for item in records]
    known = {(item["condition"], item["query_id"]) for item in all_records}
    ratings, rating_metadata = _load_ratings(args.human_ratings, known)
    metrics = {condition: _condition_metrics(records) for condition, records in records_by_condition.items()}
    comparisons = {}
    conditions = list(records_by_condition)
    for left_index, left in enumerate(conditions):
        for right in conditions[left_index + 1:]:
            key = f"{left}_vs_{right}"
            comparisons[key] = {
                "direction": "left_minus_right",
                "accepted_rate_difference_95_ci": _bootstrap_difference(records_by_condition[left], records_by_condition[right], "accepted", args.bootstrap_samples, args.seed),
                "validated_repair_rate_difference_95_ci": _bootstrap_difference(records_by_condition[left], records_by_condition[right], "validated_repair", args.bootstrap_samples, args.seed + 1),
                "target_resolution_mcnemar": _mcnemar_exact(records_by_condition[left], records_by_condition[right], "target_resolved"),
                "accepted_mcnemar": _mcnemar_exact(records_by_condition[left], records_by_condition[right], "accepted"),
            }
    human = _human_metrics(ratings, all_records, rating_metadata)
    matched_query_sets = [set(item["query_id"] for item in records) for records in records_by_condition.values()]
    required = {"no_rag", "flat_vector_rag", "graph_constrained_rag"}
    deterministic_present = "deterministic_template" in conditions
    gate_ablation_present = all(
        condition in metrics and "validation_gate_ablation" in metrics[condition]
        for condition in required
    )
    complete_human = bool(
        ratings and human.get("blinded") and human.get("adjudicated")
        and human.get("rater_count", 0) >= 2 and human.get("coverage") == 1.0
        and human.get("unresolved_tie_count") == 0
    )
    report = {
        "schema_version": 1,
        "phase": 10,
        "status": "repair_study",
        "unit": "repair_query",
        "conditions": conditions,
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "human_assessment": human,
        "readiness": {
            "three_required_llm_conditions_present": required <= set(conditions),
            "deterministic_template_present": deterministic_present,
            "validation_gate_ablation_present": gate_ablation_present,
            "query_sets_identical": len({frozenset(values) for values in matched_query_sets}) == 1,
            "blinded_human_assessment_present": bool(ratings) and bool(human.get("blinded")),
            "two_independent_raters_with_complete_coverage": complete_human,
            "automatic_acceptance_precision_recall_available": human.get("automatic_acceptance_precision") is not None,
            "dissertation_ready": (
                required <= set(conditions) and deterministic_present and gate_ablation_present
                and len({frozenset(values) for values in matched_query_sets}) == 1
                and complete_human
            ),
        },
        "interpretation_rules": [
            "Generation or structured-output success is not counted as repair success.",
            "Validated repair requires target resolution and zero new in-scope regressions.",
            "Automatic success requires the stricter Phase 9 accepted outcome.",
            "The validation ablation is paired on each identical generated proposal: the ungated counterfactual provisionally accepts every schema- and policy-valid proposal, while the gated result is the observed sandbox outcome.",
            "Repair precision and recall require blinded human acceptability labels and are omitted otherwise.",
            "Paired comparisons include only query IDs present in both conditions and bootstrap over sites.",
        ],
        "records": all_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "phase_10_repair_study.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "phase": 10,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "inputs": {
            "runs": [{"condition": run.condition, "path": str(run.report_path), "sha256": _sha256(run.report_path)} for run in runs],
            "generator_inputs": str(args.generator_inputs) if args.generator_inputs else None,
            "generator_inputs_sha256": _sha256(args.generator_inputs) if args.generator_inputs else None,
            "human_ratings": str(args.human_ratings) if args.human_ratings else None,
            "human_ratings_sha256": _sha256(args.human_ratings) if args.human_ratings else None,
            "repair_truth": str(repair_truth_path) if repair_truth_path else None,
            "repair_truth_sha256": _sha256(Path(repair_truth_path)) if repair_truth_path else None,
        },
        "outputs": {"phase_10_repair_study.json": _sha256(output)},
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True, help="Repeat CONDITION=phase_9_report.json")
    parser.add_argument("--generator-inputs", type=Path)
    parser.add_argument("--human-ratings", type=Path)
    parser.add_argument("--repair-truth", type=Path, help="Independent controlled oracle; used for exact-match precision/recall")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    report = run_study(parse_args())
    print(json.dumps({"status": report["status"], "conditions": report["conditions"], "readiness": report["readiness"]}, indent=2))


if __name__ == "__main__":
    main()
