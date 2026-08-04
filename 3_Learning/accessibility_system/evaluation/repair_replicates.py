"""Aggregate predeclared stochastic Phase 9 repair replicates.

Each input is labelled ``CONDITION:REPLICATE=phase_9_report.json``.  The
aggregator refuses unbalanced replicate/query universes and reports both
per-replicate metrics and hierarchical bootstrap intervals.  This keeps random
LLM variation visible instead of presenting a single favourable run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .repair_study import (
    ConditionRun,
    _attempt_record,
    _condition_metrics,
    _load_supplied_citations,
    _percentile,
)


RATE_FIELDS = (
    "generation_success_rate", "target_resolution_rate", "validated_repair_rate",
    "regression_free_accepted_rate", "oracle_exact_acceptance_precision",
    "oracle_exact_acceptance_recall",
)
REQUIRED_LLM_CONDITIONS = frozenset({"no_rag", "flat_vector_rag", "graph_constrained_rag"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_run(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("Run must be CONDITION:REPLICATE=/path/to/phase_9_report.json")
    label, raw_path = value.split("=", 1)
    condition, replicate = label.split(":", 1)
    if not condition.strip() or not replicate.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Condition, replicate, and path must be non-empty")
    return condition.strip(), replicate.strip(), Path(raw_path)


def _load(values: list[tuple[str, str, Path]]) -> dict[str, dict[str, ConditionRun]]:
    result: dict[str, dict[str, ConditionRun]] = defaultdict(dict)
    for condition, replicate, path in values:
        if replicate in result[condition]:
            raise ValueError(f"Duplicate replicate {condition}:{replicate}")
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if str(report.get("condition", condition)) != condition:
            raise ValueError(f"Condition label {condition!r} disagrees with {path}")
        attempts = report.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"Phase 9 report has no attempts list: {path}")
        query_ids = [str(item.get("query_id", "")) for item in attempts]
        if not all(query_ids) or len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Replicate {condition}:{replicate} needs unique non-empty query IDs")
        result[condition][replicate] = ConditionRun(condition, path.resolve(), report)
    if not result:
        raise ValueError("At least one replicate run is required")
    return dict(result)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _hierarchical_rate_ci(records: list[dict[str, Any]], field: str, samples: int, seed: int) -> list[float]:
    by_replicate_site: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_replicate_site[str(record["replicate_id"])][str(record["site_id"])].append(record)
    replicate_ids = sorted(by_replicate_site)
    if not replicate_ids:
        return [0.0, 0.0]
    rng = random.Random(seed); draws = []
    for _ in range(samples):
        sampled_replicates = [rng.choice(replicate_ids) for _ in replicate_ids]
        values: list[float] = []
        for replicate in sampled_replicates:
            by_site = by_replicate_site[replicate]
            sites = sorted(by_site)
            for site in [rng.choice(sites) for _ in sites]:
                values.extend(float(item[field]) for item in by_site[site])
        draws.append(sum(values) / len(values) if values else 0.0)
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _hierarchical_difference_ci(
    left: list[dict[str, Any]], right: list[dict[str, Any]], field: str, samples: int, seed: int,
) -> list[float]:
    left_map = {(str(item["replicate_id"]), str(item["query_id"])): item for item in left}
    right_map = {(str(item["replicate_id"]), str(item["query_id"])): item for item in right}
    common = sorted(set(left_map) & set(right_map))
    by_replicate_site: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for replicate, query_id in common:
        by_replicate_site[replicate][str(left_map[(replicate, query_id)]["site_id"])].append(query_id)
    replicates = sorted(by_replicate_site)
    if not replicates:
        return [0.0, 0.0]
    rng = random.Random(seed); draws = []
    for _ in range(samples):
        differences = []
        for replicate in [rng.choice(replicates) for _ in replicates]:
            sites = sorted(by_replicate_site[replicate])
            for site in [rng.choice(sites) for _ in sites]:
                for query_id in by_replicate_site[replicate][site]:
                    key = (replicate, query_id)
                    differences.append(float(left_map[key][field]) - float(right_map[key][field]))
        draws.append(sum(differences) / len(differences) if differences else 0.0)
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    runs = _load(args.run)
    supplied = _load_supplied_citations(args.generator_inputs)
    oracle_queries: set[str] = set()
    if args.repair_truth:
        payload = json.loads(args.repair_truth.read_text(encoding="utf-8"))
        oracle_queries = {str(item["query_id"]) for item in payload.get("cases", [])}
    records: dict[str, list[dict[str, Any]]] = {}
    per_replicate: dict[str, dict[str, dict[str, Any]]] = {}
    universes: dict[str, dict[str, list[str]]] = {}
    for condition, condition_runs in runs.items():
        records[condition] = []
        per_replicate[condition] = {}
        universes[condition] = {}
        for replicate, run in sorted(condition_runs.items()):
            current = [_attempt_record(run, item, supplied, oracle_queries) for item in run.report["attempts"]]
            for item in current:
                item["replicate_id"] = replicate
            records[condition].extend(current)
            per_replicate[condition][replicate] = _condition_metrics(current)
            universes[condition][replicate] = sorted(item["query_id"] for item in current)
    llm_replicates = {condition: set(runs.get(condition, {})) for condition in REQUIRED_LLM_CONDITIONS}
    required_present = REQUIRED_LLM_CONDITIONS <= set(runs)
    balanced_replicates = required_present and len({frozenset(values) for values in llm_replicates.values()}) == 1
    query_universes = [
        tuple(values) for condition in REQUIRED_LLM_CONDITIONS for values in universes.get(condition, {}).values()
    ]
    identical_queries = bool(query_universes) and len(set(query_universes)) == 1
    replicate_count = min((len(values) for values in llm_replicates.values()), default=0)
    condition_metrics = {}
    for condition, condition_records in records.items():
        replicate_metrics = per_replicate[condition]
        condition_metrics[condition] = {
            "replicate_count": len(replicate_metrics),
            "per_replicate": replicate_metrics,
            "pooled": _condition_metrics(condition_records),
            "mean_across_replicates": {
                field: _mean([float(metric[field]) for metric in replicate_metrics.values() if metric.get(field) is not None])
                for field in RATE_FIELDS
            },
            "between_replicate_sample_sd": {
                field: _sample_sd([float(metric[field]) for metric in replicate_metrics.values() if metric.get(field) is not None])
                for field in RATE_FIELDS
            },
            "hierarchical_95_ci": {
                field: _hierarchical_rate_ci(
                    condition_records,
                    "accepted" if field == "regression_free_accepted_rate" else (
                        "validated_repair" if field == "validated_repair_rate" else
                        "target_resolved" if field == "target_resolution_rate" else "generated"
                    ),
                    args.bootstrap_samples, args.seed + index,
                )
                for index, field in enumerate((
                    "generation_success_rate", "target_resolution_rate",
                    "validated_repair_rate", "regression_free_accepted_rate",
                ))
            },
        }
    paired = {}
    ordered = [condition for condition in ("no_rag", "flat_vector_rag", "graph_constrained_rag") if condition in records]
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            paired[f"{left}_vs_{right}"] = {
                "direction": "left_minus_right",
                "accepted_rate_difference_hierarchical_95_ci": _hierarchical_difference_ci(
                    records[left], records[right], "accepted", args.bootstrap_samples, args.seed,
                ),
                "validated_rate_difference_hierarchical_95_ci": _hierarchical_difference_ci(
                    records[left], records[right], "validated_repair", args.bootstrap_samples, args.seed + 1,
                ),
            }
    readiness = {
        "three_required_llm_conditions_present": required_present,
        "replicate_ids_balanced_across_llm_conditions": balanced_replicates,
        "query_sets_identical_across_conditions_and_replicates": identical_queries,
        "minimum_replicates_required": args.minimum_replicates,
        "minimum_replicates_observed": replicate_count,
        "predeclared_minimum_replicates_met": replicate_count >= args.minimum_replicates,
    }
    readiness["replicate_study_complete"] = all(value for key, value in readiness.items() if isinstance(value, bool))
    report = {
        "schema_version": 1, "phase": 10, "status": "stochastic_repair_replicate_study",
        "unit": "repair_query_within_replicate", "conditions": sorted(runs),
        "metrics": condition_metrics, "paired_comparisons": paired,
        "readiness": readiness,
        "interpretation_rules": [
            "Replicate and failed-attempt outcomes are never discarded.",
            "Intervals resample stochastic replicates and sites hierarchically.",
            "LLM conditions must have identical replicate labels and query universes.",
            "A single successful run is not treated as evidence of stochastic stability.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "phase_10_replicate_study.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1, "seed": args.seed, "bootstrap_samples": args.bootstrap_samples,
        "minimum_replicates": args.minimum_replicates,
        "inputs": [
            {"condition": condition, "replicate": replicate, "path": str(run.report_path), "sha256": _sha256(run.report_path)}
            for condition, condition_runs in sorted(runs.items()) for replicate, run in sorted(condition_runs.items())
        ],
        "generator_inputs": {"path": str(args.generator_inputs), "sha256": _sha256(args.generator_inputs)} if args.generator_inputs else None,
        "repair_truth": {"path": str(args.repair_truth), "sha256": _sha256(args.repair_truth)} if args.repair_truth else None,
        "outputs": {output.name: _sha256(output)},
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--generator-inputs", type=Path)
    parser.add_argument("--repair-truth", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-replicates", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    report = run_study(parse_args())
    print(json.dumps({"status": report["status"], "readiness": report["readiness"]}, indent=2))


if __name__ == "__main__":
    main()
