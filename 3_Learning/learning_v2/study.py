"""Held-out Phase 7 detection study over frozen Phase 5 specialists.

The available real-page labels are axe-derived weak labels. The runner therefore
marks its output as a pilot and refuses a ``--final`` claim unless an independent
truth file is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from .baselines import axe_findings, deterministic_findings
from .contracts import DetectorObservation
from .data import discover_cached_graphs, load_cached_graph
from .fusion import FusionPolicy, RegistryRouter, fuse_observations, validate_fused_provenance
from .catalog import CriterionRegistry
from .contracts import Candidate
from .models import ModelConfig, build_model
from .rules import rule_metadata
from .schema import FeatureContract


Pair = tuple[str, str]  # site, criterion
RulePair = tuple[str, str]  # site, rule


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class MethodOutput:
    scores: dict[Pair, float]
    predictions: set[Pair]
    coverage: set[Pair]
    latency_seconds: float = 0.0
    collection_failures: int = 0
    review: set[Pair] | None = None


def _criterion_ids(rule_id: str) -> tuple[str, ...]:
    try:
        return tuple(rule_metadata(rule_id)["wcag_ids"])
    except KeyError:
        return ()


def _to_criterion_output(
    sites: list[str],
    rules: set[str],
    rule_scores: dict[RulePair, float],
    rule_predictions: set[RulePair],
    supported_rules: set[str],
    latency: float = 0.0,
) -> MethodOutput:
    supported_criteria = {criterion for rule in supported_rules for criterion in _criterion_ids(rule)}
    scores: dict[Pair, float] = {}
    predictions: set[Pair] = set()
    coverage = {(site, criterion) for site in sites for criterion in supported_criteria}
    for site, criterion in coverage:
        applicable = [rule for rule in rules if criterion in _criterion_ids(rule) and rule in supported_rules]
        scores[(site, criterion)] = max((rule_scores.get((site, rule), 0.0) for rule in applicable), default=0.0)
        if any((site, rule) in rule_predictions for rule in applicable):
            predictions.add((site, criterion))
    return MethodOutput(scores, predictions, coverage, latency)


def _binary_counts(predicted: set[Pair], truth: set[Pair], universe: set[Pair]) -> dict[str, float]:
    predicted = predicted & universe; truth = truth & universe
    tp = len(predicted & truth); fp = len(predicted - truth); fn = len(truth - predicted); tn = len(universe - predicted - truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def _expected_calibration_error(labels: list[int], scores: list[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = len(labels); value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, score in enumerate(scores) if low <= score < high or (index == bins - 1 and score == 1)]
        if members:
            accuracy = sum(labels[i] for i in members) / len(members)
            confidence = sum(scores[i] for i in members) / len(members)
            value += len(members) / total * abs(accuracy - confidence)
    return value


def evaluate_method(output: MethodOutput, truth: set[Pair], universe: set[Pair]) -> dict:
    judged = output.coverage & universe
    counts = _binary_counts(output.predictions, truth, judged)
    ordered = sorted(judged)
    labels = [int(pair in truth) for pair in ordered]
    scores = [float(output.scores.get(pair, 0.0)) for pair in ordered]
    pr_auc = float(average_precision_score(labels, scores)) if labels and any(labels) else 0.0
    curve_precision, curve_recall, curve_thresholds = precision_recall_curve(labels, scores) if labels and any(labels) else ([0.0], [0.0], [])
    brier = float(brier_score_loss(labels, scores)) if labels else 0.0
    per_criterion = {}
    for criterion in sorted({criterion for _, criterion in universe}):
        subset = {pair for pair in judged if pair[1] == criterion}
        if subset:
            entry = _binary_counts(output.predictions, truth, subset)
            criterion_order = sorted(subset)
            criterion_labels = [int(pair in truth) for pair in criterion_order]
            criterion_scores = [float(output.scores.get(pair, 0.0)) for pair in criterion_order]
            entry["pr_auc"] = float(average_precision_score(criterion_labels, criterion_scores)) if any(criterion_labels) else 0.0
            per_criterion[criterion] = entry
    site_f1 = []
    for site in sorted({site for site, _ in universe}):
        subset = {pair for pair in judged if pair[0] == site}
        if subset:
            site_f1.append(_binary_counts(output.predictions, truth, subset)["f1"])
    supported_f1 = [entry["f1"] for entry in per_criterion.values() if entry["tp"] + entry["fn"]]
    supported_precision = [entry["precision"] for entry in per_criterion.values() if entry["tp"] + entry["fn"]]
    supported_recall = [entry["recall"] for entry in per_criterion.values() if entry["tp"] + entry["fn"]]
    supported_pr_auc = [entry["pr_auc"] for entry in per_criterion.values() if entry["tp"] + entry["fn"]]
    site_count = len({site for site, _ in universe})
    review = output.review or set()
    return {
        "unit": "site_criterion_pair",
        **counts,
        "micro_pr_auc": pr_auc,
        "micro_pr_curve": {
            "precision": [float(value) for value in curve_precision],
            "recall": [float(value) for value in curve_recall],
            "thresholds": [float(value) for value in curve_thresholds],
        },
        "macro_precision_supported": sum(supported_precision) / len(supported_precision) if supported_precision else 0.0,
        "macro_recall_supported": sum(supported_recall) / len(supported_recall) if supported_recall else 0.0,
        "macro_f1_supported": sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0,
        "macro_pr_auc_supported": sum(supported_pr_auc) / len(supported_pr_auc) if supported_pr_auc else 0.0,
        "site_macro_f1": sum(site_f1) / len(site_f1) if site_f1 else 0.0,
        "false_positives_per_page": counts["fp"] / max(1, site_count),
        "brier_score": brier,
        "expected_calibration_error_10_bin": _expected_calibration_error(labels, scores),
        "coverage": len(judged) / max(1, len(universe)),
        "manual_review_rate": len(review & universe) / max(1, len(universe)),
        "collection_failures": output.collection_failures,
        "latency_seconds": output.latency_seconds,
        "per_criterion": per_criterion,
    }


def bootstrap_ci(
    output: MethodOutput,
    truth: set[Pair],
    universe: set[Pair],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    sites = sorted({site for site, _ in universe}); rng = random.Random(seed)
    values = {"precision": [], "recall": [], "f1": [], "site_macro_f1": []}
    for _ in range(samples):
        chosen = [rng.choice(sites) for _ in sites]
        # Replicate sites with synthetic suffixes so repeated draws retain weight.
        sampled_universe = set(); sampled_truth = set(); sampled_predictions = set()
        for draw, site in enumerate(chosen):
            alias = f"{draw}:{site}"
            for pair in universe:
                if pair[0] != site or pair not in output.coverage:
                    continue
                copied = (alias, pair[1]); sampled_universe.add(copied)
                if pair in truth: sampled_truth.add(copied)
                if pair in output.predictions: sampled_predictions.add(copied)
        counts = _binary_counts(sampled_predictions, sampled_truth, sampled_universe)
        for metric in ("precision", "recall", "f1"): values[metric].append(counts[metric])
        drawn_site_f1 = []
        for draw, site in enumerate(chosen):
            alias = f"{draw}:{site}"; subset = {pair for pair in sampled_universe if pair[0] == alias}
            if subset: drawn_site_f1.append(_binary_counts(sampled_predictions, sampled_truth, subset)["f1"])
        values["site_macro_f1"].append(sum(drawn_site_f1) / len(drawn_site_f1) if drawn_site_f1 else 0.0)
    result = {}
    for metric, entries in values.items():
        ordered = sorted(entries)
        low = ordered[max(0, round(0.025 * (len(ordered) - 1)))]
        high = ordered[min(len(ordered) - 1, round(0.975 * (len(ordered) - 1)))]
        result[metric] = [low, high]
    return result


def _model_rule_output(phase5_dir: Path, cache_dir: Path, split: dict, view: str, architecture: str, device: str) -> tuple[dict[RulePair, float], set[RulePair], set[str], float]:
    run_dir = phase5_dir / view / architecture
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    calibration = json.loads((run_dir / "calibration.json").read_text())
    config = ModelConfig.from_dict(checkpoint["model_config"])
    contract = FeatureContract.from_dict(checkpoint["feature_contract"])
    model = build_model(config).to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    rule_ids = list(checkpoint["rule_ids"]); indices = list(checkpoint["rule_indices"])
    paths = discover_cached_graphs(cache_dir, view); scores = {}; predicted = set()
    started = time.perf_counter()
    with torch.no_grad():
        for site in split["test"]:
            graph = load_cached_graph(paths[site], graph_source=view, rule_indices=indices, require_labels=False)
            contract.validate(graph); graph = graph.to(device)
            probabilities = torch.sigmoid(model(graph.x, graph.edge_index, graph.tag_indices)).cpu()
            valid = getattr(graph, "label_mask", torch.ones(graph.num_nodes, dtype=torch.bool, device=device)).cpu().bool()
            for local_index, rule in enumerate(rule_ids):
                score = float(probabilities[valid, local_index].max()) if valid.any() else 0.0
                scores[(site, rule)] = score
                threshold = float(calibration["recommended"]["rule_thresholds"][rule])
                if score >= threshold: predicted.add((site, rule))
    return scores, predicted, set(rule_ids), time.perf_counter() - started


def _union(outputs: Iterable[MethodOutput], universe: set[Pair]) -> MethodOutput:
    outputs = list(outputs); scores = {}; predictions = set(); coverage = set(); latency = 0.0
    for output in outputs:
        predictions |= output.predictions; coverage |= output.coverage; latency += output.latency_seconds
        for pair, score in output.scores.items(): scores[pair] = max(scores.get(pair, 0.0), score)
    return MethodOutput(scores, predictions, coverage & universe, latency)


def _paired_f1_difference(left: MethodOutput, right: MethodOutput, truth: set[Pair], universe: set[Pair], samples: int, seed: int) -> list[float]:
    sites = sorted({site for site, _ in universe}); rng = random.Random(seed); differences = []
    common = left.coverage & right.coverage & universe
    for _ in range(samples):
        chosen = [rng.choice(sites) for _ in sites]; lpred = set(); rpred = set(); sampled_truth = set(); sampled = set()
        for draw, site in enumerate(chosen):
            for pair in common:
                if pair[0] != site: continue
                copy = (f"{draw}:{site}", pair[1]); sampled.add(copy)
                if pair in truth: sampled_truth.add(copy)
                if pair in left.predictions: lpred.add(copy)
                if pair in right.predictions: rpred.add(copy)
        differences.append(_binary_counts(lpred, sampled_truth, sampled)["f1"] - _binary_counts(rpred, sampled_truth, sampled)["f1"])
    ordered = sorted(differences)
    return [ordered[round(0.025 * (samples - 1))], ordered[round(0.975 * (samples - 1))]]


def run_study(args: argparse.Namespace) -> dict:
    if args.final and args.truth_source != "independent_manual":
        raise ValueError("Final testing requires --truth-source independent_manual")
    truth_file = getattr(args, "truth_file", None)
    if args.truth_source == "independent_manual" and not truth_file:
        raise ValueError("Independent manual truth requires --truth-file")
    output_path = args.output_dir / "phase_7_detection_study.json"
    if args.final and output_path.exists():
        raise FileExistsError(f"Frozen final result already exists: {output_path}")
    phase5 = args.phase5_dir; split = json.loads((phase5 / "pilot_split.json").read_text())
    sites = list(split["test"])
    model_raw = {}
    supported_rules = set()
    for view in args.views:
        for architecture in args.architectures:
            raw = _model_rule_output(phase5, args.cache_dir, split, view, architecture, args.device)
            model_raw[(view, architecture)] = raw; supported_rules |= raw[2]
    criteria = sorted({criterion for rule in supported_rules for criterion in _criterion_ids(rule)})
    universe = {(site, criterion) for site in sites for criterion in criteria}

    truth_rules = set(); axe_scores = {}; deterministic_scores = {}; deterministic_predictions = set()
    axe_started = time.perf_counter()
    for site in sites:
        site_dir = args.corpus_dir / site
        for finding in axe_findings(site_dir):
            if finding.rule_id in supported_rules: truth_rules.add((site, finding.rule_id)); axe_scores[(site, finding.rule_id)] = 1.0
    axe_latency = time.perf_counter() - axe_started
    deterministic_started = time.perf_counter()
    for site in sites:
        site_dir = args.corpus_dir / site
        for finding in deterministic_findings(site_dir / "0.html", site_id=site):
            if finding.rule_id in supported_rules:
                deterministic_predictions.add((site, finding.rule_id)); deterministic_scores[(site, finding.rule_id)] = 1.0
    deterministic_latency = time.perf_counter() - deterministic_started
    axe_output = _to_criterion_output(sites, supported_rules, axe_scores, truth_rules, supported_rules, axe_latency)
    if args.truth_source == "independent_manual":
        manual = json.loads(Path(truth_file).read_text(encoding="utf-8"))
        truth = {
            (str(item["site_id"]), str(item["criterion_id"]))
            for item in manual.get("pairs", []) if item.get("status") == "fail"
        }
        unknown = truth - universe
        if unknown:
            raise ValueError(f"Independent truth contains pairs outside the frozen study universe: {sorted(unknown)[:3]}")
    else:
        truth = set(axe_output.predictions)
    deterministic = _to_criterion_output(sites, supported_rules, deterministic_scores, deterministic_predictions, supported_rules, deterministic_latency)

    methods = {"axe_alone": axe_output, "custom_deterministic": deterministic}
    methods["axe_plus_custom"] = _union((axe_output, deterministic), universe)
    for architecture in args.architectures:
        components = []
        for view in args.views:
            scores, predicted, rules, latency = model_raw[(view, architecture)]
            components.append(_to_criterion_output(sites, supported_rules, scores, predicted, rules, latency))
        methods[f"{architecture}_specialist"] = _union(components, universe)
    visual_scores, visual_predicted, visual_rules, visual_latency = model_raw[("rendered-visual", "mlp")]
    methods["visual_specialist"] = _to_criterion_output(sites, supported_rules, visual_scores, visual_predicted, visual_rules, visual_latency)
    methods["interaction_specialist"] = MethodOutput({}, set(), set())
    specialist_outputs = [methods[name] for name in ("axe_alone", "custom_deterministic", "mlp_specialist", "graphsage_specialist", "gat_specialist", "visual_specialist")]
    uncalibrated_components = [axe_output, deterministic]
    for (_, _), (scores, _, rules, latency) in model_raw.items():
        predicted_at_half = {pair for pair, score in scores.items() if score >= 0.5}
        uncalibrated_components.append(_to_criterion_output(sites, supported_rules, scores, predicted_at_half, rules, latency))
    methods["uncalibrated_union"] = _union(uncalibrated_components, universe)

    # Create one observation per detector/rule. Missing interaction support is
    # explicit and cannot become a pass during fusion.
    registry = CriterionRegistry.load(args.registry)
    router = RegistryRouter(registry)
    routing_decisions = {}
    available_detectors = {"axe", "deterministic-html", "mlp", "graphsage", "gat", "visual"}
    for criterion in criteria:
        decision = router.route(
            Candidate(f"study:{criterion}", "held-out", (criterion,)),
            available_detectors=available_detectors,
        )[0]
        routing_decisions[criterion] = {
            "detector_ids": list(decision.detector_ids), "missing_detectors": list(decision.missing_detectors),
            "status": decision.status, "reason": decision.reason,
        }
    observations = []
    sources = [("axe", axe_scores, truth_rules, supported_rules, {rule: 0.5 for rule in supported_rules}),
               ("deterministic", deterministic_scores, deterministic_predictions, supported_rules, {rule: 0.5 for rule in supported_rules})]
    for (view, architecture), (scores, predicted, rules, _) in model_raw.items():
        calibration = json.loads((phase5 / view / architecture / "calibration.json").read_text())
        sources.append((f"{view}:{architecture}", scores, predicted, rules, calibration["recommended"]["rule_thresholds"]))
    source_thresholds = {}
    for source, scores, predicted, rules, thresholds in sources:
        for site in sites:
            for rule in rules:
                source_route_id = "axe" if source == "axe" else "deterministic-html" if source == "deterministic" else source.split(":")[-1]
                detector_id = f"{source}:{rule}"; source_thresholds[detector_id] = float(thresholds[rule])
                raw_score = float(scores.get((site, rule), 0.0)); failed = (site, rule) in predicted
                for criterion in _criterion_ids(rule):
                    if criterion not in criteria: continue
                    if source_route_id not in routing_decisions[criterion]["detector_ids"]:
                        continue
                    observations.append(DetectorObservation(
                        observation_id=hashlib.sha256(f"{site}|{criterion}|{rule}|{source}".encode()).hexdigest()[:20],
                        site_id=site, criterion_id=criterion, detector_id=detector_id,
                        status="fail" if failed else "pass", confidence=raw_score if failed else 1.0 - raw_score,
                        rule_id=rule, target_id=rule,
                        evidence={"raw_violation_probability": raw_score, "threshold": float(thresholds[rule]), "source": source},
                    ))
    policy = FusionPolicy(source_thresholds, fail_threshold=args.fail_threshold, review_threshold=args.review_threshold)
    fusion_started = time.perf_counter()
    fused = fuse_observations(observations, policy); validate_fused_provenance(fused)
    fusion_latency = time.perf_counter() - fusion_started
    fused_scores = {}; fused_predictions = set(); fused_review = set(); fused_coverage = set()
    for finding in fused:
        pair = (finding.site_id, finding.criterion_id); fused_coverage.add(pair)
        fused_scores[pair] = max(fused_scores.get(pair, 0.0), finding.confidence if finding.status in {"fail", "needs_review"} else 0.0)
        if finding.status == "fail": fused_predictions.add(pair)
        elif finding.status == "needs_review": fused_review.add(pair)
    methods["calibrated_routed_fusion"] = MethodOutput(fused_scores, fused_predictions, fused_coverage, fusion_latency, review=fused_review)

    results = {}
    for name, output in methods.items():
        results[name] = evaluate_method(output, truth, universe)
        results[name]["bootstrap_95_ci"] = bootstrap_ci(output, truth, universe, samples=args.bootstrap_samples, seed=args.seed)
    candidate = _union(specialist_outputs, universe)
    ceiling = _binary_counts(candidate.predictions, truth, universe)["recall"]
    comparisons = {
        "graphsage_minus_mlp_f1": _paired_f1_difference(methods["graphsage_specialist"], methods["mlp_specialist"], truth, universe, args.bootstrap_samples, args.seed),
        "gat_minus_mlp_f1": _paired_f1_difference(methods["gat_specialist"], methods["mlp_specialist"], truth, universe, args.bootstrap_samples, args.seed + 1),
        "fusion_minus_uncalibrated_f1": _paired_f1_difference(methods["calibrated_routed_fusion"], methods["uncalibrated_union"], truth, universe, args.bootstrap_samples, args.seed + 2),
    }
    report = {
        "schema_version": 1,
        "phase": "7",
        "study_status": "final" if args.final else "weak_label_pilot",
        "final_test_consumed": bool(args.final),
        "truth_source": args.truth_source,
        "truth_limitation": None if args.truth_source == "independent_manual" else "axe is both weak-label truth and a required baseline; axe metrics are circular and not evidence of generalisation",
        "split_hash": split.get("split_hash"),
        "sites": sites,
        "site_count": len(sites),
        "supported_rules": sorted(supported_rules),
        "criteria": criteria,
        "unit": "site_criterion_pair",
        "candidate_generation_ceiling": ceiling,
        "candidate_ceiling_limitation": "The axe candidate source also supplies weak-label truth in this pilot, so this ceiling is circular.",
        "frozen_policy": policy.to_dict(),
        "routing_decisions": routing_decisions,
        "methods": results,
        "paired_f1_difference_95_ci": comparisons,
        "ablations": {
            "routing": {"uncalibrated_union": results["uncalibrated_union"], "calibrated_routed_fusion": results["calibrated_routed_fusion"]},
            "architecture": {name: results[name] for name in ("mlp_specialist", "graphsage_specialist", "gat_specialist")},
            "view": {"visual_specialist": results["visual_specialist"], "interaction_specialist": results["interaction_specialist"]},
            "calibration": "calibrated fusion uses validation-frozen per-source/rule thresholds; union uses source decisions without fusion abstention",
        },
        "unsupported": {"interaction_specialist": "No Phase 5 interaction/state model or independently labelled trace corpus exists."},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_6_fusion_policy.json").write_text(json.dumps(policy.to_dict(), indent=2), encoding="utf-8")
    (args.output_dir / "phase_6_routing_decisions.json").write_text(json.dumps(routing_decisions, indent=2), encoding="utf-8")
    (args.output_dir / "fused_findings.json").write_text(json.dumps([item.to_dict() for item in fused], indent=2), encoding="utf-8")
    (args.output_dir / "phase_7_detection_study.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1, "phase": "6-7", "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples, "device": args.device,
        "inputs": {
            "phase5_dir": str(args.phase5_dir), "cache_dir": str(args.cache_dir),
            "corpus_dir": str(args.corpus_dir), "registry": str(args.registry),
            "registry_sha256": _sha256(args.registry),
            "pilot_split_sha256": _sha256(args.phase5_dir / "pilot_split.json"),
            "truth_source": args.truth_source,
            "truth_file": str(truth_file) if truth_file else None,
            "truth_file_sha256": _sha256(Path(truth_file)) if truth_file else None,
        },
        "frozen_policy": policy.to_dict(), "final_test_consumed": bool(args.final),
        "outputs": ["phase_6_fusion_policy.json", "phase_6_routing_decisions.json", "fused_findings.json", "phase_7_detection_study.json"],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("../configs/wcag_criteria.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=["dom", "a11y-tree", "rendered-visual"])
    parser.add_argument("--architectures", nargs="+", default=["mlp", "graphsage", "gat"])
    parser.add_argument("--truth-source", choices=["axe_weak_labels", "independent_manual"], default="axe_weak_labels")
    parser.add_argument("--truth-file", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--fail-threshold", type=float, default=0.65)
    parser.add_argument("--review-threshold", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    report = run_study(parse_args())
    print(json.dumps({
        "study_status": report["study_status"], "site_count": report["site_count"],
        "criteria": report["criteria"], "output": "phase_7_detection_study.json",
    }, indent=2))


if __name__ == "__main__":
    main()
