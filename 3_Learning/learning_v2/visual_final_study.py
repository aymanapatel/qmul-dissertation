"""Evaluate frozen visual ablations against independent site-criterion truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import torch

from .data import discover_cached_graphs, load_cached_graph, load_split
from .models import ModelConfig, build_model
from .schema import FeatureContract
from .study import MethodOutput, _binary_counts, _load_independent_truth, bootstrap_ci, evaluate_method


Pair = tuple[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_output(run_dir: Path, cache_paths: dict[str, Path], sites: list[str], variant: str, device: str) -> MethodOutput:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(checkpoint["model_config"])
    contract = FeatureContract.from_dict(checkpoint["feature_contract"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    rule_ids = list(checkpoint["rule_ids"]); indices = list(checkpoint["rule_indices"])
    scores: dict[Pair, float] = {}; predictions: set[Pair] = set(); coverage: set[Pair] = set()
    with torch.no_grad():
        for site in sites:
            graph = load_cached_graph(
                cache_paths[site], graph_source="rendered-visual", rule_indices=indices,
                require_labels=False, feature_variant=variant,
            )
            contract.validate(graph); graph = graph.to(device)
            probabilities = torch.sigmoid(model(graph.x, graph.edge_index, graph.tag_indices)).cpu()
            valid = getattr(graph, "label_mask", torch.ones(graph.num_nodes, dtype=torch.bool, device=device)).cpu().bool()
            for local_index, rule_id in enumerate(rule_ids):
                # The frozen rendered specialist currently owns color-contrast,
                # whose criterion-level unit is 1.4.3.
                if rule_id != "color-contrast":
                    continue
                pair = (site, "1.4.3"); coverage.add(pair)
                score = float(probabilities[valid, local_index].max()) if valid.any() else 0.0
                scores[pair] = score
                threshold = float(calibration["recommended"]["rule_thresholds"][rule_id])
                if score >= threshold:
                    predictions.add(pair)
    return MethodOutput(scores=scores, predictions=predictions, coverage=coverage)


def _resampled_metric(output: MethodOutput, truth: set[Pair], universe: set[Pair], chosen_sites: list[str], metric: str) -> float:
    predicted = set(); copied_truth = set(); copied_universe = set()
    for draw, site in enumerate(chosen_sites):
        for pair in universe:
            if pair[0] != site:
                continue
            copied = (f"{draw}:{site}", pair[1]); copied_universe.add(copied)
            if pair in truth: copied_truth.add(copied)
            if pair in output.predictions: predicted.add(copied)
    return float(_binary_counts(predicted, copied_truth, copied_universe)[metric])


def hierarchical_paired_interval(
    full: list[MethodOutput], structure: list[MethodOutput], truth: set[Pair], universe: set[Pair],
    *, metric: str, samples: int, seed: int,
) -> list[float]:
    if len(full) != len(structure) or not full:
        raise ValueError("Paired visual outputs require the same non-zero seed count")
    sites = sorted({site for site, _ in universe}); rng = random.Random(seed); values = []
    for _ in range(samples):
        chosen_sites = [rng.choice(sites) for _ in sites]
        chosen_seeds = [rng.randrange(len(full)) for _ in full]
        differences = [
            _resampled_metric(full[index], truth, universe, chosen_sites, metric)
            - _resampled_metric(structure[index], truth, universe, chosen_sites, metric)
            for index in chosen_seeds
        ]
        values.append(statistics.mean(differences))
    values.sort()
    return [values[round(0.025 * (samples - 1))], values[round(0.975 * (samples - 1))]]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output_dir / "final_visual_ablation_study.json"
    if output_path.exists():
        raise FileExistsError(f"Frozen final result already exists: {output_path}")
    split = load_split(args.split); sites = list(split["test"])
    universe = {(site, "1.4.3") for site in sites}
    payload = json.loads(args.truth_file.read_text(encoding="utf-8"))
    file_universe = {(str(item["site_id"]), str(item["criterion_id"])) for item in payload.get("pairs", [])}
    all_truth, annotation = _load_independent_truth(args.truth_file, file_universe)
    if not universe <= file_universe:
        raise ValueError(f"Independent truth is missing {len(universe - file_universe)} visual test pairs")
    truth = all_truth & universe
    cache_paths = discover_cached_graphs(args.cache_dir, "rendered-visual")
    missing = set(sites) - set(cache_paths)
    if missing:
        raise ValueError(f"Visual cache misses frozen test sites: {sorted(missing)}")
    outputs: dict[tuple[int, str, str], MethodOutput] = {}
    rows = []
    for seed in args.seeds:
        for architecture in args.architectures:
            for variant in ("full", "structure_only"):
                suffix = Path("rendered-visual") / architecture if variant == "full" else Path("rendered-visual") / variant / architecture
                run_dir = args.ablation_dir / f"seed_{seed}" / suffix
                output = _run_output(run_dir, cache_paths, sites, variant, args.device)
                outputs[(seed, architecture, variant)] = output
                rows.append({
                    "seed": seed, "architecture": architecture, "variant": variant,
                    "metrics": evaluate_method(output, truth, universe, ["1.4.3"]),
                    "site_bootstrap_95_ci": bootstrap_ci(output, truth, universe, samples=args.bootstrap_samples, seed=args.seed + seed),
                })
    comparisons = {}
    for architecture in args.architectures:
        full = [outputs[(seed, architecture, "full")] for seed in args.seeds]
        structure = [outputs[(seed, architecture, "structure_only")] for seed in args.seeds]
        comparisons[architecture] = {
            metric: {
                "full_minus_structure_only_hierarchical_95_ci": hierarchical_paired_interval(
                    full, structure, truth, universe, metric=metric,
                    samples=args.bootstrap_samples, seed=args.seed + index,
                )
            }
            for index, metric in enumerate(("precision", "recall", "f1"))
        }
        comparisons[architecture]["visual_advantage_established"] = comparisons[architecture]["recall"]["full_minus_structure_only_hierarchical_95_ci"][0] > 0
    report = {
        "schema_version": 1, "status": "final_independent_visual_ablation",
        "truth_source": "independent_manual", "unit": "site_criterion_pair",
        "criterion": "1.4.3", "site_count": len(sites), "positive_sites": len(truth),
        "annotation": annotation, "split_hash": split.get("split_hash"), "seeds": args.seeds,
        "rows": rows, "paired_comparisons": comparisons,
        "claim_boundary": "A visual advantage is established per architecture only when the hierarchical paired recall interval excludes zero.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "inputs": {
            "ablation_dir": str(args.ablation_dir.resolve()), "cache_dir": str(args.cache_dir.resolve()),
            "split": str(args.split.resolve()), "split_sha256": _sha256(args.split),
            "truth_file": str(args.truth_file.resolve()), "truth_sha256": _sha256(args.truth_file),
        },
        "config": {"seeds": args.seeds, "architectures": args.architectures, "bootstrap_samples": args.bootstrap_samples, "seed": args.seed},
        "outputs": {output_path.name: _sha256(output_path)},
    }, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--truth-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--architectures", nargs="+", default=["mlp", "graphsage"])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "site_count": report["site_count"], "positive_sites": report["positive_sites"]}, indent=2))


if __name__ == "__main__":
    main()
