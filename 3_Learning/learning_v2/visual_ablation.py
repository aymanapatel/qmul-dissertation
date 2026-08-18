"""Train controlled rendered-cue and spatial-edge ablations on one frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch

from .feature_layout import MINIMUM_RENDERED_FEATURE_DIM, RENDERED_VISUAL_FEATURE_VERSION
from .data import discover_cached_graphs, load_split
from .experiment import _train_one, validate_rule_support
from .rules import rules_for_source


VARIANTS = ("full", "without_visual_features", "without_spatial_edges", "structure_only")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir(output_dir: Path, seed: int, variant: str, architecture: str) -> Path:
    base = output_dir / f"seed_{seed}" / "rendered-visual"
    return base / (architecture if variant == "full" else f"{variant}/{architecture}")


def _completed_result(
    args: argparse.Namespace, seed: int, variant: str, architecture: str,
    rule_ids: list[str],
) -> dict | None:
    """Load an exact completed run, or return None for an interrupted run."""
    run_dir = _run_dir(args.output_dir, seed, variant, architecture)
    required = {
        "checkpoint": run_dir / "best_model.pt",
        "history": run_dir / "history.json",
        "calibration": run_dir / "calibration.json",
        "metrics": run_dir / "test_metrics.json",
        "manifest": run_dir / "manifest.json",
    }
    if not all(path.is_file() for path in required.values()):
        return None

    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    recorded_rules = [item["rule_id"] for item in manifest.get("rules", [])]
    expected = {
        "graph_source": "rendered-visual",
        "architecture": architecture,
        "feature_variant": variant,
        "split_mode": "governed",
        "split_sha256": _sha256(args.split),
        "rules": rule_ids,
    }
    observed = {
        "graph_source": manifest.get("graph_source"),
        "architecture": manifest.get("architecture"),
        "feature_variant": manifest.get("feature_variant"),
        "split_mode": manifest.get("split_mode"),
        "split_sha256": manifest.get("split_provenance", {}).get("sha256"),
        "rules": recorded_rules,
    }
    recorded_config = manifest.get("training_config", {})
    config_keys = (
        "epochs", "patience", "batch_size", "hidden", "layers", "heads",
        "dropout", "lr", "negative_ratio", "minimum_negatives",
        "precision_floor", "device",
    )
    config_mismatches = {
        key: {"expected": getattr(args, key), "observed": recorded_config.get(key)}
        for key in config_keys if recorded_config.get(key) != getattr(args, key)
    }
    if observed != expected or config_mismatches:
        raise ValueError(
            f"Cannot resume incompatible completed run at {run_dir}: "
            f"identity_expected={expected}, identity_observed={observed}, "
            f"config_mismatches={config_mismatches}. Use the original arguments or a new output directory."
        )

    metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
    calibration = json.loads(required["calibration"].read_text(encoding="utf-8"))
    unmet_precision_floor = [
        name for name, result in {
            "node": calibration.get("node", {}),
            "page": calibration.get("page", {}),
            **calibration.get("rules", {}),
        }.items() if result and not result.get("precision_floor_met", False)
    ]
    return {
        "view": "rendered-visual",
        "architecture": architecture,
        "feature_variant": variant,
        "rules": rule_ids,
        "best_epoch": manifest["best_epoch"],
        "stopped_early": manifest["stopped_early"],
        "calibration_floor_policy": "validation_f1_fallback",
        "unmet_precision_floor": unmet_precision_floor,
        "test": metrics,
    }


def _check_cache(paths: dict[str, Path], sites: set[str]) -> None:
    missing = sites - set(paths)
    if missing:
        raise ValueError(f"Visual cache is missing {len(missing)} frozen split sites")
    failures = []
    spatial_edges = 0
    for site in sorted(sites):
        raw = torch.load(paths[site], map_location="cpu", weights_only=False)["data"]
        if int(getattr(raw, "rendered_visual_feature_version", 0)) < RENDERED_VISUAL_FEATURE_VERSION:
            failures.append(f"{site}:feature_version")
        if int(raw.x.shape[1]) < MINIMUM_RENDERED_FEATURE_DIM:
            failures.append(f"{site}:feature_width")
        if not hasattr(raw, "edge_type"):
            failures.append(f"{site}:typed_spatial_edges")
        else:
            spatial_edges += int((raw.edge_type == 2).sum())
    if failures:
        raise ValueError(f"Visual ablation contract failed ({len(failures)} errors); examples: {failures[:5]}")
    # A valid small or hidden page can legitimately produce no spatial edges.
    # The ablation contract therefore requires typed edges on every graph and
    # at least one spatial relation across the frozen cohort, not on each page.
    if spatial_edges == 0:
        raise ValueError("Visual ablation contract failed: frozen cohort contains no spatial edges")


def _percentile_interval(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = [statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)]
    means.sort()
    return [means[round(0.025 * (samples - 1))], means[round(0.975 * (samples - 1))]]


def run(args: argparse.Namespace) -> dict:
    paths = discover_cached_graphs(args.cache_dir, "rendered-visual")
    split = load_split(args.split)
    sites = set(split["train"]) | set(split["val"]) | set(split["test"])
    _check_cache(paths, sites)
    all_ids, all_indices = rules_for_source("rendered-visual")
    requested = list(args.rule_ids)
    unknown = sorted(set(requested) - set(all_ids))
    if unknown:
        raise ValueError(f"Requested rules are not owned by rendered-visual: {unknown}")
    selected_pairs = [
        (rule_id, rule_index)
        for rule_id, rule_index in zip(all_ids, all_indices)
        if rule_id in requested
    ]
    if not selected_pairs:
        raise ValueError("At least one rendered-visual --rule-ids value is required")
    selected_ids, selected_indices = map(tuple, zip(*selected_pairs))
    rule_ids, rule_indices, support = validate_rule_support(
        paths, split, "rendered-visual", selected_ids, selected_indices,
        min_train_positive_sites=args.min_train_positive_sites,
        min_val_positive_sites=args.min_val_positive_sites,
    )
    args.split_mode = "governed" if split.get("selection_mode") == "governed" else "pilot"
    args.allow_zero_validation_metric = False
    args.allow_unmet_precision_floor = False
    # Ablated variants are deliberately weakened and may be incapable of
    # satisfying the deployable model's precision floor. When that happens,
    # use the calibrator's validation-only maximum-F1 threshold and retain the
    # failed floor as an explicit result. Final Phase 5 training stays strict.
    args.calibration_floor_policy = "validation_f1_fallback"
    results = []
    resumed_runs = 0
    original_seed = args.seed
    for seed in args.seeds:
        args.seed = seed
        seed_dir = args.output_dir / f"seed_{seed}"
        for variant in args.variants:
            for architecture in args.architectures:
                if args.resume:
                    completed = _completed_result(args, seed, variant, architecture, rule_ids)
                    if completed is not None:
                        print(
                            f"Resuming: skipped completed visual ablation seed={seed} "
                            f"variant={variant} architecture={architecture}",
                            flush=True,
                        )
                        results.append(completed)
                        resumed_runs += 1
                        continue
                print(f"Training visual ablation seed={seed} variant={variant} architecture={architecture}", flush=True)
                results.append(_train_one(
                    "rendered-visual", architecture, split, paths, rule_ids, rule_indices,
                    seed_dir, args, feature_variant=variant,
                ))
    args.seed = original_seed
    grouped = {}
    test_page_count = len(split["test"])
    for variant in args.variants:
        grouped[variant] = {}
        for architecture in args.architectures:
            entries = [item["test"] for item in results if item["feature_variant"] == variant and item["architecture"] == architecture]
            metric_values = {
                metric: [float(item[metric]) for item in entries]
                for metric in ("rule_precision", "rule_recall", "rule_f1", "rule_macro_f1_supported")
            }
            metric_values["false_positives_per_page"] = [
                float(item["rule_fp"]) / max(1, test_page_count) for item in entries
            ]
            grouped[variant][architecture] = {
                metric: {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
                for metric, values in metric_values.items()
            }
    comparisons = {}
    baselines = {
        "full_minus_structure_only": ("full", "structure_only"),
        "visual_features_added": ("full", "without_visual_features"),
        "spatial_edges_added": ("full", "without_spatial_edges"),
    }
    for architecture in args.architectures:
        comparisons[architecture] = {}
        for name, (left, right) in baselines.items():
            comparisons[architecture][name] = {}
            for metric in ("rule_recall", "rule_f1", "false_positives_per_page"):
                def values_for(variant: str) -> list[float]:
                    entries = [
                        item["test"] for item in results
                        if item["feature_variant"] == variant and item["architecture"] == architecture
                    ]
                    if metric == "false_positives_per_page":
                        return [float(item["rule_fp"]) / max(1, test_page_count) for item in entries]
                    return [float(item[metric]) for item in entries]
                differences = [a - b for a, b in zip(values_for(left), values_for(right))]
                comparisons[architecture][name][metric] = {
                    "paired_seed_differences": differences,
                    "mean_difference": statistics.mean(differences),
                    "bootstrap_95_ci_over_seeds": _percentile_interval(differences, args.bootstrap_samples, args.seed),
                }
    report = {
        "schema_version": 2,
        "phase": 5,
        "study": "rendered_visual_cue_ablation",
        "unit": "node_rule_and_page_rule",
        "split_hash": split.get("split_hash"),
        "rule_ids": rule_ids,
        "seeds": args.seeds,
        "completed_runs_reused": resumed_runs,
        "calibration_floor_policy": "validation_f1_fallback_when_predeclared_floor_is_unattainable",
        "variants": {
            "full": "structural, text, rendered visual features, and spatial edges",
            "without_visual_features": "visual feature tail zeroed; spatial edges retained",
            "without_spatial_edges": "visual features retained; typed spatial edges removed",
            "structure_only": "visual feature tail zeroed and typed spatial edges removed",
        },
        "rule_support": support,
        "aggregate": grouped,
        "paired_comparisons": comparisons,
        "runs": results,
        "interpretation": "Visual-cue benefit is full minus structure_only under identical sites, labels, architecture, budget, and seeds. Seed-bootstrap intervals are exploratory and are not substitutes for site-bootstrap intervals on an independently labelled final benchmark.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "visual_ablation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "inputs": {"cache_dir": str(args.cache_dir.resolve()), "split": str(args.split.resolve()), "split_sha256": _sha256(args.split)},
        "config": {"rule_ids": rule_ids, "seeds": args.seeds, "variants": args.variants, "architectures": args.architectures, "epochs": args.epochs, "bootstrap_samples": args.bootstrap_samples},
        "outputs": {"visual_ablation.json": _sha256(report_path)},
    }, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rule-ids", nargs="+", default=["color-contrast"],
        help="Predeclared rendered rule set; defaults to the Phase 5 visual specialist rule",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--architectures", nargs="+", choices=("mlp", "graphsage", "gat"), default=["mlp", "graphsage"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-train-positive-sites", type=int, default=1)
    parser.add_argument("--min-val-positive-sites", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64); parser.add_argument("--layers", type=int, default=2); parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--negative-ratio", type=float, default=8.0); parser.add_argument("--minimum-negatives", type=int, default=256)
    parser.add_argument("--precision-floor", type=float, default=0.8); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse configuration-compatible completed runs and rerun only incomplete combinations",
    )
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"study": report["study"], "split_hash": report["split_hash"], "seeds": report["seeds"]}, indent=2))


if __name__ == "__main__":
    main()
