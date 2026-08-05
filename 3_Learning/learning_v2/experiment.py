"""Fair comparison of MLP, GraphSAGE, and GAT specialists.

``--split-mode pilot`` reproduces the bounded historical pilot (quota
sub-sample of the governed split). ``--split-mode governed`` trains on the
exact governed train/validation partitions and is the only mode allowed for
final dissertation artifacts. Rule support is frozen from train/validation
only, and predeclared rules are never silently discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

from .calibration import calibrate_predictions, save_calibration
from .feature_layout import MINIMUM_RENDERED_FEATURE_DIM, RENDERED_VISUAL_FEATURE_VERSION
from .data import discover_cached_graphs, load_graphs, save_split
from .metrics import collect_predictions, metrics_from_predictions
from .models import ModelConfig, build_model
from .rules import rule_metadata, rules_for_source
from .schema import FeatureContract
from .trainer import TrainingConfig, save_checkpoint, train_epoch


def _stable_order(site_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{site_id}".encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_common_sites(inventory: dict, governed: dict, cache_dir: Path, views: list[str], max_sites: int, max_nodes: int) -> dict:
    available = None
    sizes = {}
    cache_rule_positive = {}
    cache_rule_support = {view: Counter() for view in views}
    for view in views:
        paths = discover_cached_graphs(cache_dir, view)
        available = set(paths) if available is None else available & set(paths)
        view_rule_ids, view_rule_indices = rules_for_source(view)
        for site, path in paths.items():
            try:
                container = torch.load(path, map_location="cpu", weights_only=False)
                raw = container["data"]
                sizes[(site, view)] = int(container.get("num_nodes", raw.num_nodes))
                labels = raw.node_y_multi[:, list(view_rule_indices)]
                if view == "rendered-visual" and hasattr(raw, "rendered_visible_mask"):
                    labels = labels[raw.rendered_visible_mask.bool()]
                positive_rules = {view_rule_ids[index] for index, count in enumerate(labels.sum(dim=0).tolist()) if count > 0}
                cache_rule_positive[(site, view)] = positive_rules
                cache_rule_support[view].update(positive_rules)
            except Exception:
                sizes[(site, view)] = max_nodes + 1
                cache_rule_positive[(site, view)] = set()
    available = {site for site in available or set() if all(sizes[(site, view)] <= max_nodes for view in views)}
    site_rows = {site["site_id"]: site for site in inventory["sites"]}
    quotas = {"train": max(6, round(max_sites * 0.70)), "val": max(3, round(max_sites * 0.15))}
    quotas["test"] = max(3, max_sites - quotas["train"] - quotas["val"])
    selected = {}
    target_rule = {
        view: cache_rule_support[view].most_common(1)[0][0] if cache_rule_support[view] else None
        for view in views
    }
    for split_name in ("train", "val", "test"):
        candidates = [site for site in governed[split_name] if site in available]
        positives = [site for site in candidates if site_rows[site]["rule_ids"]]
        negatives = [site for site in candidates if not site_rows[site]["rule_ids"]]
        # The max-node gate controls compute; within that gate use a stable
        # pseudo-random order rather than selecting only the smallest pages.
        positives.sort(key=lambda site: _stable_order(site, governed["seed"]))
        negatives.sort(key=lambda site: _stable_order(site, governed["seed"]))
        quota = min(quotas[split_name], len(candidates)); positive_quota = min(len(positives), max(1, round(quota * 0.75)))
        # A bounded pilot is a pipeline/coverage check, so make every specialist
        # observable in every partition before filling the remaining quota.
        chosen = []
        for view in views:
            eligible = [
                site for site in positives
                if target_rule[view] in cache_rule_positive.get((site, view), set()) and site not in chosen
            ]
            if not eligible:
                eligible = [site for site in positives if cache_rule_positive.get((site, view)) and site not in chosen]
            if eligible:
                chosen.append(eligible[0])
        chosen.extend(site for site in positives if site not in chosen and len(chosen) < positive_quota)
        chosen.extend(site for site in negatives if site not in chosen and len(chosen) < quota)
        if len(chosen) < quota:
            remainder = [site for site in positives + negatives if site not in chosen]
            chosen.extend(remainder[: quota - len(chosen)])
        selected[split_name] = sorted(chosen)
    selected["seed"] = governed["seed"]
    return selected


def _graph_num_nodes(cache_dir: Path, view: str, site: str) -> int:
    container = torch.load(cache_dir / site / f"{view}.pt", map_location="cpu", weights_only=False)
    raw = container["data"]
    return int(container.get("num_nodes", raw.num_nodes))


def _governed_split(governed: dict, cache_dir: Path, views: list[str], max_nodes: int) -> dict:
    """Use the exact governed train/validation/test partitions.

    No quota sub-sampling, label-aware selection, or implicit exclusion is
    permitted. Missing caches and an exceeded compute gate are fatal because
    silently shrinking a governed partition changes the study universe.
    """
    available = None
    for view in views:
        paths = discover_cached_graphs(cache_dir, view)
        available = set(paths) if available is None else available & set(paths)
    available = available or set()
    selected = {}
    for split_name in ("train", "val", "test"):
        kept, missing_cache, over_max_nodes = [], [], []
        for site in governed[split_name]:
            if site not in available:
                missing_cache.append(site)
                continue
            try:
                if max_nodes > 0 and any(_graph_num_nodes(cache_dir, view, site) > max_nodes for view in views):
                    over_max_nodes.append(site)
                    continue
            except Exception:
                missing_cache.append(site)
                continue
            kept.append(site)
        if missing_cache or over_max_nodes:
            raise ValueError(
                f"Exact governed {split_name} partition cannot be changed: "
                f"missing_cache={sorted(missing_cache)[:3]} ({len(missing_cache)} total), "
                f"over_max_nodes={sorted(over_max_nodes)[:3]} ({len(over_max_nodes)} total). "
                "Build the missing caches or pass --max-nodes 0 to disable the compute gate."
            )
        selected[split_name] = sorted(kept)
    selected["seed"] = governed.get("seed", 42)
    selected["selection_mode"] = "governed"
    return selected


def validation_rule_average_precision(arrays, rule_ids: list[str]) -> dict:
    """Threshold-independent validation score used for checkpoint selection."""
    valid = arrays.valid_node_mask
    if valid is None:
        valid = torch.ones(arrays.rule_labels.shape[0], dtype=torch.bool)
    per_rule = {}
    for index, rule_id in enumerate(rule_ids):
        labels = arrays.rule_labels[valid, index].detach().cpu().numpy().astype(int)
        scores = arrays.rule_probs[valid, index].detach().cpu().numpy()
        if not labels.any():
            raise ValueError(f"Validation has no positive labels for {rule_id}")
        per_rule[rule_id] = float(average_precision_score(labels, scores))
    return {
        "metric": "rule_macro_average_precision",
        "value": sum(per_rule.values()) / len(per_rule),
        "per_rule": per_rule,
    }


def validate_rule_support(
    cache_paths: dict[str, Path], split: dict, view: str,
    rule_ids: tuple[str, ...], rule_indices: tuple[int, ...],
    *, min_train_positive_sites: int = 1, min_val_positive_sites: int = 1,
) -> tuple[list[str], list[int], dict]:
    """Freeze the predeclared rule set from train/validation support.

    Every requested rule is reported for every partition. A predeclared rule
    below the documented minimum support raises instead of being silently
    discarded; test support is reported but never used for inclusion.
    """
    node_counts = {name: [0] * len(rule_ids) for name in ("train", "val", "test")}
    site_counts = {name: [0] * len(rule_ids) for name in ("train", "val", "test")}
    for split_name in ("train", "val", "test"):
        for site in split[split_name]:
            raw = torch.load(cache_paths[site], map_location="cpu", weights_only=False)["data"]
            values = raw.node_y_multi[:, list(rule_indices)]
            if view == "rendered-visual" and hasattr(raw, "rendered_visible_mask"):
                values = values[raw.rendered_visible_mask.bool()]
            counts = values.sum(dim=0).long().tolist()
            for index, count in enumerate(counts):
                node_counts[split_name][index] += int(count)
                if count > 0:
                    site_counts[split_name][index] += 1
    support = {
        split_name: {
            rule_ids[index]: {
                "positive_nodes": node_counts[split_name][index],
                "positive_sites": site_counts[split_name][index],
            }
            for index in range(len(rule_ids))
        }
        for split_name in ("train", "val", "test")
    }
    inadequate = [
        f"{rule_ids[index]} (train_sites={site_counts['train'][index]}, val_sites={site_counts['val'][index]}, "
        f"train_nodes={node_counts['train'][index]}, val_nodes={node_counts['val'][index]})"
        for index in range(len(rule_ids))
        if site_counts["train"][index] < min_train_positive_sites or site_counts["val"][index] < min_val_positive_sites
    ]
    if inadequate:
        raise ValueError(
            "Predeclared rules lack the documented train/validation support and cannot be trained silently: "
            + "; ".join(inadequate)
            + f". Full support table: {json.dumps(support, sort_keys=True)}. "
            "Remove the rule from --rule-ids explicitly or predeclare lower minimums via "
            "--min-train-positive-sites/--min-val-positive-sites."
        )
    return list(rule_ids), list(rule_indices), support


def _train_one(view: str, architecture: str, split: dict, cache_paths: dict[str, Path], rule_ids: list[str], rule_indices: list[int], output_dir: Path, args, *, feature_variant: str = "full") -> dict:
    train_graphs = load_graphs(split["train"], cache_paths, graph_source=view, rule_indices=rule_indices, feature_variant=feature_variant)
    contract = FeatureContract.from_data(train_graphs[0], view)
    val_graphs = load_graphs(split["val"], cache_paths, graph_source=view, rule_indices=rule_indices, contract=contract, feature_variant=feature_variant)
    test_graphs = load_graphs(split["test"], cache_paths, graph_source=view, rule_indices=rule_indices, contract=contract, feature_variant=feature_variant)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    config = ModelConfig(architecture, contract.feature_dim, contract.num_tags, len(rule_ids), hidden_dim=args.hidden, num_layers=args.layers, dropout=args.dropout, heads=args.heads)
    train_config = TrainingConfig(
        epochs=args.epochs, patience=args.patience, learning_rate=args.lr,
        negative_ratio=args.negative_ratio, minimum_negatives=args.minimum_negatives,
        selection_metric="rule_macro_average_precision",
    )
    torch.manual_seed(args.seed); random.seed(args.seed)
    model = build_model(config).to(args.device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run_dir = output_dir / view / (architecture if feature_variant == "full" else f"{feature_variant}/{architecture}"); run_dir.mkdir(parents=True, exist_ok=True)
    history = []; best_metric = float("-inf"); best_epoch = 0; epochs_since_best = 0; stopped_early = False
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train_loader, optimizer, train_config, args.device)
        val_arrays = collect_predictions(model, val_loader, args.device)
        val_metrics = metrics_from_predictions(val_arrays, rule_ids=rule_ids)
        selection = validation_rule_average_precision(val_arrays, rule_ids)
        metric = float(selection["value"])
        history.append({"epoch": epoch, "training": training, "validation_at_0_5": val_metrics, "validation_selection": selection})
        if metric > best_metric:
            # Strict improvement only: a tie must not move "best" to a later epoch.
            best_metric = metric; best_epoch = epoch; epochs_since_best = 0
            save_checkpoint(run_dir / "best_model.pt", model=model, optimizer=optimizer, epoch=epoch, best_metric=metric, model_config=config, training_config=train_config, feature_contract=contract, rule_ids=rule_ids, rule_indices=rule_indices)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                stopped_early = True
                break
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if best_metric <= 0.0 and not getattr(args, "allow_zero_validation_metric", False):
        raise ValueError(
            f"Validation average precision never exceeded zero for {view}/{architecture} "
            f"(best_epoch={best_epoch}); refusing to calibrate and freeze thresholds from a "
            "non-learning run. Pass --allow-zero-validation-metric only to reproduce a historical pilot."
        )
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_arrays = collect_predictions(model, val_loader, args.device)
    calibration = calibrate_predictions(val_arrays, rule_ids, precision_floor=args.precision_floor)
    unsupported_calibration = [rule_id for rule_id in rule_ids if not calibration["rules"][rule_id]["supported"]]
    if unsupported_calibration:
        raise ValueError(
            f"Validation calibration is unsupported for rules {unsupported_calibration} "
            "(no positive validation nodes); refusing to freeze thresholds for an unobserved rule."
        )
    save_calibration(run_dir / "calibration.json", calibration)
    unmet_precision_floor = [
        name for name, result in {
            "node": calibration["node"], "page": calibration["page"], **calibration["rules"]
        }.items() if not result.get("precision_floor_met", False)
    ]
    calibration_floor_policy = getattr(args, "calibration_floor_policy", "strict")
    if calibration_floor_policy not in {"strict", "validation_f1_fallback"}:
        raise ValueError(f"Unknown calibration floor policy: {calibration_floor_policy}")
    if (
        unmet_precision_floor
        and calibration_floor_policy == "strict"
        and not getattr(args, "allow_unmet_precision_floor", False)
    ):
        raise ValueError(
            f"Validation precision floor {args.precision_floor} was not met for {unmet_precision_floor} "
            f"in {view}/{architecture}. The diagnostic fallback is recorded in {run_dir / 'calibration.json'}, "
            "but it will not be frozen as a final threshold. Lower the predeclared floor explicitly or pass "
            "--allow-unmet-precision-floor only for a non-final pilot."
        )
    test_arrays = collect_predictions(model, test_loader, args.device)
    thresholds = torch.tensor([calibration["recommended"]["rule_thresholds"][rule_id] for rule_id in rule_ids])
    metrics = metrics_from_predictions(
        test_arrays, node_threshold=calibration["recommended"]["node_threshold"],
        rule_thresholds=thresholds, page_threshold=calibration["recommended"]["page_threshold"], rule_ids=rule_ids,
    )
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 2, "pilot": args.split_mode == "pilot", "graph_source": view, "architecture": architecture, "feature_variant": feature_variant,
        "split_mode": args.split_mode,
        "split_provenance": {"path": str(args.split), "sha256": _sha256(args.split)},
        "feature_contract": contract.to_dict(), "model_config": config.to_dict(), "training_config": vars(args),
        "rules": [rule_metadata(rule_id) for rule_id in rule_ids], "best_epoch": checkpoint["epoch"],
        "selection_metric": "rule_macro_average_precision",
        "best_validation_rule_average_precision": checkpoint["best_metric"], "stopped_early": stopped_early,
        "calibration_floor_policy": calibration_floor_policy,
        "unmet_precision_floor": unmet_precision_floor,
        "test_metrics": "test_metrics.json",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        "view": view, "architecture": architecture, "feature_variant": feature_variant,
        "rules": rule_ids, "best_epoch": checkpoint["epoch"], "stopped_early": stopped_early,
        "calibration_floor_policy": calibration_floor_policy,
        "unmet_precision_floor": unmet_precision_floor,
        "test": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=["dom", "a11y-tree", "rendered-visual"])
    parser.add_argument("--architectures", nargs="+", default=["mlp", "graphsage", "gat"])
    parser.add_argument("--rule-ids", nargs="+", help="Predeclared rule allow-list for the selected graph source")
    parser.add_argument("--split-mode", choices=["pilot", "governed"], default="pilot",
                        help="pilot: bounded quota sub-sample (historical reproduction only); governed: exact governed train/validation/test partitions (required for final artifacts)")
    parser.add_argument("--min-train-positive-sites", type=int, default=1, help="Documented minimum positive-site support in train for each predeclared rule")
    parser.add_argument("--min-val-positive-sites", type=int, default=1, help="Documented minimum positive-site support in validation for each predeclared rule")
    parser.add_argument("--max-sites", type=int, default=30); parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=2); parser.add_argument("--patience", type=int, default=4,
                        help="Early-stopping patience in epochs without validation improvement")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--allow-zero-validation-metric", action="store_true",
                        help="Reproduce a historical non-learning pilot; forbidden for final training")
    parser.add_argument("--allow-unmet-precision-floor", action="store_true",
                        help="Use a diagnostic fallback threshold; forbidden with --split-mode governed")
    parser.add_argument("--hidden", type=int, default=64); parser.add_argument("--layers", type=int, default=2); parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--negative-ratio", type=float, default=8.0); parser.add_argument("--minimum-negatives", type=int, default=256)
    parser.add_argument("--precision-floor", type=float, default=0.25); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-legacy-rendered", action="store_true", help="Reproduce old caches without claiming rendered visual-cue evidence")
    parser.add_argument("--allow-static-a11y", action="store_true", help="Reproduce old static a11y approximation; forbidden for dissertation claims")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.split_mode == "governed" and (args.allow_zero_validation_metric or args.allow_unmet_precision_floor):
        raise ValueError("Governed final training forbids pilot-only validation/calibration overrides")
    if "rendered-visual" in args.views and not args.allow_legacy_rendered:
        legacy = []
        for site, path in discover_cached_graphs(args.cache_dir, "rendered-visual").items():
            raw = torch.load(path, map_location="cpu", weights_only=False)["data"]
            version = int(getattr(raw, "rendered_visual_feature_version", 0))
            if version < RENDERED_VISUAL_FEATURE_VERSION or int(raw.x.shape[1]) < MINIMUM_RENDERED_FEATURE_DIM:
                legacy.append(site)
        if legacy:
            raise ValueError(
                f"Rendered cache fails the visual-evidence contract for {len(legacy)} sites "
                f"(requires version>={RENDERED_VISUAL_FEATURE_VERSION}, features>={MINIMUM_RENDERED_FEATURE_DIM}). "
                "Regenerate it for dissertation experiments, or pass --allow-legacy-rendered only to reproduce the historical pilot."
            )
    if "a11y-tree" in args.views and not args.allow_static_a11y:
        static = []
        for site, path in discover_cached_graphs(args.cache_dir, "a11y-tree").items():
            raw = torch.load(path, map_location="cpu", weights_only=False)["data"]
            if (
                not bool(getattr(raw, "live_accessibility_tree", False))
                or int(getattr(raw, "live_ax_feature_version", 0)) < 1
                or getattr(raw, "ax_capture_provenance", "") != "same_session_sidecar"
            ):
                static.append(site)
        if static:
            raise ValueError(
                f"Accessibility-tree cache lacks same-session AX provenance for {len(static)} sites. "
                "Build it with learning_v2.build_same_session_ax_cache, or pass --allow-static-a11y only to reproduce a historical pilot."
            )
    inventory = json.loads(args.inventory.read_text()); governed = json.loads(args.split.read_text())
    if args.split_mode == "governed":
        split = _governed_split(governed, args.cache_dir, args.views, args.max_nodes)
    else:
        split = _select_common_sites(inventory, governed, args.cache_dir, args.views, args.max_sites, args.max_nodes)
    split_filename = "governed_split.json" if args.split_mode == "governed" else "pilot_split.json"
    save_split(args.output_dir / split_filename, split, graph_source="multi-view", rule_ids=[])
    results = []
    requested = set(args.rule_ids or [])
    if requested:
        owned_by_requested_views = {
            rule_id for view in args.views for rule_id in rules_for_source(view)[0]
        }
        unknown_or_unowned = sorted(requested - owned_by_requested_views)
        if unknown_or_unowned:
            raise ValueError(
                f"Requested rules are unknown or not owned by the requested views: {unknown_or_unowned}"
            )
    for view in args.views:
        cache_paths = discover_cached_graphs(args.cache_dir, view)
        all_rule_ids, all_rule_indices = rules_for_source(view)
        if requested:
            selected_pairs = [
                (rule_id, rule_index) for rule_id, rule_index in zip(all_rule_ids, all_rule_indices)
                if rule_id in requested
            ]
            if not selected_pairs:
                # Another requested view owns these rules; it will train them.
                continue
            all_rule_ids, all_rule_indices = map(tuple, zip(*selected_pairs))
        rule_ids, rule_indices, support = validate_rule_support(
            cache_paths, split, view, all_rule_ids, all_rule_indices,
            min_train_positive_sites=args.min_train_positive_sites,
            min_val_positive_sites=args.min_val_positive_sites,
        )
        (args.output_dir / view).mkdir(parents=True, exist_ok=True)
        (args.output_dir / view / "rule_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
        for architecture in args.architectures:
            print(f"Training {view}/{architecture} on {sum(len(split[name]) for name in ('train','val','test'))} {args.split_mode} sites and {len(rule_ids)} predeclared rules", flush=True)
            results.append(_train_one(view, architecture, split, cache_paths, rule_ids, rule_indices, args.output_dir, args))
    summary = {"schema_version": 1, "pilot": args.split_mode == "pilot", "split_mode": args.split_mode, "split": split_filename, "results": results}
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
