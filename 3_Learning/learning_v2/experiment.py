"""Fair pilot comparison of MLP, GraphSAGE, and GAT specialists.

The pilot uses one governed site split and identical features/budgets per
architecture. Rule support is selected from train/validation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .calibration import calibrate_predictions, save_calibration
from .data import discover_cached_graphs, load_graphs, save_split
from .metrics import collect_predictions, metrics_from_predictions
from .models import ModelConfig, build_model
from .rules import rule_metadata, rules_for_source
from .schema import FeatureContract
from .trainer import TrainingConfig, save_checkpoint, train_epoch


def _stable_order(site_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{site_id}".encode()).hexdigest()


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


def _supported_rules(cache_paths: dict[str, Path], split: dict, view: str, rule_ids: tuple[str, ...], rule_indices: tuple[int, ...]) -> tuple[list[str], list[int], dict]:
    support = {"train": [0] * len(rule_ids), "val": [0] * len(rule_ids), "test": [0] * len(rule_ids)}
    for split_name in support:
        for site in split[split_name]:
            raw = torch.load(cache_paths[site], map_location="cpu", weights_only=False)["data"]
            values = raw.node_y_multi[:, list(rule_indices)]
            if view == "rendered-visual" and hasattr(raw, "rendered_visible_mask"):
                values = values[raw.rendered_visible_mask.bool()]
            counts = values.sum(dim=0).long().tolist()
            support[split_name] = [a + int(b) for a, b in zip(support[split_name], counts)]
    # Rule inclusion is frozen from train/validation support only. Test support
    # is reported but never used to choose model outputs or thresholds.
    selected = [
        index for index in range(len(rule_ids))
        if support["train"][index] >= 1 and support["val"][index] >= 1
    ]
    if not selected:
        selected = sorted(range(len(rule_ids)), key=lambda index: support["train"][index], reverse=True)[:1]
    filtered_ids = [rule_ids[index] for index in selected]
    filtered_indices = [rule_indices[index] for index in selected]
    return filtered_ids, filtered_indices, {
        split_name: {rule_ids[index]: values[index] for index in selected}
        for split_name, values in support.items()
    }


def _train_one(view: str, architecture: str, split: dict, cache_paths: dict[str, Path], rule_ids: list[str], rule_indices: list[int], output_dir: Path, args) -> dict:
    train_graphs = load_graphs(split["train"], cache_paths, graph_source=view, rule_indices=rule_indices)
    contract = FeatureContract.from_data(train_graphs[0], view)
    val_graphs = load_graphs(split["val"], cache_paths, graph_source=view, rule_indices=rule_indices, contract=contract)
    test_graphs = load_graphs(split["test"], cache_paths, graph_source=view, rule_indices=rule_indices, contract=contract)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    config = ModelConfig(architecture, contract.feature_dim, contract.num_tags, len(rule_ids), hidden_dim=args.hidden, num_layers=args.layers, dropout=args.dropout, heads=args.heads)
    train_config = TrainingConfig(epochs=args.epochs, patience=args.epochs, learning_rate=args.lr, negative_ratio=args.negative_ratio, minimum_negatives=args.minimum_negatives)
    torch.manual_seed(args.seed); random.seed(args.seed)
    model = build_model(config).to(args.device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run_dir = output_dir / view / architecture; run_dir.mkdir(parents=True, exist_ok=True)
    history = []; best_metric = -1.0
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train_loader, optimizer, train_config, args.device)
        val_arrays = collect_predictions(model, val_loader, args.device)
        val_metrics = metrics_from_predictions(val_arrays, rule_ids=rule_ids)
        metric = float(val_metrics["rule_f1"])
        history.append({"epoch": epoch, "training": training, "validation": val_metrics})
        if metric >= best_metric:
            best_metric = metric
            save_checkpoint(run_dir / "best_model.pt", model=model, optimizer=optimizer, epoch=epoch, best_metric=metric, model_config=config, training_config=train_config, feature_contract=contract, rule_ids=rule_ids, rule_indices=rule_indices)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_arrays = collect_predictions(model, val_loader, args.device)
    calibration = calibrate_predictions(val_arrays, rule_ids, precision_floor=args.precision_floor)
    save_calibration(run_dir / "calibration.json", calibration)
    test_arrays = collect_predictions(model, test_loader, args.device)
    thresholds = torch.tensor([calibration["recommended"]["rule_thresholds"][rule_id] for rule_id in rule_ids])
    metrics = metrics_from_predictions(
        test_arrays, node_threshold=calibration["recommended"]["node_threshold"],
        rule_thresholds=thresholds, page_threshold=calibration["recommended"]["page_threshold"], rule_ids=rule_ids,
    )
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 2, "pilot": True, "graph_source": view, "architecture": architecture,
        "feature_contract": contract.to_dict(), "model_config": config.to_dict(), "training_config": vars(args),
        "rules": [rule_metadata(rule_id) for rule_id in rule_ids], "best_epoch": checkpoint["epoch"],
        "best_validation_rule_f1": checkpoint["best_metric"], "test_metrics": "test_metrics.json",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"view": view, "architecture": architecture, "rules": rule_ids, "best_epoch": checkpoint["epoch"], "test": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=["dom", "a11y-tree", "rendered-visual"])
    parser.add_argument("--architectures", nargs="+", default=["mlp", "graphsage", "gat"])
    parser.add_argument("--max-sites", type=int, default=30); parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=2); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64); parser.add_argument("--layers", type=int, default=2); parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--negative-ratio", type=float, default=8.0); parser.add_argument("--minimum-negatives", type=int, default=256)
    parser.add_argument("--precision-floor", type=float, default=0.25); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(args.inventory.read_text()); governed = json.loads(args.split.read_text())
    split = _select_common_sites(inventory, governed, args.cache_dir, args.views, args.max_sites, args.max_nodes)
    save_split(args.output_dir / "pilot_split.json", split, graph_source="multi-view", rule_ids=[])
    results = []
    for view in args.views:
        cache_paths = discover_cached_graphs(args.cache_dir, view)
        all_rule_ids, all_rule_indices = rules_for_source(view)
        rule_ids, rule_indices, support = _supported_rules(cache_paths, split, view, all_rule_ids, all_rule_indices)
        (args.output_dir / view).mkdir(parents=True, exist_ok=True)
        (args.output_dir / view / "rule_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
        for architecture in args.architectures:
            print(f"Training {view}/{architecture} on {sum(len(split[name]) for name in ('train','val','test'))} pilot sites and {len(rule_ids)} supported rules", flush=True)
            results.append(_train_one(view, architecture, split, cache_paths, rule_ids, rule_indices, args.output_dir, args))
    summary = {"schema_version": 1, "pilot": True, "split": "pilot_split.json", "results": results}
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
