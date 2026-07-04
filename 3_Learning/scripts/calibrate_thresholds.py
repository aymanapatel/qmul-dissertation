#!/usr/bin/env python3
"""
calibrate_thresholds.py

Sweep validation thresholds for a trained DOM/a11y GNN checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_extractor import FeatureExtractor, ProcessedPage
from graph_sources import GRAPH_SOURCE_A11Y_TREE, GRAPH_SOURCE_DOM
from models import DOMAttentionNet
from wcag_rules import NUM_RULES


def cache_candidates(output_dir: Path, site_name: str, graph_source: str) -> List[Path]:
    return [
        output_dir / f"{site_name}_{graph_source}.pt",
        output_dir / f"{site_name}.{graph_source}.pt",
        output_dir / f"{site_name}.pt",
    ]


def load_or_process_site(
    site_name: str,
    data_dir: Path,
    output_dir: Path,
    extractor: FeatureExtractor,
    graph_source: str,
) -> Optional[Data]:
    for cache_path in cache_candidates(output_dir, site_name, graph_source):
        if cache_path.exists():
            try:
                page = ProcessedPage.load(cache_path)
                if getattr(page.data, "graph_source", GRAPH_SOURCE_DOM) == graph_source:
                    return page.data
            except Exception:
                pass

    site_dir = data_dir / site_name
    html_path = site_dir / "0.html"
    axe_path = site_dir / "page-0_home.json"
    if not html_path.exists() or not axe_path.exists():
        print(f"  [skip] missing files for {site_name}")
        return None

    page = extractor.process_page(
        html_path=html_path,
        axe_report_path=axe_path,
        extract_visual=False,
        graph_source=graph_source,
    )
    cache_path = output_dir / f"{site_name}_{graph_source}.pt"
    page.save(cache_path)
    return page.data


def build_model(checkpoint: dict, attr_dim: int, text_dim: int, device: str) -> DOMAttentionNet:
    hparams = checkpoint.get("hparams", {})
    model = DOMAttentionNet(
        num_tags=hparams.get("num_tags", 116),
        tag_embed_dim=hparams.get("tag_embed_dim", 32),
        attr_dim=attr_dim,
        text_dim=text_dim,
        hidden_dim=hparams.get("hidden_dim", 256),
        num_node_classes=hparams.get("num_node_classes", 2),
        num_graph_classes=hparams.get("num_graph_classes", 2),
        num_rules=hparams.get("num_rules", NUM_RULES),
        num_layers=hparams.get("num_layers", 4),
        heads=hparams.get("heads", 4),
        dropout=hparams.get("dropout", 0.3),
        pooling=hparams.get("pooling", "mean"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def f1_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def graph_metrics(records: List[dict], threshold: float) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for record in records:
        pred = record["graph_prob"] >= threshold
        label = bool(record["graph_label"])
        tp += int(pred and label)
        fp += int(pred and not label)
        fn += int(not pred and label)
        tn += int(not pred and not label)
    metrics = f1_from_counts(tp, fp, fn)
    metrics["accuracy"] = (tp + tn) / len(records) if records else 0.0
    metrics["threshold"] = threshold
    return metrics


def node_metrics(records: List[dict], node_threshold: float, graph_threshold: float) -> Dict[str, float]:
    tp = fp = fn = 0
    for record in records:
        graph_pass = record["graph_prob"] >= graph_threshold
        preds = (
            record["node_probs"] >= node_threshold
            if graph_pass
            else torch.zeros_like(record["node_probs"], dtype=torch.bool)
        )
        labels = record["node_labels"].bool()
        tp += (preds & labels).sum().item()
        fp += (preds & ~labels).sum().item()
        fn += (~preds & labels).sum().item()
    metrics = f1_from_counts(tp, fp, fn)
    metrics["node_threshold"] = node_threshold
    metrics["graph_threshold"] = graph_threshold
    return metrics


def rule_metrics(records: List[dict], rule_threshold: float, graph_threshold: float) -> Dict[str, float]:
    tp = fp = fn = 0
    for record in records:
        graph_pass = record["graph_prob"] >= graph_threshold
        preds = (
            record["rule_probs"] >= rule_threshold
            if graph_pass
            else torch.zeros_like(record["rule_probs"], dtype=torch.bool)
        )
        labels = record["rule_labels"].bool()
        tp += (preds & labels).sum().item()
        fp += (preds & ~labels).sum().item()
        fn += (~preds & labels).sum().item()
    metrics = f1_from_counts(tp, fp, fn)
    metrics["rule_threshold"] = rule_threshold
    metrics["graph_threshold"] = graph_threshold
    return metrics


def choose_best(candidates: List[Dict[str, float]], primary: str = "f1") -> Dict[str, float]:
    return max(
        candidates,
        key=lambda m: (
            m.get(primary, 0.0),
            m.get("precision", 0.0),
            m.get("recall", 0.0),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate graph/node/rule thresholds")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing best_model.pt and split.json")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing site subdirectories")
    parser.add_argument("--output-dir", type=str, default=None, help="Graph cache directory (default: split args output_dir or ./graphs_multi)")
    parser.add_argument("--graph-source", type=str, default=None, choices=[GRAPH_SOURCE_DOM, GRAPH_SOURCE_A11Y_TREE])
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--output", type=str, default=None, help="Calibration JSON output path")
    parser.add_argument("--min-val-graphs", type=int, default=10, help="Refuse calibration when validation split is smaller than this")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    split_path = model_dir / "split.json"
    checkpoint_path = model_dir / "best_model.pt"
    if not split_path.exists() or not checkpoint_path.exists():
        print(f"Expected {split_path} and {checkpoint_path}")
        sys.exit(1)

    split = json.loads(split_path.read_text())
    split_args = split.get("args", {})
    graph_source = args.graph_source or split_args.get("graph_source", GRAPH_SOURCE_DOM)
    output_dir = Path(args.output_dir or split_args.get("output_dir", "./graphs_multi"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Loading validation split from {split_path}")
    print(f"Graph source: {graph_source}")
    print(f"Graph cache: {output_dir}")

    extractor = FeatureExtractor(device=device)
    val_data = []
    for site_name in split.get("val", []):
        data = load_or_process_site(site_name, data_dir, output_dir, extractor, graph_source)
        if data is not None:
            val_data.append(data)

    if not val_data:
        print("No validation graphs loaded")
        sys.exit(1)
    if len(val_data) < args.min_val_graphs:
        print(
            f"Validation split has only {len(val_data)} graphs; "
            f"refusing to calibrate below --min-val-graphs={args.min_val_graphs}."
        )
        sys.exit(1)

    text_dim = 384
    for data in val_data:
        if hasattr(data, "text_embeddings") and data.text_embeddings is not None:
            text_dim = data.text_embeddings.shape[1]
            break
    attr_dim = val_data[0].x.shape[1] - text_dim

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint, attr_dim=attr_dim, text_dim=text_dim, device=device)

    records = []
    with torch.no_grad():
        for data in val_data:
            data = data.to(device)
            node_logits, node_rule_logits, graph_logits = model(data.x, data.edge_index, data.tag_indices)
            records.append(
                {
                    "graph_prob": F.softmax(graph_logits, dim=-1)[:, 1].cpu().item(),
                    "graph_label": int(data.y.cpu().item()),
                    "node_probs": F.softmax(node_logits, dim=-1)[:, 1].cpu(),
                    "node_labels": data.node_y.cpu().long(),
                    "rule_probs": torch.sigmoid(node_rule_logits).cpu(),
                    "rule_labels": data.node_y_multi.cpu().float(),
                }
            )

    graph_grid = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]
    node_grid = [round(x, 2) for x in np.arange(0.1, 0.96, 0.05)]
    rule_grid = [round(x, 2) for x in np.arange(0.1, 0.96, 0.05)]

    graph_candidates = [graph_metrics(records, t) for t in graph_grid]
    best_graph = choose_best(graph_candidates)

    node_candidates = [
        node_metrics(records, node_t, graph_t)
        for graph_t in graph_grid
        for node_t in node_grid
    ]
    best_node = choose_best(node_candidates)

    rule_candidates = [
        rule_metrics(records, rule_t, best_node["graph_threshold"])
        for rule_t in rule_grid
    ]
    best_rule = choose_best(rule_candidates)

    calibration = {
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "graph_source": graph_source,
        "validation_graphs": len(records),
        "recommended": {
            "graph_threshold": best_node["graph_threshold"],
            "node_threshold": best_node["node_threshold"],
            "rule_threshold": best_rule["rule_threshold"],
        },
        "best_graph": best_graph,
        "best_node_gated": best_node,
        "best_rule_gated": best_rule,
    }

    output_path = Path(args.output) if args.output else model_dir / "calibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Saved calibration to {output_path}")
    print(json.dumps(calibration["recommended"], indent=2))


if __name__ == "__main__":
    main()
