#!/usr/bin/env python3
"""
predict_site.py

Predict accessibility violations on a new HTML page using a trained multi-label model.
Generates a detailed report with predicted violating nodes and their likely WCAG rules.

Usage:
    python predict_site.py \
        --html /path/to/page.html \
        --model ./models_multi/best_model.pt \
        --output ./reports/prediction.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_extractor import FeatureExtractor
from graph_sources import GRAPH_SOURCE_A11Y_TREE, GRAPH_SOURCE_DOM
from models import DOMAttentionNet
from wcag_rules import INDEX_TO_RULE, NUM_RULES

# Tags that can actually have accessibility violations
ACTIONABLE_TAGS = {
    "a",
    "button",
    "img",
    "input",
    "select",
    "textarea",
    "form",
    "label",
    "fieldset",
    "legend",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "th",
    "td",
    "tr",
    "caption",
    "video",
    "audio",
    "iframe",
    "object",
    "embed",
    "svg",
    "canvas",
    "nav",
    "main",
    "article",
    "section",
    "aside",
    "header",
    "footer",
    "div",
    "span",
    "p",
    "li",
    "ul",
    "ol",
    "details",
    "summary",
    "dialog",
}


def predict_page(model, data, device="cpu"):
    """Run prediction on a single page."""
    model = model.to(device)
    model.eval()

    data = data.to(device)
    with torch.no_grad():
        node_logits, node_rule_logits, graph_logits = model(
            data.x,
            data.edge_index,
            data.tag_indices,
            data.batch if hasattr(data, "batch") else None,
        )

        # Binary node probabilities
        node_probs = F.softmax(node_logits, dim=-1)[:, 1].cpu()

        # Multi-label rule probabilities
        rule_probs = torch.sigmoid(node_rule_logits).cpu()

        # Graph-level probability
        graph_prob = (
            F.softmax(graph_logits, dim=-1)[:, 1].cpu().item()
            if graph_logits.dim() == 2
            else 0.5
        )

    return {
        "node_probs": node_probs,
        "rule_probs": rule_probs,
        "graph_violation_prob": graph_prob,
    }


def get_top_rules(rule_probs, top_k=3):
    """Get top-k predicted rules for a node."""
    top_indices = rule_probs.argsort(descending=True)[:top_k]
    rules = []
    for idx in top_indices:
        if rule_probs[idx] > 0.1:  # Only include rules with >10% confidence
            rules.append(
                {
                    "rule_id": INDEX_TO_RULE[idx.item()],
                    "probability": round(rule_probs[idx].item(), 4),
                }
            )
    return rules


def load_calibration(path: Path) -> dict:
    """Load threshold calibration JSON if supplied."""
    if not path.exists():
        print(f"Error: Calibration file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calibration_value(calibration: dict, key: str, default: float) -> float:
    """Read a threshold from either top-level or recommended calibration shape."""
    if not calibration:
        return default
    if key in calibration:
        return float(calibration[key])
    recommended = calibration.get("recommended", {})
    if key in recommended:
        return float(recommended[key])
    return default


def load_axe_summary(axe_path: Path | None) -> dict:
    """Read high-level axe counts and incomplete checks for reporting."""
    if not axe_path:
        return {"available": False}
    with open(axe_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    incomplete = []
    for item in report.get("incomplete", []):
        incomplete.append(
            {
                "rule_id": item.get("id"),
                "impact": item.get("impact"),
                "help": item.get("help"),
                "nodes": len(item.get("nodes", [])),
            }
        )

    return {
        "available": True,
        "violation_count": len(report.get("violations", [])),
        "pass_count": len(report.get("passes", [])),
        "incomplete_count": len(report.get("incomplete", [])),
        "inapplicable_count": len(report.get("inapplicable", [])),
        "incomplete": incomplete,
    }


def make_prediction_entry(
    node, idx: int, prob: float, rule_probs, node_threshold: float, status: str
) -> dict:
    """Build a JSON-safe prediction/candidate entry."""
    return {
        "node_id": idx,
        "tag": node.tag,
        "probability": round(prob, 4),
        "predicted_violation": status == "predicted_violation",
        "status": status,
        "attributes": dict(node.attrs),
        "text_preview": node.text_content[:100] if node.text_content else "",
        "predicted_rules": get_top_rules(rule_probs, top_k=3),
        "node_threshold": node_threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict accessibility violations on a webpage"
    )
    parser.add_argument("--html", type=str, required=True, help="Path to HTML file")
    parser.add_argument(
        "--axe",
        type=str,
        default=None,
        help="Optional: Path to axe report for comparison",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model .pt file"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON report path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Node probability threshold for violation prediction (default: calibration or 0.5)",
    )
    parser.add_argument(
        "--graph-threshold",
        type=float,
        default=None,
        help="Graph probability threshold for enabling node violations (default: calibration or 0.5)",
    )
    parser.add_argument(
        "--rule-threshold",
        type=float,
        default=None,
        help="Rule probability threshold for calibrated reports (default: calibration or 0.5)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Optional calibration.json from scripts/calibrate_thresholds.py",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)"
    )
    parser.add_argument(
        "--graph-source",
        type=str,
        default=GRAPH_SOURCE_DOM,
        choices=[GRAPH_SOURCE_DOM, GRAPH_SOURCE_A11Y_TREE],
        help="Graph source to build: dom or a11y-tree",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top predictions to include in report",
    )
    parser.add_argument("--visual", action="store_true", help="Extract visual features")
    args = parser.parse_args()

    html_path = Path(args.html)
    model_path = Path(args.model)
    axe_path = Path(args.axe) if args.axe else None
    calibration = load_calibration(Path(args.calibration)) if args.calibration else {}

    if not html_path.exists():
        print(f"Error: HTML file not found: {html_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    if axe_path and not axe_path.exists():
        print(f"Error: Axe report file not found: {axe_path}")
        sys.exit(1)

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("Accessibility Violation Prediction (Multi-Label)")
    print("=" * 70)
    print(f"HTML: {html_path}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Graph source: {args.graph_source}")
    if calibration:
        print(f"Calibration: {args.calibration}")
    print()

    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hparams = checkpoint.get("hparams", {})
    model_graph_source = hparams.get("graph_source", GRAPH_SOURCE_DOM)
    if model_graph_source != args.graph_source:
        print(
            f"Error: Model was trained with graph_source={model_graph_source}, "
            f"but prediction requested graph_source={args.graph_source}"
        )
        sys.exit(1)

    # Process page
    print("Processing HTML page...")
    extractor = FeatureExtractor(device=device)
    page = extractor.process_page(
        html_path=html_path,
        axe_report_path=axe_path,
        extract_visual=args.visual,
        graph_source=args.graph_source,
    )
    has_real_ground_truth = bool(args.axe) and bool(
        getattr(page.data, "has_ground_truth", False)
    )

    text_dim = (
        page.data.text_embeddings.shape[1]
        if hasattr(page.data, "text_embeddings")
        and page.data.text_embeddings is not None
        else 384
    )
    attr_dim = page.data.x.shape[1] - text_dim

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
    print(
        f"Loaded model (GAT-{hparams.get('hidden_dim', 256)}x{hparams.get('num_layers', 4)}, {hparams.get('num_rules', NUM_RULES)} rules)"
    )

    # Run prediction
    print("\nRunning prediction...")
    results = predict_page(model, page.data, device=device)

    node_probs = results["node_probs"]
    rule_probs = results["rule_probs"]
    graph_prob = results["graph_violation_prob"]
    node_threshold = (
        args.threshold
        if args.threshold is not None
        else calibration_value(calibration, "node_threshold", 0.5)
    )
    graph_threshold = (
        args.graph_threshold
        if args.graph_threshold is not None
        else calibration_value(calibration, "graph_threshold", 0.5)
    )
    rule_threshold = (
        args.rule_threshold
        if args.rule_threshold is not None
        else calibration_value(calibration, "rule_threshold", 0.5)
    )
    axe_summary = load_axe_summary(axe_path)

    # Filter to actionable tags
    actionable_mask = torch.zeros(len(page.node_map), dtype=torch.bool)
    for nid, node in page.node_map.items():
        if node.tag in ACTIONABLE_TAGS:
            actionable_mask[nid] = True

    filtered_probs = node_probs.clone()
    filtered_probs[~actionable_mask] = 0.0
    node_evidence_prob = filtered_probs.max().item() if filtered_probs.numel() else 0.0
    page_violation_prob = max(graph_prob, node_evidence_prob)
    page_is_likely_violation = page_violation_prob >= graph_threshold
    page_prediction = "likely_violation" if page_is_likely_violation else "likely_clean"

    # Generate report
    print(f"\n{'=' * 70}")
    print(f"PREDICTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total nodes analyzed: {len(page.node_map)}")
    print(f"Actionable nodes: {actionable_mask.sum().item()}")
    print(f"Graph violation probability: {graph_prob:.4f}")
    print(f"Node evidence probability: {node_evidence_prob:.4f}")
    print(f"Page violation probability: {page_violation_prob:.4f} (threshold={graph_threshold:.4f})")
    print(f"Page prediction: {page_prediction}")
    raw_over_threshold = int((filtered_probs >= node_threshold).sum().item())
    confirmed_over_threshold = raw_over_threshold if page_is_likely_violation else 0
    print(f"Nodes with violation prob >= {node_threshold}: {raw_over_threshold}")
    print(f"Confirmed node violations after page decision: {confirmed_over_threshold}")

    # Top-K predictions stay as a compact raw view, while confirmed
    # violations/candidates below include every node above the report floor.
    sorted_indices = filtered_probs.argsort(descending=True)

    print(f"\nTop {args.top_k} Most Likely Violations:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Node':<6} {'Tag':<12} {'Prob':<8} {'Predicted Rules'}")
    print("-" * 70)

    raw_top_predictions = []
    predicted_violations = []
    candidate_warnings = []
    for rank, idx in enumerate(sorted_indices, 1):
        idx = idx.item()
        prob = filtered_probs[idx].item()
        if prob < 0.2:
            break
        node = page.node_map.get(idx)

        if not node:
            continue

        tag = node.tag[:12]
        top_rules = get_top_rules(rule_probs[idx], top_k=3)
        rule_str = (
            ", ".join([f"{r['rule_id']}({r['probability']:.2f})" for r in top_rules])
            if top_rules
            else "none"
        )

        over_node_threshold = prob >= node_threshold
        confirmed_violation = page_is_likely_violation and over_node_threshold
        if rank <= args.top_k:
            marker = " [VIOLATION]" if confirmed_violation else ""
            print(f"{rank:<6} {idx:<6} {tag:<12} {prob:.4f}  {rule_str}{marker}")

            raw_entry = make_prediction_entry(
                node=node,
                idx=idx,
                prob=prob,
                rule_probs=rule_probs[idx],
                node_threshold=node_threshold,
                status="raw_candidate",
            )
            raw_entry["rank"] = rank
            raw_top_predictions.append(raw_entry)

        if confirmed_violation:
            predicted_violations.append(
                make_prediction_entry(
                    node=node,
                    idx=idx,
                    prob=prob,
                    rule_probs=rule_probs[idx],
                    node_threshold=node_threshold,
                    status="predicted_violation",
                )
            )
        elif prob >= 0.2:
            status = "candidate_warning"
            if not page_is_likely_violation and over_node_threshold:
                status = "model_warning_on_likely_clean_page"
                if axe_summary.get("available") and axe_summary.get("violation_count") == 0:
                    status = "false_positive_against_axe"
            candidate_warnings.append(
                make_prediction_entry(
                    node=node,
                    idx=idx,
                    prob=prob,
                    rule_probs=rule_probs[idx],
                    node_threshold=node_threshold,
                    status=status,
                )
            )

    # Ground truth comparison if available
    has_ground_truth = (
        has_real_ground_truth
        and hasattr(page.data, "node_y")
        and page.data.node_y is not None
    )
    gt_stats = {}

    if has_ground_truth:
        predictions = (
            (filtered_probs >= node_threshold) & page_is_likely_violation
        ).long().cpu()
        node_y = page.data.node_y.cpu().long()

        true_violations = (node_y == 1).sum().item()
        true_positives = ((predictions == 1) & (node_y == 1)).sum().item()
        false_positives = ((predictions == 1) & (node_y == 0)).sum().item()
        false_negatives = ((predictions == 0) & (node_y == 1)).sum().item()

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        gt_stats = {
            "true_violations": int(true_violations),
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

        print(f"\n{'=' * 70}")
        print(f"GROUND TRUTH COMPARISON")
        print(f"{'=' * 70}")
        print(f"True violations: {true_violations}")
        print(f"True Positives:  {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
    else:
        print(f"\n{'=' * 70}")
        print("GROUND TRUTH COMPARISON")
        print(f"{'=' * 70}")
        print("Skipped: pass --axe /path/to/page-0_home.json to compare against axe labels.")

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        full_report = {
            "html_path": str(html_path),
            "model_path": str(model_path),
            "summary": {
                "total_nodes": len(page.node_map),
                "actionable_nodes": int(actionable_mask.sum().item()),
                "graph_violation_probability": round(graph_prob, 4),
                "node_evidence_probability": round(node_evidence_prob, 4),
                "page_violation_probability": round(page_violation_prob, 4),
                "graph_threshold": graph_threshold,
                "page_prediction": page_prediction,
                "node_predictions_suppressed_by_page_gate": raw_over_threshold > 0 and not page_is_likely_violation,
                "node_predictions_gated": raw_over_threshold > 0 and not page_is_likely_violation,
                "predicted_violation_count": len(predicted_violations),
                "candidate_warning_count": len(candidate_warnings),
                "threshold": node_threshold,
                "node_threshold": node_threshold,
                "rule_threshold": rule_threshold,
                "graph_source": getattr(page.data, "graph_source", args.graph_source),
            },
            "predicted_violations": predicted_violations,
            "candidate_warnings": candidate_warnings,
            "raw_top_predictions": raw_top_predictions,
            "axe_summary": axe_summary,
        }

        if gt_stats:
            full_report["ground_truth"] = gt_stats
        else:
            full_report["ground_truth"] = {
                "available": False,
                "reason": "No axe report was provided; dummy all-zero labels were not used for metrics.",
            }

        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"\nSaved detailed report to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
