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
from graph_sources import GRAPH_SOURCE_A11Y_TREE, GRAPH_SOURCE_DOM, GRAPH_SOURCE_RENDERED_VISUAL
from models import DOMAttentionNet
from wcag_rules import (
    INDEX_TO_RULE,
    NUM_RULES,
    graph_source_for_rule,
    rule_mask_for_graph_source,
    rule_wcag_ids,
)


ARCHITECTURE_SINGLE = "single"
ARCHITECTURE_MULTI_VIEW = "multi-view"

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


def _attr_value(node, name: str) -> str:
    value = node.attrs.get(name, "")
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _role(node) -> str:
    return _attr_value(node, "role").lower().strip()


def _input_type(node) -> str:
    return _attr_value(node, "type").lower().strip()


def is_link_like(node) -> bool:
    role = _role(node)
    return (
        node.tag == "a"
        or node.tag == "area"
        or role == "link"
    )


def is_button_like(node) -> bool:
    role = _role(node)
    input_type = _input_type(node)
    return (
        node.tag == "button"
        or role == "button"
        or (node.tag == "input" and input_type in {"button", "submit", "reset"})
    )


def rule_is_compatible_with_node(rule_id: str, node) -> bool:
    """Apply cheap rule/tag constraints before reporting a model prediction.

    The GNN can assign high probability to the wrong rule family on a visually
    similar node. These constraints keep post-processing aligned with axe's
    rule preconditions without using axe labels at prediction time.
    """
    role = _role(node)
    input_type = _input_type(node)
    has_text = bool((node.text_content or "").strip())

    if rule_id == "link-name":
        return is_link_like(node)
    if rule_id == "area-alt":
        return node.tag == "area"
    if rule_id == "button-name":
        return is_button_like(node)
    if rule_id == "input-button-name":
        return node.tag == "input" and input_type in {"button", "submit", "reset"}
    if rule_id == "input-image-alt":
        return node.tag == "input" and input_type == "image"
    if rule_id == "image-alt":
        return node.tag == "img"
    if rule_id == "role-img-alt":
        return role == "img"
    if rule_id == "svg-img-alt":
        return node.tag == "svg"
    if rule_id == "object-alt":
        return node.tag in {"object", "embed"}
    if rule_id == "frame-title":
        return node.tag in {"iframe", "frame"}
    if rule_id in {"select-name"}:
        return node.tag == "select"
    if rule_id == "label":
        return node.tag in {"input", "select", "textarea"}
    if rule_id in {"summary-name"}:
        return node.tag == "summary"
    if rule_id in {"aria-command-name"}:
        return role in {"button", "link", "menuitem", "menuitemcheckbox", "menuitemradio"}
    if rule_id in {"aria-input-field-name"}:
        return role in {"combobox", "listbox", "searchbox", "slider", "spinbutton", "textbox"}
    if rule_id in {"aria-toggle-field-name"}:
        return role in {"checkbox", "radio", "switch"}
    if rule_id in {"aria-meter-name"}:
        return role == "meter"
    if rule_id in {"aria-progressbar-name"}:
        return role == "progressbar"
    if rule_id in {"aria-tooltip-name"}:
        return role == "tooltip"
    if rule_id == "list":
        return node.tag in {"ul", "ol", "menu"}
    if rule_id == "listitem":
        return node.tag == "li" or role == "listitem"
    if rule_id in {"definition-list"}:
        return node.tag == "dl"
    if rule_id == "dlitem":
        return node.tag in {"dt", "dd"}
    if rule_id == "color-contrast":
        return has_text
    if rule_id == "link-in-text-block":
        return is_link_like(node) and has_text
    if rule_id == "meta-viewport":
        return node.tag == "meta"
    if rule_id == "scrollable-region-focusable":
        return node.tag in ACTIONABLE_TAGS
    return True


def get_top_rules(
    rule_probs,
    top_k=3,
    graph_source=None,
    node=None,
    threshold=0.1,
    rule_thresholds: dict[str, float] | None = None,
):
    """Get top-k predicted rules for a node."""
    if graph_source:
        mask = rule_mask_for_graph_source(graph_source)
        rule_probs = rule_probs.clone()
        rule_probs[~mask] = 0.0
    rules = []
    for idx in rule_probs.argsort(descending=True):
        probability = float(rule_probs[idx].item())
        rule_id = INDEX_TO_RULE[idx.item()]
        rule_threshold = threshold
        if rule_thresholds and rule_id in rule_thresholds:
            rule_threshold = float(rule_thresholds[rule_id])
        if probability <= rule_threshold:
            continue
        if node is not None and not rule_is_compatible_with_node(rule_id, node):
            continue
        rules.append(
            {
                "rule_id": rule_id,
                "axe_rule_id": rule_id,
                "probability": round(probability, 4),
                "threshold": round(rule_threshold, 4),
                "source_view": graph_source_for_rule(rule_id),
                "wcag_ids": list(rule_wcag_ids(rule_id)),
            }
        )
        if len(rules) >= top_k:
            break
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


def _threshold_from_mapping(mapping: dict, key: str, default: float) -> float:
    if not isinstance(mapping, dict):
        return default
    thresholds = mapping.get("thresholds", {})
    if isinstance(thresholds, dict) and key in thresholds:
        return float(thresholds[key])
    if key in mapping:
        return float(mapping[key])
    recommended = mapping.get("recommended", {})
    if isinstance(recommended, dict) and key in recommended:
        return float(recommended[key])
    return default


def view_thresholds(
    args,
    calibration: dict,
    manifest: dict | None,
    source_view: str,
    base_node_threshold: float,
    base_graph_threshold: float,
    base_rule_threshold: float,
) -> dict:
    """Resolve global, per-view, and per-rule thresholds for a prediction view."""
    view_manifest = ((manifest or {}).get("views", {}) or {}).get(source_view, {})
    view_calibration = (calibration.get("views", {}) or {}).get(source_view, {})
    legacy_view_calibration = (calibration.get("view_thresholds", {}) or {}).get(source_view, {})
    calibration_graph_source = calibration.get("graph_source")
    has_view_calibration = bool(calibration.get("views") or calibration.get("view_thresholds"))
    scoped_calibration = {}
    if calibration and not has_view_calibration:
        if calibration_graph_source is None or calibration_graph_source == source_view:
            scoped_calibration = calibration

    node_threshold = base_node_threshold
    graph_threshold = base_graph_threshold
    rule_threshold = base_rule_threshold

    for source in (view_manifest, scoped_calibration, legacy_view_calibration, view_calibration):
        node_threshold = _threshold_from_mapping(source, "node_threshold", node_threshold)
        graph_threshold = _threshold_from_mapping(source, "graph_threshold", graph_threshold)
        rule_threshold = _threshold_from_mapping(source, "rule_threshold", rule_threshold)

    if args.threshold is not None:
        node_threshold = args.threshold
    if args.graph_threshold is not None:
        graph_threshold = args.graph_threshold
    if args.rule_threshold is not None:
        rule_threshold = args.rule_threshold

    rule_thresholds: dict[str, float] = {}
    if args.rule_threshold is None:
        for source in (manifest or {}, view_manifest, scoped_calibration, legacy_view_calibration, view_calibration):
            direct = source.get("rule_thresholds", {}) if isinstance(source, dict) else {}
            if isinstance(direct, dict):
                rule_thresholds.update({str(rule): float(value) for rule, value in direct.items()})
            rules = source.get("rules", {}) if isinstance(source, dict) else {}
            if isinstance(rules, dict):
                for rule, value in rules.items():
                    if isinstance(value, dict):
                        rule_thresholds[str(rule)] = _threshold_from_mapping(value, "rule_threshold", rule_threshold)
                    else:
                        rule_thresholds[str(rule)] = float(value)

    return {
        "node_threshold": float(node_threshold),
        "graph_threshold": float(graph_threshold),
        "rule_threshold": float(rule_threshold),
        "rule_thresholds": rule_thresholds,
    }


def checkpoint_feature_dims(checkpoint: dict, fallback_attr_dim: int, fallback_text_dim: int) -> tuple[int, int]:
    """Infer the feature dimensions a checkpoint expects."""
    hparams = checkpoint.get("hparams", {})
    state = checkpoint.get("model_state_dict", {})
    tag_embed_dim = int(hparams.get("tag_embed_dim", 32))
    text_dim = int(hparams.get("text_dim", fallback_text_dim))
    attr_dim = hparams.get("attr_dim")
    if attr_dim is not None:
        return int(attr_dim), text_dim
    input_weight = state.get("input_proj.weight")
    if input_weight is not None:
        return int(input_weight.shape[1] - tag_embed_dim - text_dim), text_dim
    return fallback_attr_dim, text_dim


def align_data_features(data, expected_attr_dim: int, expected_text_dim: int):
    """Pad or trim extracted features to the width expected by a checkpoint."""
    current_text_dim = (
        data.text_embeddings.shape[1]
        if hasattr(data, "text_embeddings") and data.text_embeddings is not None
        else expected_text_dim
    )
    current_attr_dim = data.x.shape[1] - current_text_dim
    attr = data.x[:, :current_attr_dim]
    text = data.x[:, current_attr_dim:]

    if current_attr_dim > expected_attr_dim:
        attr = attr[:, :expected_attr_dim]
    elif current_attr_dim < expected_attr_dim:
        pad = torch.zeros(attr.shape[0], expected_attr_dim - current_attr_dim, dtype=attr.dtype)
        attr = torch.cat([attr, pad], dim=-1)

    if current_text_dim > expected_text_dim:
        text = text[:, :expected_text_dim]
    elif current_text_dim < expected_text_dim:
        pad = torch.zeros(text.shape[0], expected_text_dim - current_text_dim, dtype=text.dtype)
        text = torch.cat([text, pad], dim=-1)

    data.x = torch.cat([attr, text], dim=-1)
    return data


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
    node,
    idx: int,
    prob: float,
    rule_probs,
    node_threshold: float,
    status: str,
    source_view: str | None = None,
    rule_threshold: float = 0.1,
    rule_thresholds: dict[str, float] | None = None,
) -> dict:
    """Build a JSON-safe prediction/candidate entry."""
    predicted_rules = get_top_rules(
        rule_probs,
        top_k=3,
        graph_source=source_view,
        node=node,
        threshold=rule_threshold,
        rule_thresholds=rule_thresholds,
    )
    primary_rule = predicted_rules[0] if predicted_rules else {}
    return {
        "node_id": idx,
        "source_view": source_view,
        "axe_rule_id": primary_rule.get("axe_rule_id"),
        "wcag_ids": primary_rule.get("wcag_ids", []),
        "rule_probability": primary_rule.get("probability"),
        "tag": node.tag,
        "dom_path": getattr(node, "dom_path", ""),
        "probability": round(prob, 4),
        "predicted_violation": status == "predicted_violation",
        "status": status,
        "attributes": dict(node.attrs),
        "text_preview": node.text_content[:100] if node.text_content else "",
        "visual": getattr(node, "visual", {}),
        "label_qa": list(getattr(node, "visual_label_qa", [])),
        "predicted_rules": predicted_rules,
        "node_threshold": node_threshold,
        "rule_threshold": rule_threshold,
        "rule_threshold_overrides": rule_thresholds or {},
    }


def build_model_from_checkpoint(checkpoint: dict, attr_dim: int, text_dim: int) -> DOMAttentionNet:
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
    return model


def page_thresholds(args, calibration: dict, manifest: dict | None = None) -> tuple[float, float, float]:
    manifest_thresholds = (manifest or {}).get("thresholds", {})
    effective_calibration = calibration
    if manifest and calibration.get("graph_source") and not (
        calibration.get("views") or calibration.get("view_thresholds")
    ):
        effective_calibration = {}
    node_threshold = (
        args.threshold
        if args.threshold is not None
        else calibration_value(effective_calibration, "node_threshold", manifest_thresholds.get("node_threshold", 0.5))
    )
    graph_threshold = (
        args.graph_threshold
        if args.graph_threshold is not None
        else calibration_value(effective_calibration, "graph_threshold", manifest_thresholds.get("graph_threshold", 0.5))
    )
    rule_threshold = (
        args.rule_threshold
        if args.rule_threshold is not None
        else calibration_value(effective_calibration, "rule_threshold", manifest_thresholds.get("rule_threshold", 0.5))
    )
    return node_threshold, graph_threshold, rule_threshold


def run_multi_view_prediction(args, html_path: Path, axe_path: Path | None, calibration: dict, device: str) -> None:
    bundle_dir = Path(args.model_bundle)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: multi-view manifest not found: {manifest_path}")
        sys.exit(1)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    base_node_threshold, base_graph_threshold, base_rule_threshold = page_thresholds(args, calibration, manifest)
    axe_summary = load_axe_summary(axe_path)
    extractor = FeatureExtractor(device=device)

    predicted_violations = []
    candidate_warnings = []
    raw_top_predictions = []
    view_summaries = {}
    gt_totals = {
        "true_violations": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    page_evidence_prob = 0.0
    page_graph_prob = 0.0

    for source_view, view in manifest.get("views", {}).items():
        checkpoint_path = Path(view["checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = bundle_dir / source_view / "best_model.pt"
        print(f"\n[{source_view}] Loading {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        page = extractor.process_page(
            html_path=html_path,
            axe_report_path=axe_path,
            extract_visual=source_view == GRAPH_SOURCE_RENDERED_VISUAL,
            graph_source=source_view,
        )
        text_dim = (
            page.data.text_embeddings.shape[1]
            if hasattr(page.data, "text_embeddings") and page.data.text_embeddings is not None
            else 384
        )
        attr_dim = page.data.x.shape[1] - text_dim
        expected_attr_dim, expected_text_dim = checkpoint_feature_dims(
            checkpoint,
            fallback_attr_dim=attr_dim,
            fallback_text_dim=text_dim,
        )
        page.data = align_data_features(page.data, expected_attr_dim, expected_text_dim)
        model = build_model_from_checkpoint(checkpoint, attr_dim=expected_attr_dim, text_dim=expected_text_dim)
        results = predict_page(model, page.data, device=device)
        thresholds = view_thresholds(
            args=args,
            calibration=calibration,
            manifest=manifest,
            source_view=source_view,
            base_node_threshold=base_node_threshold,
            base_graph_threshold=base_graph_threshold,
            base_rule_threshold=base_rule_threshold,
        )
        node_threshold = thresholds["node_threshold"]
        graph_threshold = thresholds["graph_threshold"]
        rule_threshold = thresholds["rule_threshold"]
        rule_threshold_overrides = thresholds["rule_thresholds"]

        node_probs = results["node_probs"]
        rule_probs = results["rule_probs"]
        graph_prob = results["graph_violation_prob"]
        page_graph_prob = max(page_graph_prob, graph_prob)

        actionable_mask = torch.zeros(len(page.node_map), dtype=torch.bool)
        for nid, node in page.node_map.items():
            actionable_mask[nid] = node.tag in ACTIONABLE_TAGS
        filtered_probs = node_probs.clone()
        filtered_probs[~actionable_mask] = 0.0
        if source_view == GRAPH_SOURCE_RENDERED_VISUAL and hasattr(page.data, "rendered_visible_mask"):
            filtered_probs[~page.data.rendered_visible_mask.cpu().bool()] = 0.0

        view_node_evidence = filtered_probs.max().item() if filtered_probs.numel() else 0.0
        page_evidence_prob = max(page_evidence_prob, view_node_evidence)
        sorted_indices = filtered_probs.argsort(descending=True)
        raw_over_threshold = int((filtered_probs >= node_threshold).sum().item())
        compatible_predicted_node_ids = set()

        for rank, idx in enumerate(sorted_indices, 1):
            idx = idx.item()
            prob = filtered_probs[idx].item()
            if prob < 0.2:
                break
            node = page.node_map.get(idx)
            if not node:
                continue
            entry = make_prediction_entry(
                node=node,
                idx=idx,
                prob=prob,
                rule_probs=rule_probs[idx],
                node_threshold=node_threshold,
                status="raw_candidate",
                source_view=source_view,
                rule_threshold=rule_threshold,
                rule_thresholds=rule_threshold_overrides,
            )
            entry["rank"] = rank

            if prob >= node_threshold:
                if entry["predicted_rules"]:
                    entry["status"] = "predicted_violation"
                    entry["predicted_violation"] = True
                    predicted_violations.append(entry)
                    compatible_predicted_node_ids.add(idx)
                else:
                    entry["status"] = "candidate_incompatible_rule"
                    entry["predicted_violation"] = False
                    candidate_warnings.append(entry)
            else:
                candidate_warnings.append(entry)

            if len(raw_top_predictions) < args.top_k:
                raw_top_predictions.append(entry)

        if axe_path and getattr(page.data, "has_ground_truth", False):
            predictions = torch.zeros(len(page.node_map), dtype=torch.long)
            for nid in compatible_predicted_node_ids:
                predictions[nid] = 1
            labels = page.data.node_y.cpu().long()
            gt_totals["true_violations"] += int((labels == 1).sum().item())
            gt_totals["true_positives"] += int(((predictions == 1) & (labels == 1)).sum().item())
            gt_totals["false_positives"] += int(((predictions == 1) & (labels == 0)).sum().item())
            gt_totals["false_negatives"] += int(((predictions == 0) & (labels == 1)).sum().item())

        view_summaries[source_view] = {
            "total_nodes": len(page.node_map),
            "actionable_nodes": int(actionable_mask.sum().item()),
            "graph_violation_probability": round(graph_prob, 4),
            "node_evidence_probability": round(view_node_evidence, 4),
            "nodes_over_threshold": raw_over_threshold,
            "rule_compatible_nodes_over_threshold": len(compatible_predicted_node_ids),
            "node_threshold": node_threshold,
            "graph_threshold": graph_threshold,
            "rule_threshold": rule_threshold,
            "rule_threshold_overrides": rule_threshold_overrides,
            "expected_attr_dim": expected_attr_dim,
            "expected_text_dim": expected_text_dim,
            "rule_ids": view.get("rule_ids", []),
        }

    page_violation_prob = max(page_graph_prob, page_evidence_prob)
    page_is_likely_violation = page_violation_prob >= base_graph_threshold

    precision = (
        gt_totals["true_positives"] / (gt_totals["true_positives"] + gt_totals["false_positives"])
        if (gt_totals["true_positives"] + gt_totals["false_positives"])
        else 0
    )
    recall = (
        gt_totals["true_positives"] / (gt_totals["true_positives"] + gt_totals["false_negatives"])
        if (gt_totals["true_positives"] + gt_totals["false_negatives"])
        else 0
    )

    print("\n" + "=" * 70)
    print("MULTI-VIEW PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Page violation probability: {page_violation_prob:.4f} (threshold={base_graph_threshold:.4f})")
    print(f"Predicted violations: {len(predicted_violations)}")
    print(f"Candidate warnings: {len(candidate_warnings)}")
    if axe_path:
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_report = {
            "html_path": str(html_path),
            "model_bundle": str(bundle_dir),
            "summary": {
                "architecture": ARCHITECTURE_MULTI_VIEW,
                "page_violation_probability": round(page_violation_prob, 4),
                "graph_violation_probability": round(page_graph_prob, 4),
                "node_evidence_probability": round(page_evidence_prob, 4),
                "graph_threshold": base_graph_threshold,
                "page_prediction": "likely_violation" if page_is_likely_violation else "likely_clean",
                "page_gate_applied_to_nodes": False,
                "predicted_violation_count": len(predicted_violations),
                "candidate_warning_count": len(candidate_warnings),
                "node_threshold": base_node_threshold,
                "rule_threshold": base_rule_threshold,
                "views": view_summaries,
            },
            "predicted_violations": predicted_violations,
            "candidate_warnings": candidate_warnings,
            "raw_top_predictions": raw_top_predictions,
            "axe_summary": axe_summary,
            "ground_truth": {
                **gt_totals,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "available": bool(axe_path),
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"\nSaved detailed report to {output_path}")


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
        "--architecture",
        type=str,
        default=ARCHITECTURE_SINGLE,
        choices=[ARCHITECTURE_SINGLE, ARCHITECTURE_MULTI_VIEW],
        help="Prediction architecture: legacy single model or multi-view bundle",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to trained single-view model .pt file"
    )
    parser.add_argument(
        "--model-bundle",
        type=str,
        default=None,
        help="Path to multi-view model bundle directory containing manifest.json",
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
        choices=[GRAPH_SOURCE_DOM, GRAPH_SOURCE_A11Y_TREE, GRAPH_SOURCE_RENDERED_VISUAL],
        help="Graph source to build for --architecture single",
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
    model_path = Path(args.model) if args.model else None
    axe_path = Path(args.axe) if args.axe else None
    calibration = load_calibration(Path(args.calibration)) if args.calibration else {}

    if not html_path.exists():
        print(f"Error: HTML file not found: {html_path}")
        sys.exit(1)
    if args.architecture == ARCHITECTURE_SINGLE and (not model_path or not model_path.exists()):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    if args.architecture == ARCHITECTURE_MULTI_VIEW and not args.model_bundle:
        print("Error: --model-bundle is required for --architecture multi-view")
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

    if args.architecture == ARCHITECTURE_MULTI_VIEW:
        print("=" * 70)
        print("Accessibility Violation Prediction (Multi-View)")
        print("=" * 70)
        print(f"HTML: {html_path}")
        print(f"Model bundle: {args.model_bundle}")
        print(f"Device: {device}")
        if calibration:
            print(f"Calibration: {args.calibration}")
        run_multi_view_prediction(args, html_path, axe_path, calibration, device)
        print("\nDone!")
        return

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

    expected_attr_dim, expected_text_dim = checkpoint_feature_dims(
        checkpoint,
        fallback_attr_dim=attr_dim,
        fallback_text_dim=text_dim,
    )
    page.data = align_data_features(page.data, expected_attr_dim, expected_text_dim)
    model = build_model_from_checkpoint(checkpoint, attr_dim=expected_attr_dim, text_dim=expected_text_dim)
    print(
        f"Loaded model (GAT-{hparams.get('hidden_dim', 256)}x{hparams.get('num_layers', 4)}, {hparams.get('num_rules', NUM_RULES)} rules)"
    )

    # Run prediction
    print("\nRunning prediction...")
    results = predict_page(model, page.data, device=device)

    node_probs = results["node_probs"]
    rule_probs = results["rule_probs"]
    graph_prob = results["graph_violation_prob"]
    base_node_threshold, base_graph_threshold, base_rule_threshold = page_thresholds(
        args,
        calibration,
        None,
    )
    thresholds = view_thresholds(
        args=args,
        calibration=calibration,
        manifest=None,
        source_view=args.graph_source,
        base_node_threshold=base_node_threshold,
        base_graph_threshold=base_graph_threshold,
        base_rule_threshold=base_rule_threshold,
    )
    node_threshold = thresholds["node_threshold"]
    graph_threshold = thresholds["graph_threshold"]
    rule_threshold = thresholds["rule_threshold"]
    rule_threshold_overrides = thresholds["rule_thresholds"]
    axe_summary = load_axe_summary(axe_path)

    # Filter to actionable tags
    actionable_mask = torch.zeros(len(page.node_map), dtype=torch.bool)
    for nid, node in page.node_map.items():
        if node.tag in ACTIONABLE_TAGS:
            actionable_mask[nid] = True

    filtered_probs = node_probs.clone()
    filtered_probs[~actionable_mask] = 0.0
    if args.visual and hasattr(page.data, "rendered_visible_mask"):
        filtered_probs[~page.data.rendered_visible_mask.cpu().bool()] = 0.0
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
    confirmed_over_threshold = raw_over_threshold
    print(f"Nodes with violation prob >= {node_threshold}: {raw_over_threshold}")
    print(f"Confirmed node candidates before rule compatibility: {confirmed_over_threshold}")

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
    compatible_predicted_node_ids = set()
    for rank, idx in enumerate(sorted_indices, 1):
        idx = idx.item()
        prob = filtered_probs[idx].item()
        if prob < 0.2:
            break
        node = page.node_map.get(idx)

        if not node:
            continue

        tag = node.tag[:12]
        top_rules = get_top_rules(
            rule_probs[idx],
            top_k=3,
            graph_source=args.graph_source,
            node=node,
            threshold=rule_threshold,
            rule_thresholds=rule_threshold_overrides,
        )
        rule_str = (
            ", ".join([f"{r['rule_id']}({r['probability']:.2f})" for r in top_rules])
            if top_rules
            else "none"
        )

        over_node_threshold = prob >= node_threshold
        confirmed_violation = over_node_threshold and bool(top_rules)
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
                source_view=args.graph_source,
                rule_threshold=rule_threshold,
                rule_thresholds=rule_threshold_overrides,
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
                    source_view=args.graph_source,
                    rule_threshold=rule_threshold,
                    rule_thresholds=rule_threshold_overrides,
                )
            )
            compatible_predicted_node_ids.add(idx)
        elif prob >= 0.2:
            status = "candidate_warning"
            if over_node_threshold and not top_rules:
                status = "candidate_incompatible_rule"
            candidate_warnings.append(
                make_prediction_entry(
                    node=node,
                    idx=idx,
                    prob=prob,
                    rule_probs=rule_probs[idx],
                    node_threshold=node_threshold,
                    status=status,
                    source_view=args.graph_source,
                    rule_threshold=rule_threshold,
                    rule_thresholds=rule_threshold_overrides,
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
        predictions = torch.zeros(len(page.node_map), dtype=torch.long)
        for nid in compatible_predicted_node_ids:
            predictions[nid] = 1
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
                "architecture": ARCHITECTURE_SINGLE,
                "total_nodes": len(page.node_map),
                "actionable_nodes": int(actionable_mask.sum().item()),
                "graph_violation_probability": round(graph_prob, 4),
                "node_evidence_probability": round(node_evidence_prob, 4),
                "page_violation_probability": round(page_violation_prob, 4),
                "graph_threshold": graph_threshold,
                "page_prediction": page_prediction,
                "page_gate_applied_to_nodes": False,
                "node_predictions_suppressed_by_page_gate": False,
                "node_predictions_gated": False,
                "predicted_violation_count": len(predicted_violations),
                "candidate_warning_count": len(candidate_warnings),
                "threshold": node_threshold,
                "node_threshold": node_threshold,
                "rule_threshold": rule_threshold,
                "rule_threshold_overrides": rule_threshold_overrides,
                "expected_attr_dim": expected_attr_dim,
                "expected_text_dim": expected_text_dim,
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
