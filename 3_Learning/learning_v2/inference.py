"""Axe-free inference with complete evaluation output and capped display output."""

from __future__ import annotations

from typing import Any

import torch

from .rules import rule_metadata
from .schema import FeatureContract


def _attribute(node: Any, name: str) -> str:
    value = getattr(node, "attrs", {}).get(name, "")
    return " ".join(map(str, value)) if isinstance(value, list) else str(value)


def compatible(rule_id: str, node: Any) -> bool:
    tag = getattr(node, "tag", ""); text = (getattr(node, "text_content", "") or "").strip(); role = _attribute(node, "role").lower()
    if rule_id == "color-contrast": return bool(text)
    if rule_id == "link-in-text-block": return (tag in {"a", "area"} or role == "link") and bool(text)
    if rule_id == "meta-viewport": return tag == "meta"
    return True


@torch.no_grad()
def predict_graph(*, model, graph, node_map: dict, checkpoint: dict, calibration: dict, contract: FeatureContract, device: str, top_k: int) -> dict:
    contract.validate(graph); model.eval(); graph = graph.to(device)
    probabilities = torch.sigmoid(model(graph.x, graph.edge_index, graph.tag_indices)).cpu()
    rule_ids = list(checkpoint["rule_ids"]); recommended = calibration["recommended"]
    node_threshold = float(recommended["node_threshold"]); page_threshold = float(recommended["page_threshold"])
    rule_thresholds = {rule_id: float(recommended["rule_thresholds"][rule_id]) for rule_id in rule_ids}
    visible = getattr(graph, "rendered_visible_mask", torch.ones(probabilities.shape[0], dtype=torch.bool, device=device)).cpu()
    node_scores = probabilities.max(dim=1).values; node_scores[~visible] = 0.0
    page_score = float(node_scores.max()) if node_scores.numel() else 0.0; page_predicted = page_score >= page_threshold
    all_predictions = []; candidates = []
    for rank, tensor_index in enumerate(node_scores.argsort(descending=True), 1):
        node_index = int(tensor_index); score = float(node_scores[node_index])
        if score < min(0.2, node_threshold): break
        node = node_map.get(node_index)
        if node is None: continue
        matched = []
        for local_index_tensor in probabilities[node_index].argsort(descending=True):
            local_index = int(local_index_tensor); rule_id = rule_ids[local_index]; probability = float(probabilities[node_index, local_index])
            if probability >= rule_thresholds[rule_id] and compatible(rule_id, node):
                matched.append({**rule_metadata(rule_id), "probability": round(probability, 6), "threshold": rule_thresholds[rule_id]})
        entry = {
            "rank": rank, "node_id": node_index, "tag": getattr(node, "tag", ""),
            "dom_path": getattr(node, "dom_path", ""), "text_preview": (getattr(node, "text_content", "") or "")[:160],
            "attributes": dict(getattr(node, "attrs", {})), "node_probability": round(score, 6),
            "node_threshold": node_threshold, "predicted_rules": matched,
        }
        if len(candidates) < top_k: candidates.append(entry)
        if page_predicted and score >= node_threshold and matched: all_predictions.append(entry)
    return {
        "schema_version": 2, "graph_source": contract.graph_source, "axe_used_for_prediction": False,
        "num_nodes": int(probabilities.shape[0]), "rule_ids": rule_ids,
        "thresholds": {"node": node_threshold, "page": page_threshold, "rules": rule_thresholds},
        "page": {"violation_probability": round(page_score, 6), "predicted_violation": page_predicted},
        "prediction_count": len(all_predictions), "predictions": all_predictions,
        "display_top_k": top_k, "top_candidates": candidates,
    }

