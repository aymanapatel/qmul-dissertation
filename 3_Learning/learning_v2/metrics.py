"""Detection metrics with masks and per-rule detail."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PredictionArrays:
    node_probs: torch.Tensor
    node_labels: torch.Tensor
    rule_probs: torch.Tensor
    rule_labels: torch.Tensor
    page_probs: torch.Tensor
    page_labels: torch.Tensor
    valid_node_mask: torch.Tensor | None = None


def _binary(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = predictions.bool().flatten()
    labels = labels.bool().flatten()
    tp = int((predictions & labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())
    tn = int((~predictions & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": tp + fp,
        "actual_positive": tp + fn,
    }


def node_probability(rule_probabilities: torch.Tensor) -> torch.Tensor:
    return rule_probabilities.max(dim=1).values


def page_probability(node_probabilities: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    values = []
    for graph_index in range(int(batch.max().item()) + 1 if batch.numel() else 0):
        scores = node_probabilities[batch == graph_index]
        values.append(scores.max() if scores.numel() else torch.tensor(0.0, device=node_probabilities.device))
    return torch.stack(values) if values else torch.empty(0, device=node_probabilities.device)


@torch.no_grad()
def collect_predictions(model, loader, device: str) -> PredictionArrays:
    model.eval()
    rule_probs = []
    rule_labels = []
    node_probs = []
    node_labels = []
    page_probs = []
    page_labels = []
    valid_masks = []
    for data in loader:
        data = data.to(device)
        probabilities = torch.sigmoid(model(data.x, data.edge_index, data.tag_indices))
        valid = getattr(data, "label_mask", torch.ones(data.num_nodes, dtype=torch.bool, device=device)).bool()
        node_probability_values = node_probability(probabilities)
        node_probs.append(node_probability_values.cpu())
        node_labels.append(data.node_y.cpu())
        rule_probs.append(probabilities.cpu())
        rule_labels.append(data.rule_y.cpu())
        valid_masks.append(valid.cpu())
        batch = getattr(data, "batch", torch.zeros(data.num_nodes, dtype=torch.long, device=device))
        page_probs.append(page_probability(node_probability_values.masked_fill(~valid, 0.0), batch).cpu())
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        for graph_index in range(graph_count):
            mask = batch == graph_index
            page_labels.append((data.node_y[mask].bool() & valid[mask]).any().float().reshape(1).cpu())
    return PredictionArrays(
        node_probs=torch.cat(node_probs), node_labels=torch.cat(node_labels),
        rule_probs=torch.cat(rule_probs), rule_labels=torch.cat(rule_labels),
        page_probs=torch.cat(page_probs), page_labels=torch.cat(page_labels),
        valid_node_mask=torch.cat(valid_masks),
    )


def metrics_from_predictions(
    arrays: PredictionArrays,
    *,
    node_threshold: float = 0.5,
    rule_thresholds: torch.Tensor | float = 0.5,
    page_threshold: float = 0.5,
    rule_ids: list[str] | None = None,
) -> dict:
    valid = arrays.valid_node_mask
    if valid is None:
        valid = torch.ones(arrays.node_labels.shape[0], dtype=torch.bool)
    node = _binary(arrays.node_probs[valid] >= node_threshold, arrays.node_labels[valid])
    thresholds = torch.as_tensor(rule_thresholds)
    predicted_rules = arrays.rule_probs >= thresholds
    valid_rules = valid[:, None].expand_as(predicted_rules)
    rule = _binary(predicted_rules[valid_rules], arrays.rule_labels[valid_rules])
    page = _binary(arrays.page_probs >= page_threshold, arrays.page_labels)
    result = {**{f"node_{k}": v for k, v in node.items()}, **{f"rule_{k}": v for k, v in rule.items()}, **{f"page_{k}": v for k, v in page.items()}}
    if rule_ids:
        result["per_rule"] = {
            rule_id: _binary(predicted_rules[valid, index], arrays.rule_labels[valid, index])
            for index, rule_id in enumerate(rule_ids)
        }
        supported = [entry["f1"] for entry in result["per_rule"].values() if entry["actual_positive"]]
        result["rule_macro_f1_supported"] = sum(supported) / len(supported) if supported else 0.0
    return result

