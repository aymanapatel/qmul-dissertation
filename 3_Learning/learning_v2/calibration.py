"""Validation-only threshold calibration."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_curve

from .metrics import PredictionArrays


def _scores(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    pred = probabilities >= threshold
    labels = labels.bool()
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "predicted_positive": tp + fp, "actual_positive": tp + fn}


def choose_threshold(probabilities: torch.Tensor, labels: torch.Tensor, *, precision_floor: float = 0.0) -> dict:
    probabilities = probabilities.detach().float().cpu().flatten()
    labels = labels.detach().bool().cpu().flatten()
    label_supported = bool(labels.any())
    if not probabilities.numel() or not label_supported:
        return {
            "threshold": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "predicted_positive": 0,
            "actual_positive": int(labels.sum()),
            "supported": label_supported,
            "precision_floor_met": False,
            "candidate_count": 0,
        }

    # Evaluate every distinct validation score. This avoids the historical
    # coarse 0.05 grid, which could miss the only threshold satisfying the
    # predeclared precision floor.
    curve_precision, curve_recall, thresholds = precision_recall_curve(
        labels.numpy().astype(int), probabilities.numpy()
    )
    candidates = []
    for index, threshold in enumerate(thresholds):
        precision = float(curve_precision[index])
        recall = float(curve_recall[index])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        predicted_positive = int((probabilities >= float(threshold)).sum())
        candidates.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_positive": predicted_positive,
            "actual_positive": int(labels.sum()),
        })
    unconstrained = max(candidates, key=lambda entry: (entry["f1"], entry["recall"], entry["threshold"]))
    eligible = [
        entry for entry in candidates
        if entry["predicted_positive"] > 0 and entry["precision"] >= precision_floor
    ]
    precision_floor_met = bool(eligible)
    best = max(eligible, key=lambda entry: (entry["f1"], entry["recall"], entry["threshold"])) if eligible else unconstrained
    return {
        **best,
        "supported": True,
        "precision_floor_met": precision_floor_met,
        "candidate_count": len(candidates),
        "unconstrained_best": unconstrained,
    }


def calibrate_predictions(arrays: PredictionArrays, rule_ids: list[str], *, precision_floor: float = 0.55) -> dict:
    valid = arrays.valid_node_mask
    if valid is None:
        valid = torch.ones(arrays.node_labels.shape[0], dtype=torch.bool)
    node = choose_threshold(arrays.node_probs[valid], arrays.node_labels[valid], precision_floor=precision_floor)
    page = choose_threshold(arrays.page_probs, arrays.page_labels, precision_floor=precision_floor)
    rules = {
        rule_id: choose_threshold(arrays.rule_probs[valid, index], arrays.rule_labels[valid, index], precision_floor=precision_floor)
        for index, rule_id in enumerate(rule_ids)
    }
    return {
        "schema_version": 2,
        "precision_floor": precision_floor,
        "recommended": {
            "node_threshold": node["threshold"],
            "page_threshold": page["threshold"],
            "rule_thresholds": {rule_id: result["threshold"] for rule_id, result in rules.items()},
        },
        "node": node, "page": page, "rules": rules,
    }


def save_calibration(path: Path, calibration: dict) -> None:
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
