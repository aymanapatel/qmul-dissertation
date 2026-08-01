"""Validation-only threshold calibration."""

from __future__ import annotations

import json
from pathlib import Path

import torch

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
    candidates = [_scores(probabilities, labels, value / 20) for value in range(1, 20)]
    supported = [entry for entry in candidates if entry["precision"] >= precision_floor]
    pool = supported or candidates
    best = max(pool, key=lambda entry: (entry["f1"], entry["recall"], entry["threshold"]))
    return {**best, "supported": bool(labels.bool().any())}


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

