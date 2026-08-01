"""Checkpoint helpers and a compact node-rule trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .losses import sampled_binary_cross_entropy
from .models import ModelConfig, NodeRuleModel, build_model
from .schema import CHECKPOINT_SCHEMA_VERSION, FeatureContract


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    patience: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    negative_ratio: float = 8.0
    minimum_negatives: int = 1024
    selection_metric: str = "rule_f1"


def save_checkpoint(
    path: Path, *, model: NodeRuleModel, optimizer: torch.optim.Optimizer, epoch: int,
    best_metric: float, model_config: ModelConfig, training_config: TrainingConfig,
    feature_contract: FeatureContract, rule_ids: list[str], rule_indices: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch, "best_metric": best_metric,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config.to_dict(), "training_config": asdict(training_config),
        "feature_contract": feature_contract.to_dict(), "rule_ids": rule_ids, "rule_indices": rule_indices,
    }, path)


def load_checkpoint(path: Path, *, device: str = "cpu") -> tuple[NodeRuleModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    version = checkpoint.get("checkpoint_schema_version")
    if version not in {1, CHECKPOINT_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported checkpoint schema at {path}")
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def train_epoch(model, loader, optimizer, config: TrainingConfig, device: str) -> dict:
    model.train(); total = 0.0; batches = 0; positives = 0; negatives = 0
    for data in loader:
        data = data.to(device); optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index, data.tag_indices)
        loss, counts = sampled_binary_cross_entropy(
            logits, data.rule_y, negative_ratio=config.negative_ratio,
            minimum_negatives=config.minimum_negatives, valid_mask=getattr(data, "label_mask", None),
        )
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += float(loss.detach()); batches += 1; positives += counts["positive_pairs"]; negatives += counts["negative_pairs"]
    return {"loss": total / max(1, batches), "sampled_positive_pairs": positives, "sampled_negative_pairs": negatives}
