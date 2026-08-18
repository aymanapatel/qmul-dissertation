"""Checkpoint helpers and a compact node-rule trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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


@torch.no_grad()
def validation_loss(
    model,
    loader,
    config: TrainingConfig,
    device: str,
    *,
    sampling_seed: int,
) -> dict:
    """Calculate full and training-matched validation BCE in one model pass.

    The sampled loss uses the training negative ratio and minimum but resets a
    CPU random generator to the same seed on every call. Consequently, every
    epoch is evaluated against the same validation pairs on every device. The
    full valid-pair BCE is retained as a deterministic audit measure.
    """
    model.eval()
    full_loss_sum = 0.0
    full_pairs = 0
    full_positive_pairs = 0
    full_negative_pairs = 0
    sampled_loss_sum = 0.0
    sampled_batches = 0
    sampled_positive_pairs = 0
    sampled_negative_pairs = 0
    generator = torch.Generator(device="cpu").manual_seed(sampling_seed)

    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.tag_indices)
        targets = data.rule_y
        if logits.shape != targets.shape:
            raise ValueError("logits and validation targets must have the same shape")

        allowed = torch.ones_like(targets, dtype=torch.bool)
        label_mask = getattr(data, "label_mask", None)
        if label_mask is not None:
            if label_mask.dim() == 1:
                label_mask = label_mask[:, None].expand_as(targets)
            if label_mask.shape != targets.shape:
                raise ValueError("validation label_mask is incompatible with targets")
            allowed &= label_mask.bool()

        pair_count = int(allowed.sum().item())
        if not pair_count:
            continue
        selected_targets = targets[allowed].float()
        full_loss_sum += float(
            F.binary_cross_entropy_with_logits(
                logits[allowed],
                selected_targets,
                reduction="sum",
            ).detach()
        )
        full_pairs += pair_count
        batch_positive_pairs = int(selected_targets.bool().sum().item())
        full_positive_pairs += batch_positive_pairs
        full_negative_pairs += pair_count - batch_positive_pairs

        sampled_loss, sampled_counts = sampled_binary_cross_entropy(
            logits,
            targets,
            negative_ratio=config.negative_ratio,
            minimum_negatives=config.minimum_negatives,
            valid_mask=label_mask,
            generator=generator,
        )
        sampled_loss_sum += float(sampled_loss.detach())
        sampled_batches += 1
        sampled_positive_pairs += sampled_counts["positive_pairs"]
        sampled_negative_pairs += sampled_counts["negative_pairs"]

    return {
        # `loss` remains an alias for backward compatibility with the first
        # validation-loss history schema.
        "loss": full_loss_sum / max(1, full_pairs),
        "loss_type": "full_valid_pair_bce",
        "valid_pairs": full_pairs,
        "positive_pairs": full_positive_pairs,
        "negative_pairs": full_negative_pairs,
        "sampled_loss": sampled_loss_sum / max(1, sampled_batches),
        "sampled_loss_type": "fixed_sample_training_matched_bce",
        "sampled_batches": sampled_batches,
        "sampled_positive_pairs": sampled_positive_pairs,
        "sampled_negative_pairs": sampled_negative_pairs,
        "negative_ratio": config.negative_ratio,
        "minimum_negatives": config.minimum_negatives,
        "sampling_seed": sampling_seed,
    }
