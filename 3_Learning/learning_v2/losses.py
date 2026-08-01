"""Sparse multi-label loss with an optional valid-node mask."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sampled_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    negative_ratio: float = 8.0,
    minimum_negatives: int = 1024,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    allowed = torch.ones_like(targets, dtype=torch.bool)
    if valid_mask is not None:
        if valid_mask.dim() == 1:
            valid_mask = valid_mask[:, None].expand_as(targets)
        if valid_mask.shape != targets.shape:
            raise ValueError("valid_mask is incompatible with targets")
        allowed &= valid_mask.bool()
    positive = targets.bool() & allowed
    negative = ~targets.bool() & allowed
    positive_indices = positive.flatten().nonzero(as_tuple=False).flatten()
    negative_indices = negative.flatten().nonzero(as_tuple=False).flatten()
    desired_negatives = max(minimum_negatives, int(positive_indices.numel() * negative_ratio))
    desired_negatives = min(desired_negatives, int(negative_indices.numel()))
    if desired_negatives and desired_negatives < negative_indices.numel():
        order = torch.randperm(negative_indices.numel(), device=negative_indices.device)[:desired_negatives]
        negative_indices = negative_indices[order]
    else:
        negative_indices = negative_indices[:desired_negatives]
    selected = torch.cat([positive_indices, negative_indices])
    if not selected.numel():
        return logits.sum() * 0.0, {"positive_pairs": 0, "negative_pairs": 0}
    loss = F.binary_cross_entropy_with_logits(logits.flatten()[selected], targets.flatten()[selected].float())
    return loss, {
        "positive_pairs": int(positive_indices.numel()),
        "negative_pairs": int(negative_indices.numel()),
    }

