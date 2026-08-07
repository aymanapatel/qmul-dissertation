"""
train.py

Training and evaluation pipeline for DOM graph multi-label classification.
Supports hard negative mining, MPS (Apple GPU), and per-rule metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models import DOMAttentionNet
from wcag_rules import (
    INDEX_TO_RULE,
    NUM_RULES,
    RULE_OWNER_A11Y_TREE,
    RULE_OWNER_DOM,
    RULE_OWNER_DOM_PAGE,
    RULE_OWNER_RENDERED_VISUAL,
    rule_indices_for_owners,
)


def max_node_probability_by_graph(
    node_probs: torch.Tensor,
    batch: Optional[torch.Tensor],
    num_graphs: int,
) -> torch.Tensor:
    """Return the strongest node violation probability for each graph."""
    if batch is None:
        return node_probs.max().reshape(1)

    values = []
    for graph_idx in range(num_graphs):
        graph_node_probs = node_probs[batch == graph_idx]
        if graph_node_probs.numel() == 0:
            values.append(torch.tensor(0.0, device=node_probs.device, dtype=node_probs.dtype))
        else:
            values.append(graph_node_probs.max())
    return torch.stack(values)


def get_device(prefer_mps: bool = True) -> str:
    """Get the best available device."""
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class FocalLoss(torch.nn.Module):
    """Focal Loss for extreme class imbalance."""
    
    def __init__(self, alpha: float = 0.95, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiLabelFocalLoss(torch.nn.Module):
    """Focal loss for multi-label classification with label smoothing."""
    
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, num_rules] raw logits
            targets: [N, num_rules] float targets (0 or 1)
        """
        # Apply label smoothing: 1 -> 1 - smoothing, 0 -> smoothing
        smoothed_targets = targets * (1 - self.label_smoothing) + (1 - targets) * self.label_smoothing
        
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        bce = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction="none")
        loss = focal_weight * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class Trainer:
    """Trainer for DOM GNN models with multi-label support."""
    
    def __init__(
        self,
        model: DOMAttentionNet,
        device: str = "auto",
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        node_loss_weight: float = 1.0,
        rule_loss_weight: float = 5.0,
        graph_loss_weight: float = 1.0,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.95,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        hard_neg_weight: float = 10.0,
        hard_pos_weight: float = 5.0,
        hard_neg_threshold: float = 0.1,
        hard_pos_threshold: float = 0.7,
        node_pos_weight_cap: float = 50.0,
        node_hard_negative_ratio: float = 0.0,
        rule_pos_weight: float = 50.0,
        node_threshold: float = 0.5,
        rule_threshold: float = 0.5,
        selection_metric: str = "node_f1_pos_plus_graph_recall",
        cache_clear_interval: int = 0,
        clean_page_node_loss_weight: float = 0.5,
        positive_page_node_evidence_loss_weight: float = 0.5,
        graph_node_consistency_loss_weight: float = 0.5,
    ):
        if device == "auto":
            device = get_device()
        
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )
        
        self.node_loss_weight = node_loss_weight
        self.rule_loss_weight = rule_loss_weight
        self.graph_loss_weight = graph_loss_weight
        self.use_focal_loss = use_focal_loss
        self.hard_neg_weight = hard_neg_weight
        self.hard_pos_weight = hard_pos_weight
        self.hard_neg_threshold = hard_neg_threshold
        self.hard_pos_threshold = hard_pos_threshold
        self.node_pos_weight_cap = node_pos_weight_cap
        self.node_hard_negative_ratio = node_hard_negative_ratio
        self.rule_pos_weight = rule_pos_weight
        self.rule_label_smoothing = label_smoothing
        self.node_threshold = node_threshold
        self.rule_threshold = rule_threshold
        self.selection_metric = selection_metric
        self.cache_clear_interval = cache_clear_interval
        self.clean_page_node_loss_weight = clean_page_node_loss_weight
        self.positive_page_node_evidence_loss_weight = positive_page_node_evidence_loss_weight
        self.graph_node_consistency_loss_weight = graph_node_consistency_loss_weight
        
        if use_focal_loss:
            self.node_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.rule_criterion = MultiLabelFocalLoss(
                gamma=focal_gamma, label_smoothing=label_smoothing
            )
        else:
            self.node_criterion = None
            self.rule_criterion = torch.nn.BCEWithLogitsLoss()
        
        self.history = {
            "train_loss": [],
            "train_node_acc": [],
            "train_graph_acc": [],
            "val_node_acc": [],
            "val_graph_acc": [],
            "val_f1": [],
            "val_node_precision": [],
            "val_node_recall": [],
            "val_node_f1": [],
            "val_rule_precision_micro": [],
            "val_rule_recall_micro": [],
            "val_rule_f1_micro": [],
            "val_graph_precision": [],
            "val_graph_recall": [],
        }

    def _available_rule_mask(self, data: Data) -> torch.Tensor:
        mask = getattr(data, "available_rule_mask", None)
        if mask is None:
            return torch.ones(NUM_RULES, dtype=torch.bool, device=self.device)
        if mask.numel() != NUM_RULES and mask.numel() % NUM_RULES == 0:
            mask = mask.reshape(-1, NUM_RULES)
        if mask.dim() > 1:
            mask = mask[0]
        return mask.to(self.device).bool()
    
    def compute_loss_from_outputs(
        self,
        data: Data,
        node_logits: torch.Tensor,
        node_rule_logits: torch.Tensor,
        graph_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss from an existing model forward pass."""
        metrics = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Formula: L_node = -sum_i sum_{c in {0,1}} y_{i,c} log(y_hat_{i,c}).
        # The implementation uses weighted cross-entropy and optional hard
        # negative mining for the binary node violation head.
        # Node-level binary loss with aggressive class weighting
        if hasattr(data, "node_y") and data.node_y is not None:
            num_pos = (data.node_y == 1).sum().item()
            num_neg = (data.node_y == 0).sum().item()
            
            if num_pos > 0:
                # Aggressive class weighting: weight positive class by neg/pos ratio
                weight_pos = min(num_neg / (num_pos + 1e-6), self.node_pos_weight_cap)
                weights = torch.tensor([1.0, weight_pos], device=self.device)
            else:
                weights = None
            
            if self.node_hard_negative_ratio > 0 and num_pos > 0 and num_neg > 0:
                per_node_loss = F.cross_entropy(node_logits, data.node_y, reduction="none")
                pos_mask = data.node_y == 1
                neg_mask = data.node_y == 0
                pos_loss = per_node_loss[pos_mask].mean()
                neg_losses = per_node_loss[neg_mask]
                hard_neg_count = min(
                    neg_losses.numel(),
                    max(1, int(num_pos * self.node_hard_negative_ratio)),
                )
                hard_neg_loss = torch.topk(neg_losses, k=hard_neg_count).values.mean()
                node_loss = 0.5 * (pos_loss + hard_neg_loss)
            else:
                node_loss = F.cross_entropy(node_logits, data.node_y, weight=weights)
            
            node_preds = node_logits.argmax(dim=-1)
            node_acc = (node_preds == data.node_y).float().mean().item()
            num_pos_pred = (node_preds == 1).sum().item()
            num_pos_true = (data.node_y == 1).sum().item()
            
            total_loss += self.node_loss_weight * node_loss
            metrics["node_loss"] = node_loss.item()
            metrics["node_acc"] = node_acc
            metrics["node_pos_pred"] = num_pos_pred
            metrics["node_pos_true"] = num_pos_true
            metrics["node_pos_weight"] = weights[1].item() if weights is not None else 0.0

            clean_page_loss = torch.tensor(0.0, device=self.device)
            positive_page_node_evidence_loss = torch.tensor(0.0, device=self.device)
            if (
                (self.clean_page_node_loss_weight > 0 or self.positive_page_node_evidence_loss_weight > 0)
                and hasattr(data, "y")
                and data.y is not None
                and hasattr(data, "batch")
            ):
                node_probs = F.softmax(node_logits, dim=-1)[:, 1]
                clean_graph_mask = data.y[data.batch] == 0
                if self.clean_page_node_loss_weight > 0 and clean_graph_mask.any():
                    clean_page_loss = node_probs[clean_graph_mask].mean()
                    total_loss += self.clean_page_node_loss_weight * clean_page_loss
                if self.positive_page_node_evidence_loss_weight > 0:
                    node_evidence_probs = max_node_probability_by_graph(
                        node_probs=node_probs,
                        batch=data.batch,
                        num_graphs=data.y.numel(),
                    )
                    positive_graph_mask = data.y == 1
                    if positive_graph_mask.any():
                        positive_page_node_evidence_loss = F.binary_cross_entropy(
                            node_evidence_probs[positive_graph_mask],
                            torch.ones_like(node_evidence_probs[positive_graph_mask]),
                        )
                        total_loss += (
                            self.positive_page_node_evidence_loss_weight
                            * positive_page_node_evidence_loss
                        )
            metrics["clean_page_node_loss"] = clean_page_loss.item()
            metrics["positive_page_node_evidence_loss"] = positive_page_node_evidence_loss.item()
        else:
            metrics["node_loss"] = 0.0
            metrics["node_acc"] = 0.0
            metrics["node_pos_pred"] = 0
            metrics["node_pos_true"] = 0
            metrics["node_pos_weight"] = 0.0
            metrics["clean_page_node_loss"] = 0.0
            metrics["positive_page_node_evidence_loss"] = 0.0
        
        # Formula: L_rule = BCEWithLogits(z_rule, y_rule), averaged over node-rule
        # pairs after masking unavailable rules. pos_weight and hard-example
        # weights compensate for sparse positive WCAG/axe labels.
        # Multi-label rule loss with hard negative mining and class balancing
        if hasattr(data, "node_y_multi") and data.node_y_multi is not None:
            rule_mask = self._available_rule_mask(data)
            rule_targets = data.node_y_multi.to(self.device)[:, rule_mask]
            masked_rule_logits = node_rule_logits[:, rule_mask]
            if rule_targets.numel() == 0:
                metrics["rule_loss"] = 0.0
                metrics["hard_neg"] = 0
                metrics["hard_pos"] = 0
                metrics["pos_weight_mean"] = 0.0
                metrics["loss"] = total_loss.item()
                return total_loss, metrics
            
            # Fixed rule-level positive weight keeps training stable across batches.
            pos_weight_val = self.rule_pos_weight
            
            # Keep absent rules at 0. Smoothing negatives with pos_weight makes the
            # sparse rule head predict nearly every node-rule pair as positive.
            smoothed_targets = rule_targets
            if self.rule_label_smoothing > 0:
                smoothed_targets = rule_targets * (1.0 - self.rule_label_smoothing)
            per_sample_loss = F.binary_cross_entropy_with_logits(
                masked_rule_logits, smoothed_targets,
                pos_weight=torch.tensor([pos_weight_val], device=self.device),
                reduction="none"
            )
            
            # Hard negative mining: upweight misclassified examples
            with torch.no_grad():
                rule_probs = torch.sigmoid(masked_rule_logits)
                hard_neg_mask = (rule_probs > self.hard_neg_threshold) & (rule_targets == 0)
                hard_pos_mask = (rule_probs < self.hard_pos_threshold) & (rule_targets == 1)
            
            weights = torch.ones_like(per_sample_loss)
            weights[hard_neg_mask] = self.hard_neg_weight
            weights[hard_pos_mask] = self.hard_pos_weight
            rule_loss = (per_sample_loss * weights).mean()
            
            total_loss += self.rule_loss_weight * rule_loss
            metrics["rule_loss"] = rule_loss.item()
            metrics["hard_neg"] = hard_neg_mask.sum().item()
            metrics["hard_pos"] = hard_pos_mask.sum().item()
            metrics["pos_weight_mean"] = pos_weight_val
        else:
            metrics["rule_loss"] = 0.0
            metrics["hard_neg"] = 0
            metrics["hard_pos"] = 0
            metrics["pos_weight_mean"] = 0.0
        
        # Formula: L_graph = -sum_c y_c^graph log(y_hat_c^graph).
        # Graph-level loss
        if hasattr(data, "y") and data.y is not None:
            graph_loss = F.cross_entropy(graph_logits, data.y)
            graph_preds = graph_logits.argmax(dim=-1)
            graph_acc = (graph_preds == data.y).float().mean().item()
            
            total_loss += self.graph_loss_weight * graph_loss
            metrics["graph_loss"] = graph_loss.item()
            metrics["graph_acc"] = graph_acc

            graph_node_consistency_loss = torch.tensor(0.0, device=self.device)
            if (
                self.graph_node_consistency_loss_weight > 0
                and hasattr(data, "node_y")
                and data.node_y is not None
            ):
                node_probs = F.softmax(node_logits, dim=-1)[:, 1]
                node_evidence_probs = max_node_probability_by_graph(
                    node_probs=node_probs,
                    batch=data.batch if hasattr(data, "batch") else None,
                    num_graphs=data.y.numel(),
                )
                graph_probs = F.softmax(graph_logits, dim=-1)[:, 1]
                # Formula: p_node^max = max_i P(y_i^node = 1),
                # p_graph = P(y^graph = 1), and
                # L_cons = (p_graph - p_node^max)^2.
                graph_node_consistency_loss = F.mse_loss(graph_probs, node_evidence_probs)
                total_loss += self.graph_node_consistency_loss_weight * graph_node_consistency_loss
            metrics["graph_node_consistency_loss"] = graph_node_consistency_loss.item()
        else:
            metrics["graph_loss"] = 0.0
            metrics["graph_acc"] = 0.0
            metrics["graph_node_consistency_loss"] = 0.0
        
        # Formula: L = lambda_node L_node + lambda_rule L_rule
        # + lambda_graph L_graph + lambda_cons L_cons.
        metrics["loss"] = total_loss.item()
        
        return total_loss, metrics

    def compute_loss(self, data: Data) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss with hard negative mining."""
        data = data.to(self.device)
        
        node_logits, node_rule_logits, graph_logits = self.model(
            data.x, data.edge_index, data.tag_indices,
            data.batch if hasattr(data, "batch") else None
        )
        
        return self.compute_loss_from_outputs(
            data=data,
            node_logits=node_logits,
            node_rule_logits=node_rule_logits,
            graph_logits=graph_logits,
        )
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            "loss": 0.0,
            "node_loss": 0.0,
            "rule_loss": 0.0,
            "graph_loss": 0.0,
            "node_acc": 0.0,
            "graph_acc": 0.0,
            "hard_neg": 0,
            "hard_pos": 0,
            "pos_weight_mean": 0.0,
            "node_pos_pred": 0,
            "node_pos_true": 0,
            "node_pos_weight": 0.0,
            "clean_page_node_loss": 0.0,
            "positive_page_node_evidence_loss": 0.0,
            "graph_node_consistency_loss": 0.0,
        }
        
        for batch_idx, data in enumerate(loader):
            self.optimizer.zero_grad(set_to_none=True) # Optimize memory for semaphore error
            loss, metrics = self.compute_loss(data)
            loss.backward()
            
            # Gradient clipping (more conservative for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics.get(key, 0)
            
            del loss, metrics, data
            if self.cache_clear_interval and (batch_idx + 1) % self.cache_clear_interval == 0:
                self.clear_device_cache()
        
        # Average metrics
        num_batches = len(loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        self.model.eval()
        
        all_graph_preds = []
        all_graph_labels = []
        all_graph_probs = []
        node_tp = node_fp = node_fn = node_tn = 0
        node_total = 0
        rule_tp = torch.zeros(NUM_RULES, dtype=torch.long)
        rule_fp = torch.zeros(NUM_RULES, dtype=torch.long)
        rule_fn = torch.zeros(NUM_RULES, dtype=torch.long)
        rule_label_count = torch.zeros(NUM_RULES, dtype=torch.long)
        rule_available = torch.zeros(NUM_RULES, dtype=torch.bool)
        has_node_labels = False
        has_rule_labels = False
        
        total_loss = 0.0
        
        for batch_idx, data in enumerate(loader):
            data = data.to(self.device)
            node_logits, node_rule_logits, graph_logits = self.model(
                data.x, data.edge_index, data.tag_indices,
                data.batch if hasattr(data, "batch") else None
            )
            
            # Collect binary node predictions
            if hasattr(data, "node_y"):
                has_node_labels = True
                node_probs = F.softmax(node_logits, dim=-1)[:, 1]
                node_preds = node_probs >= self.node_threshold
                node_labels = data.node_y.bool()
                node_tp += (node_preds & node_labels).sum().item()
                node_fp += (node_preds & ~node_labels).sum().item()
                node_fn += (~node_preds & node_labels).sum().item()
                node_tn += (~node_preds & ~node_labels).sum().item()
                node_total += node_labels.numel()
            
            # Collect multi-label rule predictions
            if hasattr(data, "node_y_multi"):
                has_rule_labels = True
                batch_rule_mask = self._available_rule_mask(data).cpu()
                rule_available |= batch_rule_mask
                rule_preds = (torch.sigmoid(node_rule_logits) >= self.rule_threshold).cpu()
                rule_labels = data.node_y_multi.cpu().bool()
                rule_preds[:, ~batch_rule_mask] = False
                rule_labels[:, ~batch_rule_mask] = False
                rule_tp += (rule_preds & rule_labels).sum(dim=0)
                rule_fp += (rule_preds & ~rule_labels).sum(dim=0)
                rule_fn += (~rule_preds & rule_labels).sum(dim=0)
                rule_label_count += rule_labels.sum(dim=0)
            
            # Collect graph predictions
            if hasattr(data, "y"):
                graph_preds = graph_logits.argmax(dim=-1).cpu()
                graph_probs = F.softmax(graph_logits, dim=-1)[:, 1].cpu()
                all_graph_preds.append(graph_preds)
                all_graph_labels.append(data.y.cpu())
                all_graph_probs.append(graph_probs)
            
            loss, _ = self.compute_loss_from_outputs(
                data=data,
                node_logits=node_logits,
                node_rule_logits=node_rule_logits,
                graph_logits=graph_logits,
            )
            total_loss += loss.item()
            
            del data, node_logits, node_rule_logits, graph_logits, loss
            if self.cache_clear_interval and (batch_idx + 1) % self.cache_clear_interval == 0:
                self.clear_device_cache()
        
        metrics = {"loss": total_loss / len(loader)}
        
        # Binary node-level metrics
        if has_node_labels:
            node_precision = node_tp / (node_tp + node_fp) if (node_tp + node_fp) else 0.0
            node_recall = node_tp / (node_tp + node_fn) if (node_tp + node_fn) else 0.0
            node_f1_pos = (
                2 * node_precision * node_recall / (node_precision + node_recall)
                if (node_precision + node_recall) else 0.0
            )
            node_neg_precision = node_tn / (node_tn + node_fn) if (node_tn + node_fn) else 0.0
            node_neg_recall = node_tn / (node_tn + node_fp) if (node_tn + node_fp) else 0.0
            node_f1_neg = (
                2 * node_neg_precision * node_neg_recall / (node_neg_precision + node_neg_recall)
                if (node_neg_precision + node_neg_recall) else 0.0
            )
            metrics["node_acc"] = (node_tp + node_tn) / node_total if node_total else 0.0
            metrics["node_f1_macro"] = (node_f1_pos + node_f1_neg) / 2
            metrics["node_f1_pos"] = node_f1_pos
            metrics["node_precision"] = node_precision
            metrics["node_recall"] = node_recall
        
        # Multi-label rule metrics
        if has_rule_labels:
            micro_tp = rule_tp[rule_available].sum().item()
            micro_fp = rule_fp[rule_available].sum().item()
            micro_fn = rule_fn[rule_available].sum().item()
            rule_precision_micro = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
            rule_recall_micro = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
            rule_f1_micro = (
                2 * rule_precision_micro * rule_recall_micro / (rule_precision_micro + rule_recall_micro)
                if (rule_precision_micro + rule_recall_micro) else 0.0
            )
            per_rule_precision = rule_tp.float() / (rule_tp + rule_fp).clamp_min(1).float()
            per_rule_recall = rule_tp.float() / (rule_tp + rule_fn).clamp_min(1).float()
            per_rule_denominator = per_rule_precision + per_rule_recall
            per_rule_f1_values = torch.where(
                per_rule_denominator > 0,
                2 * per_rule_precision * per_rule_recall / per_rule_denominator,
                torch.zeros_like(per_rule_denominator),
            )
            
            metrics["rule_precision_micro"] = rule_precision_micro
            metrics["rule_recall_micro"] = rule_recall_micro
            metrics["rule_f1_micro"] = rule_f1_micro
            metrics["rule_f1_macro"] = (
                per_rule_f1_values[rule_available].mean().item()
                if rule_available.any()
                else 0.0
            )

            def family_f1(name: str, owners: Tuple[str, ...]) -> None:
                indices = torch.tensor(rule_indices_for_owners(owners), dtype=torch.long)
                if indices.numel() == 0:
                    metrics[name] = 0.0
                    return
                family_available = rule_available[indices]
                if not family_available.any():
                    metrics[name] = 0.0
                    return
                family_indices = indices[family_available]
                tp = rule_tp[family_indices].sum().item()
                fp = rule_fp[family_indices].sum().item()
                fn = rule_fn[family_indices].sum().item()
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                metrics[name] = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )

            family_f1("semantic_rule_f1", (RULE_OWNER_A11Y_TREE,))
            family_f1("structure_rule_f1", (RULE_OWNER_DOM, RULE_OWNER_DOM_PAGE))
            family_f1("visual_rule_f1", (RULE_OWNER_RENDERED_VISUAL,))
            
            # Per-rule metrics
            per_rule_f1 = {}
            for i in range(NUM_RULES):
                if rule_available[i] and rule_label_count[i] > 0:
                    per_rule_f1[INDEX_TO_RULE[i]] = round(per_rule_f1_values[i].item(), 4)
            
            # Top 5 best and worst performing rules
            sorted_rules = sorted(per_rule_f1.items(), key=lambda x: x[1], reverse=True)
            metrics["top_rules"] = sorted_rules[:5]
            metrics["worst_rules"] = sorted_rules[-5:] if len(sorted_rules) >= 5 else sorted_rules
        
        # Graph-level metrics
        if all_graph_preds:
            graph_preds = torch.cat(all_graph_preds)
            graph_labels = torch.cat(all_graph_labels)
            graph_probs = torch.cat(all_graph_probs)
            
            metrics["graph_acc"] = accuracy_score(graph_labels, graph_preds)
            metrics["graph_f1"] = f1_score(graph_labels, graph_preds, average="weighted", zero_division=0)
            metrics["graph_precision"] = precision_score(graph_labels, graph_preds, zero_division=0)
            metrics["graph_recall"] = recall_score(graph_labels, graph_preds, zero_division=0)
            metrics["node_f1_pos_plus_graph_recall"] = (
                0.8 * metrics.get("node_f1_pos", 0.0)
                + 0.2 * metrics["graph_recall"]
            )
            graph_source = getattr(loader.dataset[0], "graph_source", None) if len(loader.dataset) else None
            if graph_source == "a11y-tree":
                metrics["a11y_node_f1"] = metrics.get("node_f1_pos", 0.0)
            elif graph_source == "rendered-visual":
                metrics["rendered_node_f1"] = metrics.get("node_f1_pos", 0.0)
            else:
                metrics["dom_node_f1"] = metrics.get("node_f1_pos", 0.0)
            
            if len(torch.unique(graph_labels)) > 1:
                metrics["graph_auc"] = roc_auc_score(graph_labels, graph_probs)
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        patience: int = 10,
        save_path: Optional[Path] = None,
        last_save_path: Optional[Path] = None,
        hparams: Optional[Dict] = None,
        resume_from: Optional[Path] = None,
    ) -> Dict:
        """Full training loop with early stopping."""
        best_metric_value = -1.0
        patience_counter = 0
        has_saved = False
        start_epoch = 0

        if resume_from is not None and resume_from.exists():
            checkpoint = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_metric_value = checkpoint.get(
                "selection_metric_value", checkpoint.get("val_f1", best_metric_value)
            )
            patience_counter = int(checkpoint.get("patience_counter", 0))
            if (
                "patience_counter" not in checkpoint
                and save_path is not None
                and save_path.exists()
            ):
                best_checkpoint = torch.load(
                    save_path,
                    map_location="cpu",
                    weights_only=False,
                )
                best_epoch = int(best_checkpoint.get("epoch", start_epoch - 1))
                patience_counter = max(0, start_epoch - best_epoch - 1)
            checkpoint_metric = checkpoint.get("selection_metric", self.selection_metric)
            if checkpoint_metric != self.selection_metric:
                print(
                    f"Warning: checkpoint was selected by {checkpoint_metric}, "
                    f"but this run is selecting by {self.selection_metric}"
                )
            has_saved = True
            print(
                f"Resuming training from {resume_from} at epoch {start_epoch + 1} "
                f"(best {self.selection_metric}={best_metric_value:.4f}, "
                f"early-stop counter={patience_counter}/{patience})"
            )
        
        for epoch in range(start_epoch, epochs):
            train_metrics = self.train_epoch(train_loader)
            
            log_str = f"Epoch {epoch+1}/{epochs}"
            log_str += f" | Loss: {train_metrics['loss']:.4f}"
            log_str += f" | NodeAcc: {train_metrics['node_acc']:.4f}"
            log_str += f" | PosPred: {train_metrics['node_pos_pred']:.1f}/{train_metrics['node_pos_true']:.1f}"
            log_str += f" | GraphAcc: {train_metrics['graph_acc']:.4f}"
            log_str += f" | HardNeg: {train_metrics['hard_neg']:.0f}"
            log_str += f" | HardPos: {train_metrics['hard_pos']:.0f}"
            log_str += f" | PosWeight: {train_metrics['pos_weight_mean']:.1f}"
            log_str += f" | NodeW: {train_metrics['node_pos_weight']:.1f}"
            if self.clean_page_node_loss_weight > 0:
                log_str += f" | CleanNodeLoss: {train_metrics['clean_page_node_loss']:.4f}"
            if self.positive_page_node_evidence_loss_weight > 0:
                log_str += f" | PosPageNodeLoss: {train_metrics['positive_page_node_evidence_loss']:.4f}"
            if self.graph_node_consistency_loss_weight > 0:
                log_str += f" | GraphNodeLoss: {train_metrics['graph_node_consistency_loss']:.4f}"
            
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_node_acc"].append(train_metrics["node_acc"])
            self.history["train_graph_acc"].append(train_metrics["graph_acc"])
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                log_str += f" || Val Loss: {val_metrics['loss']:.4f}"
                log_str += f" | Val Node P/R/F1: {val_metrics.get('node_precision', 0):.4f}/{val_metrics.get('node_recall', 0):.4f}/{val_metrics.get('node_f1_pos', 0):.4f}"
                log_str += (
                    " | Val Graph P/R/F1: "
                    f"{val_metrics.get('graph_precision', 0):.4f}/"
                    f"{val_metrics.get('graph_recall', 0):.4f}/"
                    f"{val_metrics.get('graph_f1', 0):.4f}"
                )
                
                if "rule_f1_micro" in val_metrics:
                    log_str += (
                        " | Rule P/R/F1(micro): "
                        f"{val_metrics.get('rule_precision_micro', 0):.4f}/"
                        f"{val_metrics.get('rule_recall_micro', 0):.4f}/"
                        f"{val_metrics['rule_f1_micro']:.4f}"
                    )
                
                self.history["val_node_acc"].append(val_metrics.get("node_acc", 0))
                self.history["val_graph_acc"].append(val_metrics.get("graph_acc", 0))
                self.history["val_node_precision"].append(val_metrics.get("node_precision", 0))
                self.history["val_node_recall"].append(val_metrics.get("node_recall", 0))
                self.history["val_node_f1"].append(val_metrics.get("node_f1_pos", 0))
                self.history["val_graph_precision"].append(val_metrics.get("graph_precision", 0))
                self.history["val_graph_recall"].append(val_metrics.get("graph_recall", 0))
                self.history["val_rule_precision_micro"].append(val_metrics.get("rule_precision_micro", 0))
                self.history["val_rule_recall_micro"].append(val_metrics.get("rule_recall_micro", 0))
                self.history["val_rule_f1_micro"].append(val_metrics.get("rule_f1_micro", 0))
                
                if self.selection_metric not in val_metrics:
                    available = ", ".join(sorted(val_metrics.keys()))
                    raise ValueError(
                        f"Selection metric '{self.selection_metric}' was not produced. "
                        f"Available validation metrics: {available}"
                    )
                current_metric = val_metrics[self.selection_metric]
                self.history["val_f1"].append(current_metric)
                
                self.scheduler.step(current_metric)
                
                if current_metric > best_metric_value:
                    best_metric_value = current_metric
                    patience_counter = 0
                    if save_path:
                        checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_f1": best_metric_value,
                            "selection_metric": self.selection_metric,
                            "selection_metric_value": best_metric_value,
                            "patience_counter": patience_counter,
                            "scheduler_state_dict": self.scheduler.state_dict(),
                        }
                        if hparams:
                            checkpoint["hparams"] = hparams
                        torch.save(checkpoint, save_path)
                        has_saved = True
                        print(
                            f"  -> Saved best model "
                            f"({self.selection_metric}={best_metric_value:.4f})"
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break

                if last_save_path:
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_f1": best_metric_value,
                        "selection_metric": self.selection_metric,
                        "selection_metric_value": best_metric_value,
                        "last_metric_value": current_metric,
                        "patience_counter": patience_counter,
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    }
                    if hparams:
                        checkpoint["hparams"] = hparams
                    torch.save(checkpoint, last_save_path)
            
            self.clear_device_cache()
            print(log_str)
        
        # Save final model if none was saved
        if save_path and not has_saved:
            checkpoint = {
                "epoch": epochs - 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_f1": best_metric_value,
                "selection_metric": self.selection_metric,
                "selection_metric_value": best_metric_value,
                "patience_counter": patience_counter,
                "scheduler_state_dict": self.scheduler.state_dict(),
            }
            if hparams:
                checkpoint["hparams"] = hparams
            torch.save(checkpoint, save_path)
            print(f"  -> Saved final model ({self.selection_metric}={best_metric_value:.4f})")
        
        return self.history
    
    # Fix for `leaked semaphore objects`
    def clear_device_cache(self):
        """Release cached accelerator memory where supported."""
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
    
    def load_best(self, save_path: Path):
        """Load best model checkpoint."""
        checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        metric_name = checkpoint.get("selection_metric", "val_f1")
        metric_value = checkpoint.get("selection_metric_value", checkpoint.get("val_f1", 0.0))
        print(f"Loaded best model from epoch {checkpoint['epoch']} ({metric_name}={metric_value:.4f})")


def create_data_loaders(
    data_list: List[Data],
    batch_size: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split data into train/val/test and create loaders."""
    n = len(data_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_data = data_list[:n_train]
    val_data = data_list[n_train:n_train + n_val]
    test_data = data_list[n_train + n_val:]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
