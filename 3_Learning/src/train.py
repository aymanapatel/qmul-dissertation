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
from wcag_rules import NUM_RULES, INDEX_TO_RULE


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
        graph_loss_weight: float = 0.5,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.95,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        hard_neg_weight: float = 10.0,
        hard_pos_weight: float = 5.0,
        hard_neg_threshold: float = 0.1,
        hard_pos_threshold: float = 0.7,
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
        }
    
    def compute_loss(self, data: Data) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss with hard negative mining."""
        data = data.to(self.device)
        
        node_logits, node_rule_logits, graph_logits = self.model(
            data.x, data.edge_index, data.tag_indices,
            data.batch if hasattr(data, "batch") else None
        )
        
        metrics = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Node-level binary loss with aggressive class weighting
        if hasattr(data, "node_y") and data.node_y is not None:
            num_pos = (data.node_y == 1).sum().item()
            num_neg = (data.node_y == 0).sum().item()
            
            if num_pos > 0:
                # Aggressive class weighting: weight positive class by neg/pos ratio
                weight_pos = min(num_neg / (num_pos + 1e-6), 500.0)
                weights = torch.tensor([1.0, weight_pos], device=self.device)
            else:
                weights = None
            
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
        else:
            metrics["node_loss"] = 0.0
            metrics["node_acc"] = 0.0
            metrics["node_pos_pred"] = 0
            metrics["node_pos_true"] = 0
            metrics["node_pos_weight"] = 0.0
        
        # Multi-label rule loss with hard negative mining and class balancing
        if hasattr(data, "node_y_multi") and data.node_y_multi is not None:
            rule_targets = data.node_y_multi.to(self.device)
            
            # Use fixed pos_weight=200 based on ~0.5% violation rate
            # This is more stable than per-batch adaptive weighting
            pos_weight_val = 200.0
            
            # Base BCE loss with fixed class weighting and label smoothing
            smoothed_targets = rule_targets * 0.9 + (1 - rule_targets) * 0.1
            per_sample_loss = F.binary_cross_entropy_with_logits(
                node_rule_logits, smoothed_targets,
                pos_weight=torch.tensor([pos_weight_val], device=self.device),
                reduction="none"
            )
            
            # Hard negative mining: upweight misclassified examples
            with torch.no_grad():
                rule_probs = torch.sigmoid(node_rule_logits)
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
        
        # Graph-level loss
        if hasattr(data, "y") and data.y is not None:
            graph_loss = F.cross_entropy(graph_logits, data.y)
            graph_preds = graph_logits.argmax(dim=-1)
            graph_acc = (graph_preds == data.y).float().mean().item()
            
            total_loss += self.graph_loss_weight * graph_loss
            metrics["graph_loss"] = graph_loss.item()
            metrics["graph_acc"] = graph_acc
        else:
            metrics["graph_loss"] = 0.0
            metrics["graph_acc"] = 0.0
        
        metrics["loss"] = total_loss.item()
        
        return total_loss, metrics
    
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
        }
        
        for batch_idx, data in enumerate(loader):
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(data)
            loss.backward()
            
            # Gradient clipping (more conservative for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics.get(key, 0)
        
        # Average metrics
        num_batches = len(loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        self.model.eval()
        
        all_node_preds = []
        all_node_labels = []
        all_rule_probs = []
        all_rule_labels = []
        all_graph_preds = []
        all_graph_labels = []
        all_graph_probs = []
        
        total_loss = 0.0
        
        for data in loader:
            data = data.to(self.device)
            node_logits, node_rule_logits, graph_logits = self.model(
                data.x, data.edge_index, data.tag_indices,
                data.batch if hasattr(data, "batch") else None
            )
            
            # Collect binary node predictions
            if hasattr(data, "node_y"):
                node_preds = node_logits.argmax(dim=-1).cpu()
                all_node_preds.append(node_preds)
                all_node_labels.append(data.node_y.cpu())
            
            # Collect multi-label rule predictions
            if hasattr(data, "node_y_multi"):
                rule_probs = torch.sigmoid(node_rule_logits).cpu()
                all_rule_probs.append(rule_probs)
                all_rule_labels.append(data.node_y_multi.cpu())
            
            # Collect graph predictions
            if hasattr(data, "y"):
                graph_preds = graph_logits.argmax(dim=-1).cpu()
                graph_probs = F.softmax(graph_logits, dim=-1)[:, 1].cpu()
                all_graph_preds.append(graph_preds)
                all_graph_labels.append(data.y.cpu())
                all_graph_probs.append(graph_probs)
            
            loss, _ = self.compute_loss(data)
            total_loss += loss.item()
        
        metrics = {"loss": total_loss / len(loader)}
        
        # Binary node-level metrics
        if all_node_preds:
            node_preds = torch.cat(all_node_preds)
            node_labels = torch.cat(all_node_labels)
            metrics["node_acc"] = accuracy_score(node_labels, node_preds)
            metrics["node_f1_macro"] = f1_score(node_labels, node_preds, average="macro", zero_division=0)
            metrics["node_f1_pos"] = f1_score(node_labels, node_preds, pos_label=1, zero_division=0)
            metrics["node_precision"] = precision_score(node_labels, node_preds, pos_label=1, zero_division=0)
            metrics["node_recall"] = recall_score(node_labels, node_preds, pos_label=1, zero_division=0)
        
        # Multi-label rule metrics
        if all_rule_probs:
            rule_probs = torch.cat(all_rule_probs)
            rule_labels = torch.cat(all_rule_labels)
            rule_preds = (rule_probs > 0.5).long()
            
            # Overall metrics
            metrics["rule_f1_micro"] = f1_score(
                rule_labels.numpy(), rule_preds.numpy(), average="micro", zero_division=0
            )
            metrics["rule_f1_macro"] = f1_score(
                rule_labels.numpy(), rule_preds.numpy(), average="macro", zero_division=0
            )
            
            # Per-rule metrics
            per_rule_f1 = {}
            for i in range(NUM_RULES):
                rule_f1 = f1_score(
                    rule_labels[:, i].numpy(), rule_preds[:, i].numpy(), zero_division=0
                )
                if rule_labels[:, i].sum() > 0:  # Only report rules that appear in data
                    per_rule_f1[INDEX_TO_RULE[i]] = round(rule_f1, 4)
            
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
        hparams: Optional[Dict] = None,
    ) -> Dict:
        """Full training loop with early stopping."""
        best_val_f1 = -1.0
        patience_counter = 0
        has_saved = False
        
        for epoch in range(epochs):
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
            
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_node_acc"].append(train_metrics["node_acc"])
            self.history["train_graph_acc"].append(train_metrics["graph_acc"])
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                log_str += f" || Val Loss: {val_metrics['loss']:.4f}"
                log_str += f" | Val Node F1: {val_metrics.get('node_f1_pos', 0):.4f}"
                log_str += f" | Val Graph F1: {val_metrics.get('graph_f1', 0):.4f}"
                
                if "rule_f1_micro" in val_metrics:
                    log_str += f" | Rule F1(micro): {val_metrics['rule_f1_micro']:.4f}"
                
                self.history["val_node_acc"].append(val_metrics.get("node_acc", 0))
                self.history["val_graph_acc"].append(val_metrics.get("graph_acc", 0))
                self.history["val_f1"].append(val_metrics.get("graph_f1", 0))
                
                # Learning rate scheduling on rule F1 micro (best overall metric)
                current_f1 = val_metrics.get("rule_f1_micro", val_metrics.get("node_f1_pos", 0))
                self.scheduler.step(current_f1)
                
                # Early stopping based on rule F1
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    patience_counter = 0
                    if save_path:
                        checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_f1": best_val_f1,
                        }
                        if hparams:
                            checkpoint["hparams"] = hparams
                        torch.save(checkpoint, save_path)
                        has_saved = True
                        print(f"  -> Saved best model (F1={best_val_f1:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            
            print(log_str)
        
        # Save final model if none was saved
        if save_path and not has_saved:
            checkpoint = {
                "epoch": epochs - 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_f1": best_val_f1,
            }
            if hparams:
                checkpoint["hparams"] = hparams
            torch.save(checkpoint, save_path)
            print(f"  -> Saved final model (F1={best_val_f1:.4f})")
        
        return self.history
    
    def load_best(self, save_path: Path):
        """Load best model checkpoint."""
        checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']} (F1={checkpoint['val_f1']:.4f})")


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
