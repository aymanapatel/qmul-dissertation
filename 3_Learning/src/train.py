"""
train.py

Training and evaluation pipeline for DOM graph classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models import DOMGCN


class FocalLoss(torch.nn.Module):
    """Focal Loss for extreme class imbalance.
    
    FL(pt) = -α(1-pt)^γ log(pt)
    
    Args:
        alpha: Weighting factor for rare class (positive class)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, alpha: float = 0.95, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        
        # Get predicted probabilities
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting (emphasize positive class)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class Trainer:
    """Trainer for DOM GNN models."""
    
    def __init__(
        self,
        model: DOMGCN,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        node_loss_weight: float = 1.0,
        graph_loss_weight: float = 1.0,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.95,
        focal_gamma: float = 2.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )
        self.node_loss_weight = node_loss_weight
        self.graph_loss_weight = graph_loss_weight
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            self.node_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.node_criterion = None
        
        self.history = {
            "train_loss": [],
            "train_node_acc": [],
            "train_graph_acc": [],
            "val_node_acc": [],
            "val_graph_acc": [],
            "val_f1": [],
        }
    
    def compute_loss(self, data: Data) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined node-level + graph-level loss with class weighting."""
        data = data.to(self.device)
        
        node_logits, graph_logits = self.model(
            data.x, data.edge_index, data.tag_indices, data.batch if hasattr(data, "batch") else None
        )
        
        # Node-level loss with focal loss for extreme imbalance
        if hasattr(data, "node_y") and data.node_y is not None:
            if self.use_focal_loss and self.node_criterion is not None:
                node_loss = self.node_criterion(node_logits, data.node_y)
            else:
                # Fallback to weighted cross-entropy
                num_pos = (data.node_y == 1).sum().item()
                num_neg = (data.node_y == 0).sum().item()
                total = num_pos + num_neg
                
                if num_pos > 0:
                    weight_pos = total / (2.0 * num_pos)
                    weight_neg = total / (2.0 * num_neg)
                    weights = torch.tensor([weight_neg, weight_pos], device=self.device)
                else:
                    weights = None
                
                node_loss = F.cross_entropy(node_logits, data.node_y, weight=weights)
            
            node_preds = node_logits.argmax(dim=-1)
            node_acc = (node_preds == data.node_y).float().mean().item()
        else:
            node_loss = torch.tensor(0.0, device=self.device)
            node_acc = 0.0
        
        # Graph-level loss
        if hasattr(data, "y") and data.y is not None:
            graph_loss = F.cross_entropy(graph_logits, data.y)
            graph_preds = graph_logits.argmax(dim=-1)
            graph_acc = (graph_preds == data.y).float().mean().item()
        else:
            graph_loss = torch.tensor(0.0, device=self.device)
            graph_acc = 0.0
        
        total_loss = (
            self.node_loss_weight * node_loss +
            self.graph_loss_weight * graph_loss
        )
        
        metrics = {
            "loss": total_loss.item(),
            "node_loss": node_loss.item(),
            "graph_loss": graph_loss.item(),
            "node_acc": node_acc,
            "graph_acc": graph_acc,
        }
        
        return total_loss, metrics
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            "loss": 0.0,
            "node_loss": 0.0,
            "graph_loss": 0.0,
            "node_acc": 0.0,
            "graph_acc": 0.0,
        }
        
        for batch_idx, data in enumerate(loader):
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(data)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
        
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
        all_graph_preds = []
        all_graph_labels = []
        all_graph_probs = []
        
        total_loss = 0.0
        
        for data in loader:
            data = data.to(self.device)
            node_logits, graph_logits = self.model(
                data.x, data.edge_index, data.tag_indices,
                data.batch if hasattr(data, "batch") else None
            )
            
            # Collect predictions
            if hasattr(data, "node_y"):
                node_preds = node_logits.argmax(dim=-1).cpu()
                all_node_preds.append(node_preds)
                all_node_labels.append(data.node_y.cpu())
            
            if hasattr(data, "y"):
                graph_preds = graph_logits.argmax(dim=-1).cpu()
                graph_probs = F.softmax(graph_logits, dim=-1)[:, 1].cpu()
                all_graph_preds.append(graph_preds)
                all_graph_labels.append(data.y.cpu())
                all_graph_probs.append(graph_probs)
            
            loss, _ = self.compute_loss(data)
            total_loss += loss.item()
        
        metrics = {"loss": total_loss / len(loader)}
        
        # Node-level metrics
        if all_node_preds:
            node_preds = torch.cat(all_node_preds)
            node_labels = torch.cat(all_node_labels)
            metrics["node_acc"] = accuracy_score(node_labels, node_preds)
            metrics["node_f1_macro"] = f1_score(node_labels, node_preds, average="macro", zero_division=0)
            metrics["node_f1_pos"] = f1_score(node_labels, node_preds, pos_label=1, zero_division=0)
            metrics["node_precision"] = precision_score(node_labels, node_preds, pos_label=1, zero_division=0)
            metrics["node_recall"] = recall_score(node_labels, node_preds, pos_label=1, zero_division=0)
        
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
    ) -> Dict:
        """Full training loop with early stopping."""
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            
            log_str = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_metrics['loss']:.4f}"
            log_str += f" | Node Acc: {train_metrics['node_acc']:.4f}"
            log_str += f" | Graph Acc: {train_metrics['graph_acc']:.4f}"
            
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_node_acc"].append(train_metrics["node_acc"])
            self.history["train_graph_acc"].append(train_metrics["graph_acc"])
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                log_str += f" || Val Loss: {val_metrics['loss']:.4f}"
                log_str += f" | Val Node Acc: {val_metrics.get('node_acc', 0):.4f}"
                log_str += f" | Val Graph Acc: {val_metrics.get('graph_acc', 0):.4f}"
                log_str += f" | Val Graph F1: {val_metrics.get('graph_f1', 0):.4f}"
                log_str += f" | Val Node F1(pos): {val_metrics.get('node_f1_pos', 0):.4f}"
                
                self.history["val_node_acc"].append(val_metrics.get("node_acc", 0))
                self.history["val_graph_acc"].append(val_metrics.get("graph_acc", 0))
                self.history["val_f1"].append(val_metrics.get("graph_f1", 0))
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics.get("node_f1_pos", 0))
                
                # Early stopping based on node F1 for positive class
                current_f1 = val_metrics.get("node_f1_pos", 0)
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    patience_counter = 0
                    if save_path:
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_f1": best_val_f1,
                        }, save_path)
                        print(f"  -> Saved best model (node F1={best_val_f1:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            
            print(log_str)
        
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
