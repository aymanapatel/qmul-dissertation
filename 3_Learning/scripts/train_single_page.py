#!/usr/bin/env python3
"""
train_single_page.py

Train a DOM-GNN model on a single page with data augmentation.
Since we only have one page, we create augmented variants through:
- Random edge dropout
- Node feature masking
- Subgraph sampling
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge, subgraph

from feature_extractor import ProcessedPage
from models import DOMGCN
from train import Trainer


def augment_graph(data: Data, aug_type: str = "dropedge", p: float = 0.1) -> Data:
    """Create augmented version of a graph."""
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    tag_indices = data.tag_indices.clone()
    node_y = data.node_y.clone() if hasattr(data, "node_y") and data.node_y is not None else None
    y = data.y.clone() if hasattr(data, "y") and data.y is not None else None
    
    subset = None
    if aug_type == "dropedge":
        # Randomly drop edges
        edge_index, _ = dropout_edge(edge_index, p=p, force_undirected=False)
    elif aug_type == "masknodes":
        # Randomly mask node features
        mask = torch.rand(x.shape[0]) > p
        x = x * mask.unsqueeze(1).float()
    elif aug_type == "subgraph":
        # Sample a connected subgraph
        num_nodes = x.shape[0]
        sample_size = int(num_nodes * (1 - p))
        perm = torch.randperm(num_nodes)
        subset = perm[:sample_size]
        edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        x = x[subset]
        tag_indices = tag_indices[subset]
        if node_y is not None:
            node_y = node_y[subset]
    
    aug_data = Data(
        x=x,
        edge_index=edge_index,
        tag_indices=tag_indices,
        node_y=node_y,
        y=y,
        num_nodes=x.shape[0],
    )
    # Preserve text_embeddings for dimension inference
    if hasattr(data, "text_embeddings") and data.text_embeddings is not None:
        if aug_type == "subgraph":
            aug_data.text_embeddings = data.text_embeddings[subset]
        else:
            aug_data.text_embeddings = data.text_embeddings.clone()
    return aug_data


def create_augmented_dataset(base_data: Data, num_augments: int = 20) -> list:
    """Create dataset with augmented copies."""
    dataset = [base_data.clone()]
    
    aug_types = ["dropedge", "masknodes", "subgraph"]
    
    for i in range(num_augments):
        aug_type = aug_types[i % len(aug_types)]
        p = 0.05 + (i % 5) * 0.03  # Vary augmentation strength
        aug_data = augment_graph(base_data, aug_type=aug_type, p=p)
        dataset.append(aug_data)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train DOM-GNN on single page")
    parser.add_argument("--graph", type=str, required=True, help="Path to processed graph .pt file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--save", type=str, default="./models", help="Model save directory")
    args = parser.parse_args()
    
    graph_path = Path(args.graph)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("DOM-GNN Training")
    print("=" * 60)
    
    # Load processed page
    page = ProcessedPage.load(graph_path)
    base_data = page.data
    
    print(f"Base graph: {base_data.num_nodes} nodes, {base_data.edge_index.shape[1]} edges")
    print(f"Violating nodes: {base_data.node_y.sum().item()}")
    
    # Create augmented dataset
    print("Creating augmented dataset...")
    dataset = create_augmented_dataset(base_data, num_augments=30)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    val_data = dataset[train_size:]
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    
    print(f"Train: {len(train_data)} graphs, Val: {len(val_data)} graphs")
    
    # Initialize model
    # Compute dimensions dynamically from stored text embeddings
    text_dim = (
        base_data.text_embeddings.shape[1]
        if hasattr(base_data, "text_embeddings") and base_data.text_embeddings is not None
        else 384
    )
    attr_dim = base_data.x.shape[1] - text_dim
    model = DOMGCN(
        num_tags=116,
        tag_embed_dim=32,
        attr_dim=attr_dim,
        text_dim=text_dim,
        hidden_dim=args.hidden,
        num_node_classes=2,
        num_graph_classes=2,
        num_layers=args.layers,
        dropout=args.dropout,
        pooling="mean",
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(
        model=model,
        device=args.device,
        lr=args.lr,
        node_loss_weight=1.0,
        graph_loss_weight=1.0,
    )
    
    save_path = save_dir / "best_model.pt"
    hparams = {
        "num_tags": 116,
        "tag_embed_dim": 32,
        "hidden_dim": args.hidden,
        "num_node_classes": 2,
        "num_graph_classes": 2,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "pooling": "mean",
    }
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=15,
        save_path=save_path,
        hparams=hparams,
    )
    
    # Evaluate on original graph
    print("\n" + "=" * 60)
    print("Final Evaluation on Original Graph")
    print("=" * 60)
    
    trainer.load_best(save_path)
    model.eval()
    
    with torch.no_grad():
        base_data = base_data.to(args.device)
        node_logits, graph_logits = model(
            base_data.x,
            base_data.edge_index,
            base_data.tag_indices,
        )
        
        node_probs = F.softmax(node_logits, dim=-1)[:, 1]
        node_preds = node_logits.argmax(dim=-1)
        graph_pred = graph_logits.argmax(dim=-1)
        graph_prob = F.softmax(graph_logits, dim=-1)[:, 1]
        
        print(f"\nGraph-level prediction:")
        print(f"  Predicted: {'VIOLATIONS' if graph_pred.item() == 1 else 'CLEAN'}")
        print(f"  Confidence: {graph_prob.item():.4f}")
        print(f"  Ground truth: {'VIOLATIONS' if base_data.y.item() == 1 else 'CLEAN'}")
        
        print(f"\nNode-level predictions:")
        print(f"  Predicted violating: {node_preds.sum().item()}")
        print(f"  True violating: {base_data.node_y.sum().item()}")
        
        if base_data.node_y.sum() > 0:
            # Top predicted violation nodes
            top_k = 15
            top_indices = node_probs.argsort(descending=True)[:top_k]
            print(f"\n  Top {top_k} predicted violation nodes:")
            for rank, idx in enumerate(top_indices):
                prob = node_probs[idx].item()
                true_label = base_data.node_y[idx].item()
                print(f"    {rank+1}. Node {idx.item()}: prob={prob:.4f} (true={true_label})")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
