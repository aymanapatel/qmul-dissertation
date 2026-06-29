#!/usr/bin/env python3
"""
visualize_graph.py

Visualize the DOM graph structure and model predictions.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from feature_extractor import ProcessedPage
from models import DOMGCN


def visualize_predictions(page: ProcessedPage, model: DOMGCN, device: str = "cpu", max_nodes: int = 200):
    """Visualize node predictions on the DOM graph."""
    model = model.to(device)
    model.eval()
    
    data = page.data.to(device)
    with torch.no_grad():
        node_logits, graph_logits = model(data.x, data.edge_index, data.tag_indices)
        node_probs = F.softmax(node_logits, dim=-1)[:, 1].cpu()
        node_preds = node_logits.argmax(dim=-1).cpu()
    
    # Convert to NetworkX for visualization
    # Limit to first max_nodes for readability
    subset = torch.arange(min(max_nodes, data.num_nodes))
    
    # Create a subgraph
    edge_mask = (data.edge_index[0] < max_nodes) & (data.edge_index[1] < max_nodes)
    sub_edge_index = data.edge_index[:, edge_mask]
    
    G = nx.DiGraph()
    num_viz = min(max_nodes, data.num_nodes)
    for i in range(num_viz):
        node = page.node_map.get(i)
        tag = node.tag if node else "unknown"
        true_label = data.node_y[i].item() if hasattr(data, "node_y") and data.node_y is not None else 0
        G.add_node(i, 
                  tag=tag,
                  pred=node_preds[i].item(),
                  true=true_label,
                  prob=node_probs[i].item())
    
    for src, dst in sub_edge_index.t().tolist():
        G.add_edge(src, dst)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Layout
    pos = nx.spring_layout(G.to_undirected(), k=0.5, iterations=50, seed=42)
    
    # Plot 1: Tag types
    tag_colors = {}
    for node in G.nodes():
        tag = G.nodes[node]["tag"]
        if tag not in tag_colors:
            tag_colors[tag] = len(tag_colors)
    
    color_map = plt.colormaps['tab20']
    node_colors = [color_map(tag_colors[G.nodes[n]["tag"]] % 20) for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, ax=axes[0], alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.1, ax=axes[0], arrows=False)
    axes[0].set_title("DOM Structure by Tag Type")
    axes[0].axis("off")
    
    # Plot 2: Ground truth violations
    true_violators = [n for n in G.nodes() if G.nodes[n]["true"] == 1]
    true_normal = [n for n in G.nodes() if G.nodes[n]["true"] == 0]
    
    nx.draw_networkx_nodes(G, pos, nodelist=true_normal, node_color="lightblue", 
                          node_size=50, ax=axes[1], alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=true_violators, node_color="red", 
                          node_size=200, ax=axes[1], alpha=1.0, label="Violation")
    nx.draw_networkx_edges(G, pos, alpha=0.1, ax=axes[1], arrows=False)
    axes[1].set_title("Ground Truth Violations")
    axes[1].legend()
    axes[1].axis("off")
    
    # Plot 3: Predicted violations
    pred_violators = [n for n in G.nodes() if G.nodes[n]["pred"] == 1]
    pred_normal = [n for n in G.nodes() if G.nodes[n]["pred"] == 0]
    
    nx.draw_networkx_nodes(G, pos, nodelist=pred_normal, node_color="lightgreen", 
                          node_size=50, ax=axes[2], alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=pred_violators, node_color="orange", 
                          node_size=150, ax=axes[2], alpha=0.8, label="Predicted")
    
    # Highlight true positives
    true_positives = [n for n in G.nodes() if G.nodes[n]["true"] == 1 and G.nodes[n]["pred"] == 1]
    if true_positives:
        nx.draw_networkx_nodes(G, pos, nodelist=true_positives, node_color="darkgreen", 
                              node_size=300, ax=axes[2], alpha=1.0, label="True Positive")
    
    nx.draw_networkx_edges(G, pos, alpha=0.1, ax=axes[2], arrows=False)
    axes[2].set_title("Predicted Violations")
    axes[2].legend()
    axes[2].axis("off")
    
    plt.tight_layout()
    return fig


def plot_training_history(history: dict, output_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    
    axes[0, 1].plot(epochs, history["train_node_acc"], label="Train Node Acc")
    if history.get("val_node_acc"):
        axes[0, 1].plot(epochs, history["val_node_acc"], label="Val Node Acc")
    axes[0, 1].set_title("Node Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    
    axes[1, 0].plot(epochs, history["train_graph_acc"], label="Train Graph Acc")
    if history.get("val_graph_acc"):
        axes[1, 0].plot(epochs, history["val_graph_acc"], label="Val Graph Acc")
    axes[1, 0].set_title("Graph Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    
    if history.get("val_f1"):
        axes[1, 1].plot(epochs, history["val_f1"], label="Val F1")
    axes[1, 1].set_title("F1 Score")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved training plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DOM graph predictions")
    parser.add_argument("--graph", type=str, required=True, help="Path to processed graph .pt")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .pt")
    parser.add_argument("--output", type=str, default="./visualizations", help="Output directory")
    parser.add_argument("--max-nodes", type=int, default=200, help="Max nodes to visualize")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()
    
    graph_path = Path(args.graph)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load page
    page = ProcessedPage.load(graph_path)
    
    # Compute dimensions dynamically from stored text embeddings
    text_dim = (
        page.data.text_embeddings.shape[1]
        if hasattr(page.data, "text_embeddings") and page.data.text_embeddings is not None
        else 384
    )
    attr_dim = page.data.x.shape[1] - text_dim
    
    # Load model if provided
    if args.model and Path(args.model).exists():
        model = DOMGCN(
            num_tags=116, tag_embed_dim=32, attr_dim=attr_dim, text_dim=text_dim,
            hidden_dim=64, num_node_classes=2, num_graph_classes=2,
            num_layers=2, dropout=0.3, pooling="mean",
        )
        checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {args.model}")
    else:
        print("No model provided — using random initialization")
        model = DOMGCN(
            num_tags=116, tag_embed_dim=32, attr_dim=attr_dim, text_dim=text_dim,
            hidden_dim=64, num_node_classes=2, num_graph_classes=2,
            num_layers=2, dropout=0.3, pooling="mean",
        )
    
    # Visualize
    print("Creating visualization...")
    fig = visualize_predictions(page, model, args.device, args.max_nodes)
    viz_path = output_dir / f"{graph_path.stem}_predictions.png"
    fig.savefig(viz_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {viz_path}")
    
    # Plot history if available
    history_path = Path(args.model).parent / "history.pt" if args.model else None
    if history_path and history_path.exists():
        history = torch.load(history_path, weights_only=False)
        plot_path = output_dir / "training_history.png"
        plot_training_history(history, plot_path)


if __name__ == "__main__":
    main()
