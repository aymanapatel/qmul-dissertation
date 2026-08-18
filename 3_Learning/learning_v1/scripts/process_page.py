#!/usr/bin/env python3
"""
process_page.py

Main script to process an HTML page into a PyG graph and optionally train a model.
Usage:
    python process_page.py --html /path/to/page.html --axe /path/to/axe.json --output ./graphs
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch

from feature_extractor import FeatureExtractor
from graph_sources import GRAPH_SOURCE_A11Y_TREE, GRAPH_SOURCE_DOM, GRAPH_SOURCE_RENDERED_VISUAL
from models import DOMGCN


def generate_ascii_viz(page, model, device="cpu", max_nodes=50):
    """Generate a simple ASCII/text visualization of predictions."""
    import torch.nn.functional as F
    
    data = page.data.to(device)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        node_logits, graph_logits = model(data.x, data.edge_index, data.tag_indices)
        node_probs = F.softmax(node_logits, dim=-1)[:, 1].cpu()
        node_preds = node_logits.argmax(dim=-1).cpu()
    
    print("\n" + "=" * 70)
    print("NODE-LEVEL PREDICTIONS (sorted by violation probability)")
    print("=" * 70)
    print(f"{'Rank':<6} {'Node':<6} {'Tag':<12} {'Prob':<8} {'True':<6} {'Text'}")
    print("-" * 70)
    
    # Get top nodes by predicted probability
    top_indices = node_probs.argsort(descending=True)
    
    rank = 1
    shown = 0
    for idx in top_indices:
        idx = idx.item()
        prob = node_probs[idx].item()
        pred = node_preds[idx].item()
        true = data.node_y[idx].item() if hasattr(data, "node_y") else -1
        
        node = page.node_map.get(idx)
        if not node:
            continue
        
        tag = node.tag[:12]
        text = node.text_content[:50].replace("\n", " ").strip()
        
        # Highlight true positives and false positives
        marker = ""
        if pred == 1 and true == 1:
            marker = " ✓ TRUE POS"
        elif pred == 1 and true == 0:
            marker = " ✗ FALSE POS"
        elif pred == 0 and true == 1:
            marker = " ✗ FALSE NEG"
        
        print(f"{rank:<6} {idx:<6} {tag:<12} {prob:.4f}  {true:<6} {text}{marker}")
        
        rank += 1
        shown += 1
        if shown >= max_nodes:
            break
    
    print("=" * 70)
    print(f"Graph prediction: {'VIOLATIONS' if graph_logits.argmax(dim=-1).item() == 1 else 'CLEAN'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Process HTML page into GNN graph")
    parser.add_argument("--html", type=str, required=True, help="Path to HTML file")
    parser.add_argument("--axe", type=str, default=None, help="Path to axe-core JSON report")
    parser.add_argument("--output", type=str, default="./graphs", help="Output directory for processed graphs")
    parser.add_argument("--visual", action="store_true", help="Extract visual features via Playwright")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument(
        "--graph-source",
        type=str,
        default=GRAPH_SOURCE_DOM,
        choices=[GRAPH_SOURCE_DOM, GRAPH_SOURCE_A11Y_TREE, GRAPH_SOURCE_RENDERED_VISUAL],
        help="Graph source to build: dom, a11y-tree, or rendered-visual",
    )
    parser.add_argument("--train", action="store_true", help="Run a quick training demo")
    parser.add_argument("--viz", action="store_true", help="Generate ASCII visualization of predictions")
    parser.add_argument("--max-nodes", type=int, default=None, help="Max nodes to parse (for debugging)")
    args = parser.parse_args()
    
    html_path = Path(args.html)
    axe_path = Path(args.axe) if args.axe else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not html_path.exists():
        print(f"Error: HTML file not found: {html_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("DOM-to-GNN Pipeline")
    print("=" * 60)
    print(f"HTML: {html_path}")
    print(f"Axe:  {axe_path}")
    print(f"Visual: {args.visual}")
    print(f"Graph source: {args.graph_source}")
    print()
    
    # Initialize feature extractor
    extractor = FeatureExtractor(device=args.device)
    
    # Process page
    page = extractor.process_page(
        html_path=html_path,
        axe_report_path=axe_path,
        extract_visual=args.visual,
        graph_source=args.graph_source,
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Graph Statistics")
    print("=" * 60)
    print(f"Total nodes: {page.data.num_nodes}")
    print(f"Total edges: {page.data.edge_index.shape[1]}")
    print(f"Node feature dim: {page.data.x.shape[1]}")
    print(f"Graph source: {getattr(page.data, 'graph_source', GRAPH_SOURCE_DOM)}")
    print(f"Node labels: {page.data.node_y.sum().item()} / {len(page.data.node_y)} have violations")
    print(f"Graph label: {'VIOLATIONS' if page.data.y.item() == 1 else 'CLEAN'}")
    
    # Tag distribution
    from collections import Counter
    tag_counts = Counter([node.tag for node in page.node_map.values() if not node.is_text])
    print(f"\nTop 10 most common tags:")
    for tag, count in tag_counts.most_common(10):
        print(f"  {tag}: {count}")
    
    # Save processed graph
    output_path = output_dir / f"{html_path.stem}_graph.pt"
    page.save(output_path)
    
    # Quick training demo if requested
    if args.train:
        print("\n" + "=" * 60)
        print("Training Demo")
        print("=" * 60)
        
        # For demo, we'll create synthetic augmented copies by dropping random edges
        # In practice, you'd have multiple pages
        data_list = [page.data]
        
        # Create a simple model
        # Compute dimensions dynamically from stored text embeddings
        text_dim = (
            page.data.text_embeddings.shape[1]
            if hasattr(page.data, "text_embeddings") and page.data.text_embeddings is not None
            else 384
        )
        attr_dim = page.data.x.shape[1] - text_dim
        model = DOMGCN(
            num_tags=116,
            tag_embed_dim=32,
            attr_dim=attr_dim,
            text_dim=text_dim,
            hidden_dim=64,
            num_node_classes=2,
            num_graph_classes=2,
            num_layers=2,
            dropout=0.3,
            pooling="mean",
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Since we only have one page, do a quick forward pass demo
        model = model.to(args.device)
        page.data = page.data.to(args.device)
        
        with torch.no_grad():
            node_logits, graph_logits = model(
                page.data.x,
                page.data.edge_index,
                page.data.tag_indices,
            )
            
            node_preds = node_logits.argmax(dim=-1)
            graph_pred = graph_logits.argmax(dim=-1)
            
            print(f"\nForward pass results:")
            print(f"  Node logits shape: {node_logits.shape}")
            print(f"  Graph logits shape: {graph_logits.shape}")
            print(f"  Predicted violating nodes: {node_preds.sum().item()}")
            print(f"  Predicted page label: {'VIOLATIONS' if graph_pred.item() == 1 else 'CLEAN'}")
            
            # Compare with ground truth
            if page.data.node_y is not None:
                true_violating = page.data.node_y.sum().item()
                print(f"  True violating nodes: {true_violating}")
                
                if true_violating > 0:
                    # Find top predicted violation nodes
                    node_probs = torch.softmax(node_logits, dim=-1)[:, 1]
                    top_k = min(10, len(node_probs))
                    top_indices = node_probs.argsort(descending=True)[:top_k]
                    print(f"\n  Top {top_k} predicted violation nodes:")
                    for idx in top_indices:
                        node = page.node_map.get(idx.item())
                        if node:
                            print(f"    Node {idx.item()}: <{node.tag}> {node.text_content[:60]}...")
    
    # Generate ASCII visualization if requested
    if args.viz or args.train:
        print("\nGenerating visualization...")
        text_dim = (
            page.data.text_embeddings.shape[1]
            if hasattr(page.data, "text_embeddings") and page.data.text_embeddings is not None
            else 384
        )
        attr_dim = page.data.x.shape[1] - text_dim
        viz_model = DOMGCN(
            num_tags=116,
            tag_embed_dim=32,
            attr_dim=attr_dim,
            text_dim=text_dim,
            hidden_dim=64,
            num_node_classes=2,
            num_graph_classes=2,
            num_layers=2,
            dropout=0.3,
            pooling="mean",
        )
        generate_ascii_viz(page, viz_model, args.device, max_nodes=30)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
