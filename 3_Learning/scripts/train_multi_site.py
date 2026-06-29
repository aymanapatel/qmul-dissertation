#!/usr/bin/env python3
"""
train_multi_site.py

Train a DOM-GNN model across multiple websites with multi-label rule classification.

Usage:
    python train_multi_site.py \
        --data-dir /path/to/axe-core \
        --output-dir ./graphs_multi \
        --model-dir ./models_multi \
        --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_extractor import FeatureExtractor, ProcessedPage
from models import DOMAttentionNet
from train import Trainer, get_device
from wcag_rules import NUM_RULES


def find_valid_sites(data_dir: Path) -> List[Path]:
    """Find all site directories with both 0.html and page-0_home.json."""
    valid_sites = []
    for site_dir in sorted(data_dir.iterdir()):
        if site_dir.is_dir():
            html_file = site_dir / "0.html"
            axe_file = site_dir / "page-0_home.json"
            if html_file.exists() and axe_file.exists():
                valid_sites.append(site_dir)
    return valid_sites


def process_site(
    site_dir: Path,
    output_dir: Path,
    extractor: FeatureExtractor,
    extract_visual: bool = False,
    resume: bool = True,
) -> Optional[Data]:
    """Process a single site into a PyG Data object. Cache to disk."""
    site_name = site_dir.name
    cache_path = output_dir / f"{site_name}.pt"

    if resume and cache_path.exists():
        try:
            page = ProcessedPage.load(cache_path)
            # Verify it has multi-label data
            if not hasattr(page.data, "node_y_multi"):
                print(f"  [outdated cache] {site_name} — reprocessing")
                # Fall through to reprocess
            else:
                print(f"  [cached] {site_name} — {page.data.num_nodes} nodes")
                return page.data
        except Exception as e:
            print(f"  [cache corrupt] {site_name}: {e}")

    html_path = site_dir / "0.html"
    axe_path = site_dir / "page-0_home.json"

    try:
        with open(axe_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        num_violations = len(report.get("violations", []))
    except Exception as e:
        print(f"  [skip] {site_name}: cannot read axe report: {e}")
        return None

    try:
        page = extractor.process_page(
            html_path=html_path,
            axe_report_path=axe_path,
            extract_visual=extract_visual,
        )
    except Exception as e:
        print(f"  [skip] {site_name}: processing failed: {e}")
        return None

    # Save cache
    page.save(cache_path)
    print(
        f"  [processed] {site_name} — "
        f"{page.data.num_nodes} nodes, {num_violations} violation types, "
        f"{(page.data.node_y_multi.sum(dim=0) > 0).sum().item()} unique rules"
    )
    return page.data


def compute_feature_dims(data_list: List[Data]) -> Tuple[int, int]:
    """Compute text_dim and attr_dim from data list."""
    text_dim = 384  # default
    for data in data_list:
        if hasattr(data, "text_embeddings") and data.text_embeddings is not None:
            text_dim = data.text_embeddings.shape[1]
            break

    attr_dim = data_list[0].x.shape[1] - text_dim
    return text_dim, attr_dim


def split_data(
    data_list: List[Data],
    site_names: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Data], List[Data], List[Data], List[str], List[str], List[str]]:
    """Stratified split by graph-level violation label."""
    labels = np.array([data.y.item() for data in data_list])
    n = len(data_list)
    
    # Check if we have enough samples for stratified split
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_class_size = counts.min()
    
    if min_class_size < 2 or n < 10:
        # Fallback to random split for small datasets
        print(f"  Warning: Small dataset ({n} samples), using random split")
        indices = np.arange(n)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    else:
        # Stratified split
        test_ratio = 1.0 - train_ratio - val_ratio
        
        try:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=random_state)
            train_idx, temp_idx = next(sss1.split(data_list, labels))
            
            temp_labels = labels[temp_idx]
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_state)
            val_idx_rel, test_idx_rel = next(sss2.split(temp_idx, temp_labels))
            
            val_idx = temp_idx[val_idx_rel]
            test_idx = temp_idx[test_idx_rel]
        except ValueError:
            # Stratified split failed, fall back to random
            print(f"  Warning: Stratified split failed, using random split")
            indices = np.arange(n)
            np.random.seed(random_state)
            np.random.shuffle(indices)
            
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]

    train_names = [site_names[i] for i in train_idx]
    val_names = [site_names[i] for i in val_idx]
    test_names = [site_names[i] for i in test_idx]

    return train_data, val_data, test_data, train_names, val_names, test_names


def main():
    parser = argparse.ArgumentParser(description="Train DOM-GNN across multiple sites")
    # Data paths
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing site subdirectories")
    parser.add_argument("--output-dir", type=str, default="./graphs_multi", help="Directory to cache processed graphs")
    parser.add_argument("--model-dir", type=str, default="./models_multi", help="Directory to save trained models")
    # Processing
    parser.add_argument("--max-sites", type=int, default=None, help="Maximum number of sites (default: all)")
    parser.add_argument("--visual", action="store_true", help="Extract visual features (slower)")
    parser.add_argument("--resume", action="store_true", help="Skip already-cached graphs")
    # Model
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of GAT layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "attention", "meanmax"], help="Graph pooling method")
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (graphs per batch)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--node-loss-weight", type=float, default=1.0, help="Weight for node-level loss")
    parser.add_argument("--rule-loss-weight", type=float, default=5.0, help="Weight for rule-level loss")
    parser.add_argument("--graph-loss-weight", type=float, default=0.5, help="Weight for graph-level loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--hard-neg-weight", type=float, default=10.0, help="Hard negative mining weight")
    parser.add_argument("--hard-pos-weight", type=float, default=5.0, help="Hard positive mining weight")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = get_device() if args.device == "auto" else args.device
    print(f"Using device: {device}")

    print("=" * 70)
    print("Multi-Site DOM-GNN Training (Multi-Label)")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")
    print()

    # Phase 1: Find and process sites
    print("Phase 1: Finding valid sites...")
    valid_sites = find_valid_sites(data_dir)
    print(f"Found {len(valid_sites)} sites with both HTML and axe reports")

    if args.max_sites and args.max_sites < len(valid_sites):
        valid_sites = valid_sites[:args.max_sites]
        print(f"Limited to first {args.max_sites} sites")

    if not valid_sites:
        print("No valid sites found! Exiting.")
        sys.exit(1)

    print(f"\nPhase 2: Processing {len(valid_sites)} sites into graphs...")
    print("-" * 70)

    extractor = FeatureExtractor(device=device)
    data_list = []
    site_names = []
    skipped = 0

    for i, site_dir in enumerate(valid_sites, 1):
        print(f"[{i}/{len(valid_sites)}] ", end="")
        data = process_site(
            site_dir=site_dir,
            output_dir=output_dir,
            extractor=extractor,
            extract_visual=args.visual,
            resume=args.resume,
        )
        if data is not None:
            data_list.append(data)
            site_names.append(site_dir.name)
        else:
            skipped += 1

    print(f"\nSuccessfully loaded {len(data_list)} graphs ({skipped} skipped)")
    if len(data_list) == 0:
        print("No graphs to train on! Exiting.")
        sys.exit(1)

    # Dataset statistics
    violation_sites = sum(1 for d in data_list if d.y.item() == 1)
    clean_sites = len(data_list) - violation_sites
    total_nodes = sum(d.num_nodes for d in data_list)
    total_violation_nodes = sum(d.node_y.sum().item() for d in data_list)

    # Rule statistics
    if hasattr(data_list[0], "node_y_multi"):
        rule_counts = torch.zeros(NUM_RULES)
        for d in data_list:
            rule_counts += (d.node_y_multi.sum(dim=0) > 0).float()
        
        print(f"\nDataset Statistics:")
        print(f"  Sites with violations: {violation_sites} / {len(data_list)}")
        print(f"  Clean sites: {clean_sites}")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Violating nodes: {total_violation_nodes} ({100*total_violation_nodes/total_nodes:.3f}%)")
        print(f"  Rules present: {(rule_counts > 0).sum().item()} / {NUM_RULES}")
        
        # Print top rules
        from wcag_rules import INDEX_TO_RULE
        top_rules = torch.argsort(rule_counts, descending=True)[:10]
        print(f"\nTop 10 rules by site count:")
        for idx in top_rules:
            if rule_counts[idx] > 0:
                print(f"  {INDEX_TO_RULE[idx.item()]}: {rule_counts[idx].int().item()} sites")

    # Phase 3: Split data
    print(f"\nPhase 3: Splitting data (train=70% / val=15% / test=15%)...")
    train_data, val_data, test_data, train_names, val_names, test_names = split_data(
        data_list, site_names, train_ratio=0.7, val_ratio=0.15, random_state=args.seed
    )

    print(f"  Train: {len(train_data)} sites ({sum(d.y.item() for d in train_data)} with violations)")
    print(f"  Val:   {len(val_data)} sites ({sum(d.y.item() for d in val_data)} with violations)")
    print(f"  Test:  {len(test_data)} sites ({sum(d.y.item() for d in test_data)} with violations)")

    # Save split
    split_info = {
        "train": train_names,
        "val": val_names,
        "test": test_names,
        "args": vars(args),
    }
    split_path = model_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved split to {split_path}")

    # Phase 4: Setup model
    print(f"\nPhase 4: Setting up model...")
    text_dim, attr_dim = compute_feature_dims(data_list)
    print(f"  Feature dimensions: text={text_dim}, attr={attr_dim}")
    print(f"  Model: GAT(hidden={args.hidden}, layers={args.layers}, heads={args.heads})")
    print(f"  Multi-label: {NUM_RULES} rules")

    model = DOMAttentionNet(
        num_tags=116,
        tag_embed_dim=32,
        attr_dim=attr_dim,
        text_dim=text_dim,
        hidden_dim=args.hidden,
        num_node_classes=2,
        num_graph_classes=2,
        num_rules=NUM_RULES,
        num_layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 5: Training
    print(f"\nPhase 5: Training...")
    print("-" * 70)

    # Weighted sampler for training
    train_labels = np.array([d.y.item() for d in train_data])
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_data),
        replacement=True,
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        node_loss_weight=args.node_loss_weight,
        rule_loss_weight=args.rule_loss_weight,
        graph_loss_weight=args.graph_loss_weight,
        use_focal_loss=True,
        focal_gamma=args.focal_gamma,
        hard_neg_weight=args.hard_neg_weight,
        hard_pos_weight=args.hard_pos_weight,
    )

    hparams = {
        "num_tags": 116,
        "tag_embed_dim": 32,
        "hidden_dim": args.hidden,
        "num_node_classes": 2,
        "num_graph_classes": 2,
        "num_rules": NUM_RULES,
        "num_layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "pooling": args.pooling,
    }

    save_path = model_dir / "best_model.pt"
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        save_path=save_path,
        hparams=hparams,
    )

    # Save history
    history_path = model_dir / "history.pt"
    torch.save(history, history_path)
    print(f"Saved training history to {history_path}")

    # Phase 6: Final evaluation on test set
    print(f"\nPhase 6: Final Evaluation on Test Set ({len(test_data)} sites)")
    print("=" * 70)

    trainer.load_best(save_path)
    test_metrics = trainer.evaluate(test_loader)

    print(f"Test Loss:          {test_metrics['loss']:.4f}")
    if "node_acc" in test_metrics:
        print(f"Test Node Acc:      {test_metrics['node_acc']:.4f}")
        print(f"Test Node F1(pos):  {test_metrics['node_f1_pos']:.4f}")
    if "rule_f1_micro" in test_metrics:
        print(f"Test Rule F1(micro): {test_metrics['rule_f1_micro']:.4f}")
        print(f"Test Rule F1(macro): {test_metrics['rule_f1_macro']:.4f}")
    if "graph_acc" in test_metrics:
        print(f"Test Graph Acc:     {test_metrics['graph_acc']:.4f}")
        print(f"Test Graph F1:      {test_metrics['graph_f1']:.4f}")
    
    # Per-rule breakdown
    if "top_rules" in test_metrics:
        print(f"\nTop performing rules:")
        for rule, f1 in test_metrics["top_rules"]:
            print(f"  {rule}: {f1:.4f}")
        print(f"\nWorst performing rules:")
        for rule, f1 in test_metrics["worst_rules"]:
            print(f"  {rule}: {f1:.4f}")

    # Save test metrics
    test_metrics_path = model_dir / "test_metrics.json"
    test_metrics_serializable = {}
    for k, v in test_metrics.items():
        if k in ["top_rules", "worst_rules"]:
            test_metrics_serializable[k] = v
        elif hasattr(v, "item"):
            test_metrics_serializable[k] = v.item()
        else:
            test_metrics_serializable[k] = float(v)

    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics_serializable, f, indent=2)
    print(f"\nSaved test metrics to {test_metrics_path}")
    print(f"Saved best model to {save_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
