"""Cached graph loading, label sanitisation, and site-level splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from .schema import FeatureContract
from .feature_layout import STRUCTURAL_ATTRIBUTE_DIM, RENDERED_VISUAL_ATTRIBUTE_DIM


def discover_cached_graphs(cache_dir: Path, graph_source: str) -> dict[str, Path]:
    paths = {path.parent.name: path for path in sorted(cache_dir.glob(f"*/{graph_source}.pt"))}
    if not paths:
        raise FileNotFoundError(f"No {graph_source}.pt graphs found beneath {cache_dir}")
    return paths


def _load_raw_graph(path: Path) -> Data:
    container = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(container, dict) or "data" not in container:
        raise ValueError(f"Unsupported graph cache container: {path}")
    return container["data"]


def sanitise_graph(
    raw: Data,
    *,
    graph_source: str,
    rule_indices: Sequence[int],
    require_labels: bool,
    site_id: str = "",
    cache_path: str = "",
    feature_variant: str = "full",
) -> Data:
    actual_source = getattr(raw, "graph_source", None)
    if actual_source != graph_source:
        raise ValueError(f"Expected {graph_source} graph, received {actual_source}")
    graph = Data(
        x=raw.x.detach().cpu().float(),
        edge_index=raw.edge_index.detach().cpu().long(),
        tag_indices=raw.tag_indices.detach().cpu().long(),
    )
    if hasattr(raw, "edge_type"):
        graph.edge_type = raw.edge_type.detach().cpu().long()
    graph.graph_source = graph_source
    graph.site_id = site_id
    graph.cache_path = cache_path
    graph.rendered_visual_feature_version = int(getattr(raw, "rendered_visual_feature_version", 0))
    graph.live_ax_feature_version = int(getattr(raw, "live_ax_feature_version", 0))
    graph.live_accessibility_tree = bool(getattr(raw, "live_accessibility_tree", False))
    for name in ("backend_dom_node_ids", "dom_indices"):
        value = getattr(raw, name, None)
        if value is not None:
            setattr(graph, name, value.detach().cpu().long())
    if feature_variant not in {"full", "without_visual_features", "without_spatial_edges", "structure_only"}:
        raise ValueError(f"Unknown feature variant: {feature_variant}")
    if feature_variant != "full":
        if graph_source != "rendered-visual":
            raise ValueError("Visual feature variants apply only to rendered-visual graphs")
        if graph.x.shape[1] < STRUCTURAL_ATTRIBUTE_DIM + RENDERED_VISUAL_ATTRIBUTE_DIM:
            raise ValueError("Rendered feature tail is absent; regenerate the visual cache before ablation")
        if feature_variant in {"without_visual_features", "structure_only"}:
            graph.x[:, STRUCTURAL_ATTRIBUTE_DIM:STRUCTURAL_ATTRIBUTE_DIM + RENDERED_VISUAL_ATTRIBUTE_DIM] = 0.0
        if feature_variant in {"without_spatial_edges", "structure_only"}:
            if not hasattr(graph, "edge_type"):
                raise ValueError("Typed edges are required for the spatial-edge ablation")
            keep = graph.edge_type != 2
            graph.edge_index = graph.edge_index[:, keep]
            graph.edge_type = graph.edge_type[keep]
    graph.feature_variant = feature_variant
    for name in ("rendered_visible_mask", "visual_match_found_mask", "rendered_visual_label_qa_mask"):
        value = getattr(raw, name, None)
        if value is not None:
            setattr(graph, name, value.detach().cpu().bool())
    label_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    if graph_source == "rendered-visual" and hasattr(graph, "rendered_visible_mask"):
        label_mask &= graph.rendered_visible_mask
    graph.label_mask = label_mask
    if require_labels:
        labels = getattr(raw, "node_y_multi", None)
        if labels is None:
            raise ValueError("Training labels are missing from the graph")
        if labels.dim() != 2 or not rule_indices or max(rule_indices) >= labels.shape[1]:
            raise ValueError("Rule labels have an incompatible shape")
        graph.rule_y = labels[:, list(rule_indices)].detach().cpu().float()
        graph.node_y = graph.rule_y.bool().any(dim=1).float()
        graph.page_y = (graph.node_y.bool() & graph.label_mask).any().float().reshape(1)
    return graph


def load_cached_graph(path: Path, *, graph_source: str, rule_indices: Sequence[int], require_labels: bool, feature_variant: str = "full") -> Data:
    return sanitise_graph(
        _load_raw_graph(path), graph_source=graph_source, rule_indices=rule_indices,
        require_labels=require_labels, site_id=path.parent.name, cache_path=str(path), feature_variant=feature_variant,
    )


def page_label(path: Path, rule_indices: Sequence[int]) -> int:
    raw = _load_raw_graph(path)
    labels = getattr(raw, "node_y_multi", None)
    if labels is None:
        raise ValueError(f"Training labels are missing from {path}")
    mask = torch.ones(labels.shape[0], dtype=torch.bool)
    if getattr(raw, "graph_source", "") == "rendered-visual" and hasattr(raw, "rendered_visible_mask"):
        mask &= raw.rendered_visible_mask.bool()
    return int(labels[mask][:, list(rule_indices)].bool().any().item())


def _safe_split(items: list[str], labels: list[int], *, test_size: float, seed: int) -> tuple[list[str], list[str]]:
    stratify = labels if len(set(labels)) > 1 and min(labels.count(0), labels.count(1)) >= 2 else None
    left, right = train_test_split(items, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify)
    return sorted(left), sorted(right)


def create_site_split(
    graph_paths: dict[str, Path], *, rule_indices: Sequence[int], seed: int = 42,
    train_ratio: float = 0.70, val_ratio: float = 0.15,
) -> dict:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios")
    site_ids = sorted(graph_paths)
    if len(site_ids) < 6:
        raise ValueError("At least six cached sites are required for a three-way split")
    labels = [page_label(graph_paths[site], rule_indices) for site in site_ids]
    train_sites, remainder = _safe_split(site_ids, labels, test_size=1.0 - train_ratio, seed=seed)
    label_by_site = dict(zip(site_ids, labels))
    remainder_labels = [label_by_site[site] for site in remainder]
    test_fraction = (1.0 - train_ratio - val_ratio) / (1.0 - train_ratio)
    val_sites, test_sites = _safe_split(remainder, remainder_labels, test_size=test_fraction, seed=seed)
    split = {"seed": seed, "train": train_sites, "val": val_sites, "test": test_sites}
    validate_split(split, available_sites=set(site_ids))
    return split


def validate_split(split: dict, *, available_sites: set[str] | None = None) -> None:
    train, val, test = map(lambda name: set(split.get(name, [])), ("train", "val", "test"))
    if not train or not val or not test:
        raise ValueError("Train, validation, and test splits must all be non-empty")
    if train & val or train & test or val & test:
        raise ValueError("Site leakage detected between dataset splits")
    if available_sites is not None and (train | val | test) - available_sites:
        raise ValueError("Split references missing cached sites")


def split_hash(split: dict) -> str:
    stable = {key: split[key] for key in ("seed", "train", "val", "test") if key in split}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def save_split(path: Path, split: dict, *, graph_source: str, rule_ids: Sequence[str]) -> None:
    payload = {"schema_version": 2, "graph_source": graph_source, "rule_ids": list(rule_ids), **split}
    payload["split_hash"] = split_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: Path, *, available_sites: set[str] | None = None) -> dict:
    split = json.loads(path.read_text(encoding="utf-8"))
    validate_split(split, available_sites=available_sites)
    expected = split.get("split_hash")
    if expected and expected != split_hash(split):
        raise ValueError("Split hash mismatch")
    return split


def load_graphs(
    site_ids: Iterable[str], graph_paths: dict[str, Path], *, graph_source: str,
    rule_indices: Sequence[int], contract: FeatureContract | None = None, feature_variant: str = "full",
) -> list[Data]:
    graphs = []
    for site_id in site_ids:
        graph = load_cached_graph(graph_paths[site_id], graph_source=graph_source, rule_indices=rule_indices, require_labels=True, feature_variant=feature_variant)
        if contract is not None:
            contract.validate(graph)
        graphs.append(graph)
    return graphs
