"""Build a11y-tree graphs from crawler-captured same-session AX sidecars.

Unlike ``regenerate_live_ax_cache``, this module never launches a browser or
reconstructs an accessibility tree from saved HTML.  It uses the Chromium AX
tree and backend-DOM mapping captured by the crawler in ``0.ax.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from bs4 import BeautifulSoup, Tag
from torch_geometric.data import Data

from .collection_selection import select_multilabel_stratified_sites, select_split_sites
from .data import split_hash
from .regenerate_live_ax_cache import (
    EDGE_AX_RELATION,
    EDGE_PARENT_CHILD,
    EDGE_SIBLING,
    RELATION_NAMES,
    _node_features,
)


SAME_SESSION_AX_FEATURE_VERSION = 2
SNAPSHOT_MARKER_ATTRIBUTE = "data-gnn-node-id"
IGNORED_HTML_TAGS = {"script", "style", "noscript"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapping_coverage(ax_path: Path) -> tuple[float, dict[str, int]]:
    """Return the captured AX-to-snapshot coverage recorded by the crawler."""
    payload = json.loads(ax_path.read_text(encoding="utf-8"))
    stats = payload.get("mapping_stats") or {}
    try:
        total = int(stats.get("ax_nodes", 0))
        mapped = int(stats.get("ax_nodes_mapped_to_snapshot", 0))
    except (TypeError, ValueError):
        total, mapped = 0, 0
    if total <= 0:
        nodes = [node for node in payload.get("nodes", []) if isinstance(node, dict) and not node.get("ignored")]
        mapping = {str(key) for key in (payload.get("backend_dom_to_snapshot_node") or {})}
        total = len(nodes)
        mapped = sum(str(node.get("backendDOMNodeId")) in mapping for node in nodes)
    return (mapped / total if total else 0.0), {"ax_nodes": total, "ax_nodes_mapped_to_snapshot": mapped}


def _load_html_records(html_path: Path, visual_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Load marked rendered HTML and attach same-session visual geometry."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")
    visual_payload = json.loads(visual_path.read_text(encoding="utf-8"))
    visual_by_marker = {
        str(item.get("snapshot_node_id")): item
        for item in visual_payload.get("nodes", [])
        if item.get("snapshot_node_id") is not None
    }
    records: dict[int, dict[str, Any]] = {}
    for element in soup.find_all(True):
        marker = element.attrs.get(SNAPSHOT_MARKER_ATTRIBUTE)
        if marker is None or element.name in IGNORED_HTML_TAGS:
            continue
        try:
            marker_id = int(marker)
        except (TypeError, ValueError):
            continue
        parent_marker: int | None = None
        parent = element.parent
        while isinstance(parent, Tag):
            value = parent.attrs.get(SNAPSHOT_MARKER_ATTRIBUTE)
            if value is not None:
                try:
                    parent_marker = int(value)
                except (TypeError, ValueError):
                    pass
                break
            parent = parent.parent
        visual = visual_by_marker.get(str(marker_id), {})
        attributes = {
            str(key): " ".join(map(str, value)) if isinstance(value, list) else str(value)
            for key, value in element.attrs.items()
            if key != SNAPSHOT_MARKER_ATTRIBUTE
        }
        records[marker_id] = {
            "index": marker_id,
            "parent_index": parent_marker,
            "tag": str(element.name or "span"),
            "attrs": attributes,
            "text": element.get_text(" ", strip=True)[:2000],
            "x": visual.get("x", -1),
            "y": visual.get("y", -1),
            "width": visual.get("width", -1),
            "height": visual.get("height", -1),
            "visible": bool(visual.get("visible", False)),
        }
    return records, _selector_to_marker_map(soup)


def _selector_to_marker_map(soup: BeautifulSoup) -> dict[str, int | None]:
    """Return a lazy selector cache populated by ``_target_marker`` below."""
    # Retaining the parsed DOM avoids loading the snapshot in a new browser.
    return {"__soup__": soup}  # type: ignore[return-value]


def _target_marker(selector_cache: dict[str, int | None], selector: str) -> int | None:
    if selector in selector_cache:
        return selector_cache[selector]
    soup = selector_cache["__soup__"]  # type: ignore[assignment]
    assert isinstance(soup, BeautifulSoup)
    try:
        match = soup.select_one(selector)
    except Exception:
        match = None
    marker: int | None = None
    if isinstance(match, Tag):
        value = match.attrs.get(SNAPSHOT_MARKER_ATTRIBUTE)
        try:
            marker = int(value) if value is not None else None
        except (TypeError, ValueError):
            marker = None
    selector_cache[selector] = marker
    return marker


def _sidecar_ax_nodes(ax_path: Path) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, int], dict[str, Any]]:
    payload = json.loads(ax_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes")
    mapping = payload.get("backend_dom_to_snapshot_node")
    if not isinstance(nodes, list) or not isinstance(mapping, dict):
        raise ValueError("0.ax.json must contain nodes and backend_dom_to_snapshot_node")
    included = [node for node in nodes if isinstance(node, dict) and not node.get("ignored")]
    if not included:
        raise ValueError("0.ax.json has no non-ignored AX nodes")
    backend_to_marker: dict[int, int] = {}
    for backend, marker in mapping.items():
        try:
            backend_to_marker[int(backend)] = int(marker)
        except (TypeError, ValueError):
            continue
    backend_to_ax = {
        int(node["backendDOMNodeId"]): index
        for index, node in enumerate(included)
        if node.get("backendDOMNodeId") is not None
    }
    return included, backend_to_marker, backend_to_ax, payload


def build_site(site_dir: Path, output: Path, text_model) -> dict[str, Any]:
    html_path = site_dir / "0.html"
    visual_path = site_dir / "0.visual.json"
    ax_path = site_dir / "0.ax.json"
    axe_path = site_dir / "page-0_home.json"
    records, selector_cache = _load_html_records(html_path, visual_path)
    included, backend_to_marker, backend_to_ax, ax_payload = _sidecar_ax_nodes(ax_path)
    report = json.loads(axe_path.read_text(encoding="utf-8"))

    marker_to_ax = {
        backend_to_marker[backend]: ax_index
        for backend, ax_index in backend_to_ax.items()
        if backend in backend_to_marker
    }
    rows: list[torch.Tensor] = []
    tag_indices: list[int] = []
    backend_ids: list[int] = []
    marker_ids: list[int] = []
    names: list[str] = []
    for node in included:
        backend = int(node.get("backendDOMNodeId", -1))
        marker = backend_to_marker.get(backend, -1)
        features, tag_index, name = _node_features(node, records.get(marker))
        rows.append(features)
        tag_indices.append(tag_index)
        backend_ids.append(backend)
        marker_ids.append(marker)
        names.append(name)
    embeddings = text_model.encode(names, batch_size=64, convert_to_tensor=True, show_progress_bar=False).detach().cpu().float()
    features = torch.cat([torch.stack(rows), embeddings], dim=1)

    ax_index = {str(node.get("nodeId")): index for index, node in enumerate(included)}
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_types: list[int] = []
    for parent, node in enumerate(included):
        children = [ax_index[str(child)] for child in node.get("childIds", []) if str(child) in ax_index]
        for child in children:
            edge_src.append(parent); edge_dst.append(child); edge_types.append(EDGE_PARENT_CHILD)
        for left, right in zip(children, children[1:]):
            edge_src.extend([left, right]); edge_dst.extend([right, left]); edge_types.extend([EDGE_SIBLING, EDGE_SIBLING])
        for property_ in node.get("properties", []):
            if property_.get("name") not in RELATION_NAMES:
                continue
            related = (property_.get("value") or {}).get("relatedNodes", []) or []
            for item in related:
                backend = item.get("backendDOMNodeId")
                try:
                    related_index = backend_to_ax.get(int(backend))
                except (TypeError, ValueError):
                    related_index = None
                if related_index is not None:
                    edge_src.append(parent); edge_dst.append(related_index); edge_types.append(EDGE_AX_RELATION)

    from wcag_rules import NUM_RULES, RULE_INDEX, rule_mask_for_graph_source

    labels = torch.zeros((len(included), NUM_RULES), dtype=torch.float32)
    mapping_loss: list[dict[str, str]] = []
    for violation in report.get("violations", []):
        rule_id = str(violation.get("id", ""))
        if rule_id not in RULE_INDEX:
            continue
        for violation_node in violation.get("nodes", []):
            target = violation_node.get("target") or []
            selector = str(target[0]) if target and isinstance(target[0], str) else ""
            marker = _target_marker(selector_cache, selector) if selector else None
            current = marker if marker is not None else -1
            while current >= 0 and current not in marker_to_ax:
                parent = records.get(current, {}).get("parent_index")
                current = int(parent) if parent is not None else -1
            if current in marker_to_ax:
                labels[marker_to_ax[current], RULE_INDEX[rule_id]] = 1.0
            else:
                mapping_loss.append({"rule_id": rule_id, "selector": selector})

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.empty((2, 0), dtype=torch.long)
    data = Data(
        x=features,
        edge_index=edge_index,
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        tag_indices=torch.tensor(tag_indices, dtype=torch.long),
        node_y_multi=labels,
        node_y=labels.bool().any(dim=1).long(),
        y=torch.tensor([int(labels.any())]),
        num_nodes=len(included),
    )
    data.graph_source = "a11y-tree"
    data.live_accessibility_tree = True
    data.live_ax_feature_version = SAME_SESSION_AX_FEATURE_VERSION
    data.ax_capture_provenance = "same_session_sidecar"
    data.backend_dom_node_ids = torch.tensor(backend_ids, dtype=torch.long)
    data.dom_indices = torch.tensor(marker_ids, dtype=torch.long)
    data.available_rule_mask = rule_mask_for_graph_source("a11y-tree").unsqueeze(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "data": data,
        "html_path": str(html_path),
        "ax_path": str(ax_path),
        "num_nodes": len(included),
        "graph_source": "a11y-tree",
        "collector": "same-session-cdp-ax-sidecar-v1",
    }, output)
    return {
        "site_id": site_dir.name,
        "status": "captured",
        "path": str(output),
        "nodes": len(included),
        "edges": len(edge_types),
        "feature_dim": int(data.x.shape[1]),
        "feature_version": SAME_SESSION_AX_FEATURE_VERSION,
        "dom_mapped_nodes": sum(marker >= 0 for marker in marker_ids),
        "sidecar_dom_mapped_nodes": len(backend_to_marker),
        "positive_labels": int(labels.sum()),
        "mapping_loss_count": len(mapping_loss),
        "mapping_loss_examples": mapping_loss[:20],
        "source_html_sha256": _sha256(html_path),
        "source_visual_sha256": _sha256(visual_path),
        "source_ax_sha256": _sha256(ax_path),
        "source_axe_sha256": _sha256(axe_path),
        "ax_mapping_stats": ax_payload.get("mapping_stats", {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    split = json.loads(args.split.read_text(encoding="utf-8"))
    rules = set(args.selection_rules or [])
    positives = {rule: set() for rule in rules}
    for partition in ("train", "val", "test"):
        for site in split[partition]:
            report_path = args.corpus_dir / site / "page-0_home.json"
            if report_path.is_file():
                found = {str(item.get("id")) for item in json.loads(report_path.read_text()).get("violations", [])}
                for rule in rules & found:
                    positives[rule].add(site)
    ordered = select_multilabel_stratified_sites(
        split, positives, args.max_sites, args.positive_fraction, args.minimum_positive_sites_per_rule
    ) if rules else select_split_sites(split, args.max_sites)
    exclusions: list[dict[str, Any]] = []
    eligible_ordered: list[tuple[str, str]] = []
    for partition, site_id in ordered:
        ax_path = args.corpus_dir / site_id / "0.ax.json"
        if not ax_path.is_file():
            eligible_ordered.append((partition, site_id))
            continue
        try:
            ratio, stats = mapping_coverage(ax_path)
        except Exception as exc:
            exclusions.append({
                "site_id": site_id,
                "partition": partition,
                "status": "excluded_mapping_coverage_unreadable",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            })
            continue
        if ratio < args.minimum_ax_mapping_ratio:
            exclusions.append({
                "site_id": site_id,
                "partition": partition,
                "status": "excluded_low_ax_mapping_coverage",
                "mapping_ratio": ratio,
                **stats,
            })
        else:
            eligible_ordered.append((partition, site_id))
    selected_split = {
        "seed": split.get("seed", 42),
        **{partition: [site for name, site in eligible_ordered if name == partition] for partition in ("train", "val", "test")},
    }
    selected_split["split_hash"] = split_hash(selected_split)
    model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)
    records = list(exclusions)
    for index, (partition, site_id) in enumerate(eligible_ordered, 1):
        print(f"[{index}/{len(eligible_ordered)}] {partition} {site_id}", flush=True)
        site_dir = args.corpus_dir / site_id
        output = args.output_dir / site_id / "a11y-tree.pt"
        needed = [site_dir / name for name in ("0.html", "0.visual.json", "0.ax.json", "page-0_home.json")]
        if not all(path.is_file() for path in needed):
            records.append({"site_id": site_id, "partition": partition, "status": "missing_source"})
            continue
        if args.resume and output.is_file():
            try:
                raw = torch.load(output, map_location="cpu", weights_only=False)["data"]
                if getattr(raw, "ax_capture_provenance", "") == "same_session_sidecar" and int(getattr(raw, "live_ax_feature_version", 0)) == SAME_SESSION_AX_FEATURE_VERSION:
                    records.append({"site_id": site_id, "partition": partition, "status": "reused", "path": str(output), "nodes": int(raw.num_nodes)})
                    continue
            except Exception:
                pass
        try:
            record = build_site(site_dir, output, model)
            record["partition"] = partition
            records.append(record)
        except Exception as exc:
            records.append({"site_id": site_id, "partition": partition, "status": "collection_failed", "error": f"{type(exc).__name__}: {exc}"[:1000]})
    counts = {status: sum(item["status"] == status for item in records) for status in sorted({item["status"] for item in records})}
    manifest = {
        "schema_version": 1,
        "collector": "same-session-cdp-ax-sidecar-v1",
        "live_ax_feature_version": SAME_SESSION_AX_FEATURE_VERSION,
        "split": str(args.split.resolve()),
        "split_sha256": _sha256(args.split),
        "split_hash": split.get("split_hash"),
        "corpus_dir": str(args.corpus_dir.resolve()),
        "requested_sites": len(ordered),
        "eligible_sites": len(eligible_ordered),
        "outcome_counts": counts,
        "records": records,
        "selection": {"rules": sorted(rules), "positive_fraction": args.positive_fraction, "minimum_positive_sites_per_rule": args.minimum_positive_sites_per_rule},
        "mapping_coverage_policy": {
            "minimum_ax_mapping_ratio": args.minimum_ax_mapping_ratio,
            "excluded_site_count": len(exclusions),
            "exclusions": exclusions,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "collection_split.json").write_text(json.dumps(selected_split, indent=2), encoding="utf-8")
    (args.output_dir / "same_session_ax_cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-sites", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-rules", nargs="+")
    parser.add_argument("--positive-fraction", type=float, default=0.6)
    parser.add_argument("--minimum-positive-sites-per-rule", type=int, default=1)
    parser.add_argument("--minimum-ax-mapping-ratio", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.minimum_ax_mapping_ratio <= 1:
        raise SystemExit("--minimum-ax-mapping-ratio must be between 0 and 1")
    report = run(args)
    print(json.dumps({"requested_sites": report["requested_sites"], "eligible_sites": report["eligible_sites"], "outcomes": report["outcome_counts"]}, indent=2))


if __name__ == "__main__":
    main()
