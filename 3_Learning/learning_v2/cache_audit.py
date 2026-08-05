"""Audit cached graph views before dissertation experiments consume them."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .data import discover_cached_graphs
from .feature_layout import (
    MINIMUM_RENDERED_FEATURE_DIM, RENDERED_VISUAL_ATTRIBUTE_DIM,
    RENDERED_VISUAL_FEATURE_VERSION, STRUCTURAL_ATTRIBUTE_DIM, TEXT_EMBEDDING_DIM,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cache(cache_dir: Path, views: tuple[str, ...] = ("dom", "a11y-tree", "rendered-visual")) -> dict[str, Any]:
    paths = {view: discover_cached_graphs(cache_dir, view) for view in views}
    all_sites = set().union(*(set(values) for values in paths.values()))
    aligned = set.intersection(*(set(values) for values in paths.values()))
    view_summary: dict[str, Any] = {}
    rendered_failures = []
    live_ax_failures = []
    for view, site_paths in paths.items():
        feature_dims = Counter(); versions = Counter(); live_ax_versions = Counter(); invalid = []
        for site, path in site_paths.items():
            try:
                container = torch.load(path, map_location="cpu", weights_only=False)
                raw = container["data"]
                feature_dims[int(raw.x.shape[1])] += 1
                version = int(getattr(raw, "rendered_visual_feature_version", 0))
                versions[version] += 1
                live_ax_version = int(getattr(raw, "live_ax_feature_version", 0))
                live_ax_versions[live_ax_version] += 1
                if str(getattr(raw, "graph_source", "")) != view:
                    invalid.append({"site_id": site, "reason": "graph_source_mismatch"})
                if view == "rendered-visual":
                    reasons = []
                    if version < RENDERED_VISUAL_FEATURE_VERSION:
                        reasons.append(f"feature_version_{version}_below_{RENDERED_VISUAL_FEATURE_VERSION}")
                    if int(raw.x.shape[1]) < MINIMUM_RENDERED_FEATURE_DIM:
                        reasons.append(f"feature_dim_{int(raw.x.shape[1])}_below_{MINIMUM_RENDERED_FEATURE_DIM}")
                    if not hasattr(raw, "visual_match_found_mask"):
                        reasons.append("visual_match_mask_missing")
                    if reasons:
                        rendered_failures.append({"site_id": site, "reasons": reasons})
                if view == "a11y-tree":
                    reasons = []
                    if not bool(getattr(raw, "live_accessibility_tree", False)) or live_ax_version < 1:
                        reasons.append("chromium_live_ax_provenance_missing")
                    if getattr(raw, "ax_capture_provenance", "") != "same_session_sidecar":
                        reasons.append("same_session_sidecar_provenance_missing")
                    if reasons:
                        live_ax_failures.append({"site_id": site, "reasons": reasons})
            except Exception as exc:
                invalid.append({"site_id": site, "reason": f"load_failed:{type(exc).__name__}"})
        view_summary[view] = {
            "site_count": len(site_paths),
            "feature_dimensions": {str(key): value for key, value in sorted(feature_dims.items())},
            "rendered_visual_feature_versions": {str(key): value for key, value in sorted(versions.items())},
            "live_ax_feature_versions": {str(key): value for key, value in sorted(live_ax_versions.items())},
            "invalid_count": len(invalid),
            "invalid_examples": invalid[:20],
        }
    return {
        "schema_version": 1,
        "cache_dir": str(cache_dir.resolve()),
        "site_count_union": len(all_sites),
        "aligned_site_count": len(aligned),
        "views": view_summary,
        "rendered_visual_contract": {
            "minimum_feature_version": RENDERED_VISUAL_FEATURE_VERSION,
            "minimum_feature_dimension": MINIMUM_RENDERED_FEATURE_DIM,
            "structural_attribute_dimension": STRUCTURAL_ATTRIBUTE_DIM,
            "rendered_visual_attribute_dimension": RENDERED_VISUAL_ATTRIBUTE_DIM,
            "text_embedding_dimension": TEXT_EMBEDDING_DIM,
            "failed_site_count": len(rendered_failures),
            "failure_examples": rendered_failures[:30],
            "passes": not rendered_failures,
        },
        "live_accessibility_tree_contract": {
            "minimum_feature_version": 1,
            "failed_site_count": len(live_ax_failures),
            "failure_examples": live_ax_failures[:30],
            "passes": not live_ax_failures,
        },
        "dissertation_gates": {
            "three_views_aligned": len(aligned) == len(all_sites),
            "rendered_visual_cues_versioned": not rendered_failures,
            "live_browser_accessibility_tree_proven": not live_ax_failures,
            "same_session_ax_sidecar_proven": not live_ax_failures,
        },
        "limitations": [
            "A final a11y-tree cache must carry the same_session_sidecar provenance marker from the crawler's 0.ax.json file.",
            "A rendered-visual filename alone is not evidence of visual cues; feature version, width, and matching masks must pass.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=("dom", "a11y-tree", "rendered-visual"), default=("dom", "a11y-tree", "rendered-visual"))
    return parser.parse_args()


def main() -> None:
    args = parse_args(); report = audit_cache(args.cache_dir, tuple(args.views))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "cache_dir": str(args.cache_dir.resolve()),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"aligned_sites": report["aligned_site_count"], "gates": report["dissertation_gates"]}, indent=2))


if __name__ == "__main__":
    main()
