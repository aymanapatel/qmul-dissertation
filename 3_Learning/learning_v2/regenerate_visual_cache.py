"""Regenerate versioned rendered-visual graphs for one frozen governed split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .collection_selection import select_multilabel_stratified_sites, select_split_sites
from .data import split_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_extractor import FeatureExtractor, ProcessedPage, RENDERED_VISUAL_FEATURE_VERSION  # noqa: E402
from graph_sources import GRAPH_SOURCE_RENDERED_VISUAL  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    selection_rules = set(args.selection_rules or [])
    positive_sites_by_rule = {rule: set() for rule in selection_rules}
    if selection_rules:
        for partition in ("train", "val", "test"):
            for site in split[partition]:
                report_path = args.corpus_dir / site / "page-0_home.json"
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    found = {str(item.get("id")) for item in report.get("violations", [])}
                    for rule in selection_rules & found:
                        positive_sites_by_rule[rule].add(site)
    ordered = select_multilabel_stratified_sites(
        split, positive_sites_by_rule, args.max_sites, args.positive_fraction, args.minimum_positive_sites_per_rule,
    ) if selection_rules else select_split_sites(split, args.max_sites)
    selected_split = {
        "seed": split.get("seed", 42),
        **{partition: [site for name, site in ordered if name == partition] for partition in ("train", "val", "test")},
    }
    selected_split["split_hash"] = split_hash(selected_split)
    extractor = FeatureExtractor(device=args.device)
    old_manifest_path = args.output_dir / "rendered_visual_cache_manifest.json"
    old_records = {}
    if old_manifest_path.is_file():
        old_records = {item["site_id"]: item for item in json.loads(old_manifest_path.read_text(encoding="utf-8")).get("records", [])}
    records = []
    for index, (partition, site_id) in enumerate(ordered, 1):
        site_dir = args.corpus_dir / site_id
        html_path = site_dir / "0.html"; axe_path = site_dir / "page-0_home.json"
        output = args.output_dir / site_id / "rendered-visual.pt"
        print(f"[{index}/{len(ordered)}] {partition} {site_id}", flush=True)
        if not html_path.is_file() or not axe_path.is_file():
            records.append({"site_id": site_id, "partition": partition, "status": "missing_source"})
            continue
        if args.resume and output.is_file():
            try:
                cached = ProcessedPage.load(output).data
                if int(getattr(cached, "rendered_visual_feature_version", 0)) == RENDERED_VISUAL_FEATURE_VERSION and hasattr(cached, "edge_type"):
                    record = dict(old_records.get(site_id, {}))
                    record.update({"site_id": site_id, "partition": partition, "status": "reused", "path": str(output), "nodes": int(cached.num_nodes), "edges": int(cached.edge_index.shape[1]), "feature_dim": int(cached.x.shape[1])})
                    records.append(record)
                    continue
            except Exception:
                pass
        try:
            page = extractor.process_page(html_path=html_path, axe_report_path=axe_path, extract_visual=True, graph_source=GRAPH_SOURCE_RENDERED_VISUAL)
            data = page.data
            if int(getattr(data, "rendered_visual_feature_version", 0)) != RENDERED_VISUAL_FEATURE_VERSION:
                raise ValueError("collector emitted wrong rendered feature version")
            if not hasattr(data, "edge_type"):
                raise ValueError("collector emitted untyped edges")
            output.parent.mkdir(parents=True, exist_ok=True); page.save(output)
            records.append({
                "site_id": site_id, "partition": partition, "status": "captured", "path": str(output),
                "source_html_sha256": _sha256(html_path), "source_axe_sha256": _sha256(axe_path),
                "nodes": int(data.num_nodes), "edges": int(data.edge_index.shape[1]), "feature_dim": int(data.x.shape[1]),
                "feature_version": int(data.rendered_visual_feature_version), "visible_nodes": int(data.rendered_visible_mask.sum()),
                "visual_matched_nodes": int(data.visual_match_found_mask.sum()), "spatial_edges": int((data.edge_type == 2).sum()),
            })
        except Exception as exc:
            records.append({"site_id": site_id, "partition": partition, "status": "collection_failed", "error": f"{type(exc).__name__}: {exc}"[:1000]})
    counts = {status: sum(item["status"] == status for item in records) for status in sorted({item["status"] for item in records})}
    manifest = {
        "schema_version": 1, "collector": "saved_html_playwright_rendered_visual_v2",
        "split": str(args.split.resolve()), "split_sha256": _sha256(args.split), "split_hash": split.get("split_hash"),
        "corpus_dir": str(args.corpus_dir.resolve()), "feature_version": RENDERED_VISUAL_FEATURE_VERSION,
        "requested_sites": len(ordered), "outcome_counts": counts, "records": records,
        "selection": {
            "rules": sorted(selection_rules), "positive_fraction": args.positive_fraction,
            "minimum_positive_sites_per_rule": args.minimum_positive_sites_per_rule,
            "available_site_support": {rule: len(sites) for rule, sites in sorted(positive_sites_by_rule.items())},
            "selected_site_support": {rule: sum(site in sites for _, site in ordered) for rule, sites in sorted(positive_sites_by_rule.items())},
            "selected_partition_support": {
                partition: {
                    rule: sum(name == partition and site in sites for name, site in ordered)
                    for rule, sites in sorted(positive_sites_by_rule.items())
                }
                for partition in ("train", "val", "test")
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "collection_split.json").write_text(json.dumps(selected_split, indent=2), encoding="utf-8")
    (args.output_dir / "rendered_visual_cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True); parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-sites", type=int); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-rules", nargs="+", help="Predeclared axe rules used only to ensure positive/negative pilot support")
    parser.add_argument("--positive-fraction", type=float, default=0.6)
    parser.add_argument("--minimum-positive-sites-per-rule", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args()); print(json.dumps({"requested_sites": report["requested_sites"], "outcomes": report["outcome_counts"]}, indent=2))


if __name__ == "__main__":
    main()
