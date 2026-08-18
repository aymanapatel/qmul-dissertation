from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest
from torch_geometric.data import Data

from learning_v2.cache_audit import audit_cache
from learning_v2.data import sanitise_graph
from learning_v2.collection_selection import select_multilabel_stratified_sites, select_split_sites, select_stratified_sites
from learning_v2.visual_ablation import _percentile_interval
from learning_v2.corrected_pilot_report import build_report, render_html
from learning_v2.annotation_packet import _case_id, _cohen_kappa, _criterion_records, finalize_packet
from learning_v2.assemble_multiview_bundle import _mapping
from learning_v2.visual_final_study import hierarchical_paired_interval
from learning_v2.study import MethodOutput


def _raw(view: str, width: int, version: int = 0) -> Data:
    data = Data(
        x=torch.ones((3, width)), edge_index=torch.tensor([[0, 1, 0], [1, 2, 2]]),
        edge_type=torch.tensor([0, 1, 2]), tag_indices=torch.tensor([1, 2, 3]),
        node_y_multi=torch.zeros((3, 46)), rendered_visible_mask=torch.ones(3, dtype=torch.bool),
    )
    data.graph_source = view; data.rendered_visual_feature_version = version
    if view == "rendered-visual":
        data.visual_match_found_mask = torch.ones(3, dtype=torch.bool)
    return data


def test_structure_only_ablation_zeroes_visual_tail_and_removes_spatial_edges():
    raw = _raw("rendered-visual", 517, version=2)
    graph = sanitise_graph(raw, graph_source="rendered-visual", rule_indices=[0], require_labels=True, feature_variant="structure_only")
    assert graph.x[:, 113:133].count_nonzero() == 0
    assert graph.edge_type.tolist() == [0, 1]
    assert graph.edge_index.shape[1] == 2
    assert graph.feature_variant == "structure_only"


def test_cache_audit_rejects_a_rendered_filename_without_visual_cues(tmp_path):
    for view in ("dom", "a11y-tree", "rendered-visual"):
        site = tmp_path / "site"; site.mkdir(exist_ok=True)
        torch.save({"data": _raw(view, 497), "num_nodes": 3}, site / f"{view}.pt")
    report = audit_cache(tmp_path)
    assert report["aligned_site_count"] == 1
    assert report["rendered_visual_contract"]["passes"] is False
    assert report["rendered_visual_contract"]["failed_site_count"] == 1


def test_bounded_collection_preserves_all_frozen_partitions():
    split = {"train": [f"tr-{i}" for i in range(70)], "val": [f"va-{i}" for i in range(15)], "test": [f"te-{i}" for i in range(15)]}
    selected = select_split_sites(split, 20)
    counts = {name: sum(partition == name for partition, _ in selected) for name in ("train", "val", "test")}
    assert counts == {"train": 14, "val": 3, "test": 3}
    positives = {f"{prefix}-{i}" for prefix, count in (("tr", 20), ("va", 10), ("te", 10)) for i in range(count)}
    stratified = select_stratified_sites(split, positives, 20, 0.5)
    for partition, prefix in (("train", "tr"), ("val", "va"), ("test", "te")):
        chosen = [site for name, site in stratified if name == partition]
        assert sum(site in positives for site in chosen) >= len(chosen) // 2


def test_multilabel_selection_covers_each_rule_in_each_partition():
    split = {
        "train": [f"tr-{i}" for i in range(12)],
        "val": [f"va-{i}" for i in range(6)],
        "test": [f"te-{i}" for i in range(6)],
    }
    support = {
        "image-alt": {"tr-8", "tr-9", "va-4", "va-5", "te-4", "te-5"},
        "link-name": {"tr-7", "tr-9", "va-3", "va-5", "te-3", "te-5"},
    }
    selected = select_multilabel_stratified_sites(split, support, 18, 0.5, minimum_per_rule=2)
    for partition in ("train", "val", "test"):
        chosen = {site for name, site in selected if name == partition}
        for sites in support.values():
            assert len(chosen & sites) >= 2


def test_seed_bootstrap_interval_is_reproducible():
    assert _percentile_interval([0.1, 0.2, -0.1], 100, 7) == _percentile_interval([0.1, 0.2, -0.1], 100, 7)


def test_corrected_pilot_report_never_promotes_weak_labels_to_final():
    support = {name: {rule: 2 for rule in ("image-alt", "label")} for name in ("train", "val", "test")}
    visual_record = {"feature_version": 2, "feature_dim": 517, "spatial_edges": 1}
    live_record = {"feature_version": 1, "feature_dim": 509}
    visual = {"feature_version": 2, "records": [visual_record] * 50, "selection": {"minimum_positive_sites_per_rule": 2, "selected_partition_support": support}}
    live = {"live_ax_feature_version": 1, "records": [live_record] * 50}
    split = {"train": ["a"], "val": ["b"], "test": ["c"], "split_hash": "frozen"}
    comparison = {"results": [{"architecture": "mlp", "rules": [], "best_epoch": 1, "test": {}}]}
    report, _ = build_report(visual, live, split, split, comparison, comparison, {"paired_comparisons": {"mlp": {"full_minus_structure_only": {}}}})
    assert report["completion_gates"]["versioned_rendered_visual_contract"] is True
    assert report["completion_gates"]["live_chromium_accessibility_tree_contract"] is True
    assert report["dissertation_ready"] is False


def test_corrected_pilot_html_is_standalone_and_escapes_values():
    rendered = render_html({
        "dissertation_ready": False,
        "split": {"hash": "<unsafe>", "site_counts": {"test": 1}},
        "held_out_detection_study": {"methods": [], "graphsage_minus_mlp_f1_95_ci": [0, 0]},
        "controlled_repair_study": {"conditions": []},
        "completion_gates": {"independent_truth": False},
        "remaining_required_work": ["Annotate & adjudicate"],
    })
    assert rendered.startswith("<!doctype html>")
    assert "&lt;unsafe&gt;" in rendered
    assert "Annotate &amp; adjudicate" in rendered
    assert "<unsafe>" not in rendered


def test_annotation_packet_ids_are_stable_and_scope_comes_from_rules(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text('{"criteria":{"1.1.1":{"name":"Non-text Content"}}}', encoding="utf-8")
    assert _case_id("site", "1.1.1", 42) == _case_id("site", "1.1.1", 42)
    assert set(_criterion_records(registry, ["image-alt"])) == {"1.1.1"}
    assert _cohen_kappa(["pass", "fail"], ["pass", "fail"]) == 1.0


def test_detection_annotation_finalizer_emits_complete_truth(tmp_path):
    (tmp_path / "coordinator").mkdir(); (tmp_path / "rater_packets").mkdir()
    identity = {"case_id": "c1", "site_id": "site", "criterion_id": "1.1.1"}
    (tmp_path / "coordinator/identity_map.json").write_text(json.dumps({"cases": [identity]}), encoding="utf-8")
    (tmp_path / "coordinator/independent_detection_truth.json").write_text(json.dumps({"pairs": [{**identity, "status": None, "adjudicated": False, "evidence": ""}]}), encoding="utf-8")
    for index in (1, 2):
        (tmp_path / f"rater_packets/rater_{index}.json").write_text(json.dumps({"cases": [{"case_id": "c1", "status": "fail", "confidence": 5, "evidence_notes": "Missing text alternative"}]}), encoding="utf-8")
    output = tmp_path / "final.json"
    result = finalize_packet(tmp_path, output)
    assert result["agreement"]["raw_agreement"] == 1.0
    assert result["pairs"][0]["status"] == "fail"
    assert output.is_file()


def test_detection_annotation_finalizer_accepts_completed_top_level_rater_sheets(tmp_path):
    (tmp_path / "coordinator").mkdir(); (tmp_path / "rater_packets").mkdir()
    identity = {"case_id": "c1", "site_id": "site", "criterion_id": "1.1.1"}
    (tmp_path / "coordinator/identity_map.json").write_text(
        json.dumps({"cases": [identity]}), encoding="utf-8",
    )
    (tmp_path / "coordinator/independent_detection_truth.json").write_text(
        json.dumps({"pairs": [{**identity, "status": None, "adjudicated": False, "evidence": ""}]}),
        encoding="utf-8",
    )
    row = {"case_id": "c1", "status": "pass", "confidence": 4, "evidence_notes": "Named image."}
    for index in (1, 2):
        (tmp_path / f"rater_packets/rater_{index}.json").write_text(json.dumps([row]), encoding="utf-8")
    result = finalize_packet(tmp_path, tmp_path / "final.json")
    assert result["pairs"][0]["status"] == "pass"


def test_multiview_bundle_mapping_rejects_duplicate_views():
    assert _mapping(["a11y-tree=/one", "rendered-visual=/two"])["a11y-tree"] == Path("/one")
    with pytest.raises(ValueError, match="unique"):
        _mapping(["a11y-tree=/one", "a11y-tree=/two"])


def test_hierarchical_visual_interval_is_reproducible():
    universe = {("a", "1.4.3"), ("b", "1.4.3")}; truth = {("a", "1.4.3")}
    full = [MethodOutput({}, {("a", "1.4.3")}, universe)]
    structure = [MethodOutput({}, set(), universe)]
    left = hierarchical_paired_interval(full, structure, truth, universe, metric="recall", samples=50, seed=7)
    right = hierarchical_paired_interval(full, structure, truth, universe, metric="recall", samples=50, seed=7)
    assert left == right
