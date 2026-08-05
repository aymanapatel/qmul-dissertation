import json

import torch

from learning_v2.build_same_session_ax_cache import SAME_SESSION_AX_FEATURE_VERSION, build_site, mapping_coverage
from wcag_rules import RULE_INDEX


class _TextModel:
    def encode(self, names, **kwargs):
        return torch.zeros((len(names), 384), dtype=torch.float32)


def test_builds_from_saved_same_session_ax_sidecar(tmp_path):
    site = tmp_path / "example.test"
    site.mkdir()
    (site / "0.html").write_text(
        '<html data-gnn-node-id="0"><body data-gnn-node-id="1"><img id="logo" data-gnn-node-id="2"></body></html>',
        encoding="utf-8",
    )
    (site / "0.visual.json").write_text(json.dumps({"nodes": [
        {"snapshot_node_id": "0", "x": 0, "y": 0, "width": 100, "height": 100, "visible": True},
        {"snapshot_node_id": "1", "x": 0, "y": 0, "width": 100, "height": 100, "visible": True},
        {"snapshot_node_id": "2", "x": 1, "y": 1, "width": 10, "height": 10, "visible": True},
    ]}), encoding="utf-8")
    (site / "0.ax.json").write_text(json.dumps({
        "nodes": [
            {"nodeId": "1", "backendDOMNodeId": 10, "role": {"value": "RootWebArea"}, "name": {"value": ""}, "childIds": ["2"]},
            {"nodeId": "2", "backendDOMNodeId": 12, "role": {"value": "image"}, "name": {"value": ""}, "childIds": []},
        ],
        "backend_dom_to_snapshot_node": {"10": "0", "12": "2"},
        "mapping_stats": {"ax_nodes": 2, "ax_nodes_mapped_to_snapshot": 2},
    }), encoding="utf-8")
    (site / "page-0_home.json").write_text(json.dumps({"violations": [{
        "id": "image-alt", "nodes": [{"target": ["#logo"]}],
    }]}), encoding="utf-8")

    output = tmp_path / "graphs" / "example.test" / "a11y-tree.pt"
    record = build_site(site, output, _TextModel())
    data = torch.load(output, map_location="cpu", weights_only=False)["data"]

    assert record["status"] == "captured"
    assert data.ax_capture_provenance == "same_session_sidecar"
    assert data.live_ax_feature_version == SAME_SESSION_AX_FEATURE_VERSION
    assert data.num_nodes == 2
    assert data.node_y_multi[:, RULE_INDEX["image-alt"]].sum().item() == 1


def test_reads_mapping_coverage_from_sidecar(tmp_path):
    path = tmp_path / "0.ax.json"
    path.write_text(json.dumps({"mapping_stats": {"ax_nodes": 100, "ax_nodes_mapped_to_snapshot": 9}}), encoding="utf-8")
    ratio, stats = mapping_coverage(path)
    assert ratio == 0.09
    assert stats == {"ax_nodes": 100, "ax_nodes_mapped_to_snapshot": 9}
