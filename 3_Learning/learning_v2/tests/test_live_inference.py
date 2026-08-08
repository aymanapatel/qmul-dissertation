from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from learning_v2 import live_inference


def test_live_runner_builds_two_graphs_once_and_runs_six_specialists(tmp_path, monkeypatch):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    for name in ("0.html", "0.visual.json", "0.ax.json", "page-0_home.json"):
        (site_dir / name).write_text("<html></html>" if name.endswith(".html") else "{}")

    phase5 = tmp_path / "phase5"
    for architecture in live_inference.DEFAULT_ARCHITECTURES:
        for view in ("a11y-tree", "rendered-visual"):
            run_dir = phase5 / view / architecture
            run_dir.mkdir(parents=True)
            for name in ("best_model.pt", "calibration.json", "manifest.json"):
                (run_dir / name).write_text("{}")

    graph = SimpleNamespace(num_nodes=4, edge_index=torch.tensor([[0, 1], [1, 2]]))
    build_count = {"a11y": 0, "rendered": 0}

    def fake_build_site(site, destination, text_model):
        build_count["a11y"] += 1
        torch.save({"data": graph}, destination)

    class RenderedPage:
        data = graph
        node_map = {}

        def save(self, destination):
            destination.write_bytes(b"graph")

    class Extractor:
        text_model = object()

        def process_page(self, **kwargs):
            build_count["rendered"] += 1
            return RenderedPage()

    calls = []

    def fake_score(**kwargs):
        calls.append((kwargs["architecture"], kwargs["view"]))
        finding = {
            "rule_id": "image-alt", "wcag_ids": ["1.1.1"],
            "probability": 0.9, "threshold": 0.7,
            "architecture": kwargs["architecture"], "graph_view": kwargs["view"],
            "detector_id": f'{kwargs["view"]}:{kwargs["architecture"]}:image-alt',
            "evidence": {"selector": "img"},
        }
        return {
            "view": kwargs["view"], "architecture": kwargs["architecture"],
            "node_count": 4, "edge_count": 2, "checkpoint_sha256": "a" * 64,
            "axe_used_for_prediction": False, "feature_contract": {},
            "rules": [], "findings": [finding],
        }

    def fake_route(predictions, policy):
        return [{**item, "criterion_id": "1.1.1", "routing_status": "fail", "routing_confidence": 0.8} for item in predictions]

    monkeypatch.setattr(live_inference, "build_site", fake_build_site)
    monkeypatch.setattr(live_inference, "_extractor", lambda device: Extractor())
    monkeypatch.setattr(live_inference, "_score_view", fake_score)
    monkeypatch.setattr(live_inference, "_fuse_predictions", fake_route)
    events = []

    report = live_inference.run_live_specialists(
        site_dir, tmp_path / "graphs", phase5_dir=phase5,
        fusion_policy_path=tmp_path / "policy.json",
        progress=lambda event_id, status, label, details: events.append((event_id, status)),
    )

    assert build_count == {"a11y": 1, "rendered": 1}
    assert len(calls) == 6
    assert set(calls) == {
        (architecture, view)
        for architecture in ("mlp", "graphsage", "gat")
        for view in ("a11y-tree", "rendered-visual")
    }
    assert report["architectures"] == ["mlp", "graphsage", "gat"]
    assert len(report["model_runs"]) == 6
    assert len(report["findings"]) == 6
    assert ("route_findings", "completed") in events


def test_live_runner_fails_before_inference_when_any_checkpoint_is_missing(tmp_path):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    for name in ("0.html", "0.visual.json", "0.ax.json", "page-0_home.json"):
        (site_dir / name).write_text("{}")

    try:
        live_inference.run_live_specialists(site_dir, tmp_path / "graphs", phase5_dir=tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "Frozen specialist artifacts are incomplete" in str(exc)
        assert "a11y-tree/mlp/best_model.pt" in str(exc)
    else:
        raise AssertionError("Expected missing frozen artifacts to fail the audit")
