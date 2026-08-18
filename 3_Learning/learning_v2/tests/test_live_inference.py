from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import torch
from bs4 import BeautifulSoup

from learning_v2 import live_inference


def _context(rule_id: str, html: str, selector: str, visual=None):
    soup = BeautifulSoup(html, "lxml")
    element = soup.select_one(selector)
    return live_inference._repair_context(
        rule_id=rule_id, element=element, soup=soup,
        selector=selector, visual=visual,
    )


def test_live_prompt_context_builds_bounded_candidates_for_all_four_rules():
    assert live_inference._normalise_selector(
        "[document]:nth-of-type(1) > html:nth-of-type(1) > body:nth-of-type(1) > a:nth-of-type(1)"
    ) == "html:nth-of-type(1) > body:nth-of-type(1) > a:nth-of-type(1)"
    link = _context(
        "link-name", "<html><head><title>Plans</title></head><body><a href='/pricing'></a></body></html>", "a",
    )
    assert link["nearby"]["document_title"] == "Plans"
    assert link["bounded_candidates"][0]["operation"] == {
        "operation": "set_attribute", "selector": "a", "attribute_name": "aria-label",
        "css_property": None, "new_value": "Pricing",
    }
    assert link["bounded_candidates"][0]["requires_human_review"] is True

    image = _context(
        "image-alt",
        "<html><body><figure><img src='mars-spaceman.jpg'><figcaption>A person on Mars</figcaption></figure></body></html>",
        "img",
    )
    caption = next(item for item in image["bounded_candidates"] if item["derived_from"] == "visible_figcaption")
    assert caption["operation"]["new_value"] == "A person on Mars"
    assert caption["operation"]["attribute_name"] == "alt"
    assert caption["requires_human_review"] is False

    label = _context(
        "label", "<html><body><input id='email' placeholder='Email address'></body></html>", "input",
    )
    assert label["bounded_candidates"][0]["operation"] == {
        "operation": "insert_label_before", "selector": "input", "attribute_name": None,
        "css_property": None, "new_value": "Email address",
    }

    contrast = _context(
        "color-contrast", "<html><body><span>Low contrast</span></body></html>", "span",
        visual={
            "foreground_rgb": [119, 119, 119], "background_rgb": [255, 255, 255],
            "contrast_ratio": 4.478, "required_contrast_ratio": 4.5,
        },
    )
    assert contrast["current_state"]["meets_requirement"] is False
    assert contrast["bounded_candidates"][0]["operation"] == {
        "operation": "set_style_property", "selector": "span", "attribute_name": None,
        "css_property": "color", "new_value": "#000000",
    }
    assert contrast["bounded_candidates"][0]["computed_contrast_ratio"] == 21.0
    assert contrast["bounded_candidates"][0]["requires_human_review"] is False


def test_contrast_context_suppresses_repair_when_observed_ratio_passes():
    context = _context(
        "color-contrast", "<html><body><span>Readable</span></body></html>", "span",
        visual={
            "foreground_rgb": [255, 255, 255], "background_rgb": [0, 90, 106],
            "contrast_ratio": 7.86, "required_contrast_ratio": 4.5,
        },
    )
    assert context["current_state"]["meets_requirement"] is True
    assert context["bounded_candidates"] == []


def test_image_context_uses_adjacent_text_and_never_a_filename_hash():
    context = _context(
        "image-alt",
        "<html><body><section><p>Quarterly sales by region</p>"
        "<img src='b2c4d6e8f0a1b3c5d7e9f1a2b4c6d8e0.svg'></section></body></html>",
        "img",
    )
    assert [item["derived_from"] for item in context["bounded_candidates"]] == [
        "previous_sibling_visible_text"
    ]
    assert context["bounded_candidates"][0]["operation"]["new_value"] == "Quarterly sales by region"
    assert context["bounded_candidates"][0]["requires_human_review"] is False


def test_image_context_with_only_asset_hashes_retains_nearby_for_automatic_mode():
    context = _context(
        "image-alt",
        "<html><body><section><h2>Image fixtures</h2>"
        "<img src='7a3f9e2b1c8d4e5f6a0b9c8d7e1f2a3b.svg'>"
        "<img src='b2c4d6e8f0a1b3c5d7e9f1a2b4c6d8e0.svg'>"
        "</section></body></html>",
        "img:nth-of-type(2)",
    )
    assert context["nearby"]["nearest_heading"]["text"] == "Image fixtures"
    assert context["bounded_candidates"] == [{
        "operation": {
            "operation": "set_attribute",
            "selector": "img:nth-of-type(2)",
            "attribute_name": "alt",
            "css_property": None,
            "new_value": "Image fixtures illustration 2",
        },
        "derived_from": "nearest_heading_and_image_position",
        "source_value": {"nearest_heading": "Image fixtures", "image_position": 2},
        "verification_level": "contextual_inference",
        "requires_human_review": False,
    }]


def test_image_context_uses_previous_sibling_accessible_name_and_local_prose():
    sibling_context = _context(
        "image-alt",
        "<html><body><img src='first.svg' alt='Service status overview'>"
        "<img src='second.svg'></body></html>",
        "img:nth-of-type(2)",
    )
    sibling = next(
        item for item in sibling_context["bounded_candidates"]
        if item["derived_from"] == "previous_sibling_alt"
    )
    assert sibling["operation"]["new_value"] == "Service status overview"
    assert sibling["requires_human_review"] is False

    container_context = _context(
        "image-alt",
        "<html><body><section><p>Quarterly sales by region</p><button>Expand</button>"
        "<img src='chart.svg'></section></body></html>",
        "img",
    )
    local = next(
        item for item in container_context["bounded_candidates"]
        if item["derived_from"] == "immediate_container_visible_text"
    )
    assert local["operation"]["new_value"] == "Quarterly sales by region"


def test_visual_evidence_payload_serialises_bounds_and_contrast_failures(tmp_path):
    (tmp_path / "0.html").write_text(
        '<html data-gnn-node-id="0"><body data-gnn-node-id="1">'
        '<p data-gnn-node-id="2">Low contrast</p></body></html>',
        encoding="utf-8",
    )
    (tmp_path / "0.visual.json").write_text(json.dumps({
        "viewport": {"width": 800, "height": 600},
        "nodes": [{
            "snapshot_node_id": "2", "x": 20, "y": 30, "width": 200, "height": 40,
            "visible": True, "in_viewport": True, "clipped": False,
            "foreground_rgb": [200, 200, 200], "background_rgb": [255, 255, 255],
            "contrast_ratio": 1.67, "required_contrast_ratio": 4.5,
            "contrast_deficit": 0.62, "has_direct_text": True,
            "font_size": 16, "font_weight": 400, "opacity": 1,
        }],
    }), encoding="utf-8")
    payload = live_inference._visual_evidence_payload(tmp_path)
    assert payload["element_count"] == 1
    assert payload["contrast_failure_count"] == 1
    failure = payload["contrast_failures"][0]
    assert failure["selector"] == "html:nth-of-type(1) > body:nth-of-type(1) > p:nth-of-type(1)"
    assert failure["bounds"] == {"x": 20.0, "y": 30.0, "width": 200.0, "height": 40.0}
    assert failure["visual"]["contrast_ratio"] == 1.67
    assert failure["source"] == "same-session-rendered-visual-capture"
    assert failure["repair_context"]["bounded_candidates"][0]["operation"] == {
        "operation": "set_style_property",
        "selector": "html:nth-of-type(1) > body:nth-of-type(1) > p:nth-of-type(1)",
        "attribute_name": None, "css_property": "color", "new_value": "#000000",
    }


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
