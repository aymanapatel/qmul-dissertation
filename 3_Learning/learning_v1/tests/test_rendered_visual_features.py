import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from feature_extractor import FeatureExtractor  # noqa: E402
from graph_sources import build_graph  # noqa: E402


def _extract(html_path: Path):
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    result = build_graph(html_path, "rendered-visual")
    try:
        extractor.extract_visual_features(html_path, result.node_map)
    except Exception as exc:
        pytest.skip(f"Playwright visual extraction unavailable: {exc}")
    return result.node_map


def _node_by_id(node_map, element_id: str):
    for node in node_map.values():
        if node.attrs.get("id") == element_id:
            return node
    raise AssertionError(f"node id {element_id} not found")


def test_visual_features_capture_contrast_and_transparent_background(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        """
        <html>
          <body style="background: white">
            <p id="high" style="color: black">Readable</p>
            <div style="background: rgb(240, 240, 240)">
              <p id="low" style="color: rgb(245, 245, 245); background: transparent">Faint</p>
            </div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    node_map = _extract(html)
    high = _node_by_id(node_map, "high")
    low = _node_by_id(node_map, "low")

    assert high.visual["visual_match_found"] is True
    assert low.visual["visual_match_found"] is True
    assert high.visual["contrast_ratio"] > 10
    assert low.visual["contrast_ratio"] < 2
    assert low.visual["background_rgb"] == [240, 240, 240]
    assert low.visual["required_contrast_ratio"] == 4.5
    assert low.visual["contrast_deficit"] > 0.5
    assert low.visual["has_direct_text"] is True


def test_visual_features_map_duplicate_tags_by_stable_node_id(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        """
        <html><body>
          <span class="pill" id="first" style="color: black">First</span>
          <span class="pill" id="second" style="color: white; background: black">Second</span>
        </body></html>
        """,
        encoding="utf-8",
    )

    node_map = _extract(html)
    first = _node_by_id(node_map, "first")
    second = _node_by_id(node_map, "second")

    assert first.visual["visual_match_found"] is True
    assert second.visual["visual_match_found"] is True
    assert first.text_content == "First"
    assert second.text_content == "Second"
    assert first.visual["foreground_rgb"] != second.visual["foreground_rgb"]


def test_visual_features_preserve_document_stylesheets(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        """
        <html>
          <head>
            <style>
              body { background: rgb(250, 250, 250); }
              .faint { color: rgb(240, 240, 240); }
            </style>
          </head>
          <body><p id="styled" class="faint">Styled text</p></body>
        </html>
        """,
        encoding="utf-8",
    )

    node_map = _extract(html)
    styled = _node_by_id(node_map, "styled")

    assert styled.visual["foreground_rgb"] == [240, 240, 240]
    assert styled.visual["background_rgb"] == [250, 250, 250]
    assert styled.visual["contrast_ratio"] < 1.2


def test_visual_features_resolve_relative_external_stylesheets(tmp_path):
    (tmp_path / "page.css").write_text(
        "body { background: rgb(32, 32, 32); } .copy { color: rgb(48, 48, 48); }",
        encoding="utf-8",
    )
    html = tmp_path / "page.html"
    html.write_text(
        """
        <html>
          <head><link rel="stylesheet" href="page.css"></head>
          <body><p id="external" class="copy">External CSS</p></body>
        </html>
        """,
        encoding="utf-8",
    )

    node_map = _extract(html)
    external = _node_by_id(node_map, "external")

    assert external.visual["foreground_rgb"] == [48, 48, 48]
    assert external.visual["background_rgb"] == [32, 32, 32]
    assert external.visual["contrast_ratio"] < 1.3


def test_visual_features_prefer_live_capture_sidecar(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        """
        <html data-gnn-node-id="0">
          <head data-gnn-node-id="1"></head>
          <body data-gnn-node-id="2">
            <p id="captured" data-gnn-node-id="3">Captured text</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    html.with_suffix(".visual.json").write_text(
        json.dumps(
            {
                "version": 1,
                "viewport": {"width": 1440, "height": 900},
                "nodes": [
                    {
                        "snapshot_node_id": "3",
                        "x": 144,
                        "y": 90,
                        "width": 720,
                        "height": 45,
                        "visible": True,
                        "foreground_rgb": [120, 120, 120],
                        "background_rgb": [255, 255, 255],
                        "contrast_ratio": 4.42,
                        "font_size": 16,
                        "font_weight": 400,
                        "opacity": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    node_map = _extract(html)
    captured = _node_by_id(node_map, "captured")

    assert captured.visual["visual_source"] == "captured-live"
    assert captured.visual["contrast_ratio"] == pytest.approx(4.42)
    assert captured.bbox == {"x": 144, "y": 90, "width": 720, "height": 45}
