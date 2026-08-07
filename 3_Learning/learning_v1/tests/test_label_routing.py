import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from feature_extractor import FeatureExtractor  # noqa: E402
from graph_sources import build_graph  # noqa: E402


def write_axe_report(path: Path, rule_id: str, target: str) -> None:
    path.write_text(
        json.dumps(
            {
                "violations": [
                    {
                        "id": rule_id,
                        "impact": "serious",
                        "nodes": [{"target": [target]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_label_routing_by_graph_source(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        """
        <html><body>
          <img id="hero" src="hero.png">
          <ul id="bad-list"><div>bad child</div></ul>
          <p id="low-contrast">Low contrast</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    extractor = FeatureExtractor.__new__(FeatureExtractor)

    image_report = tmp_path / "image.json"
    write_axe_report(image_report, "image-alt", "#hero")
    a11y_nodes = build_graph(html, "a11y-tree").node_map
    dom_nodes = build_graph(html, "dom").node_map
    a11y_y, _ = extractor.load_axe_labels(image_report, a11y_nodes, graph_source="a11y-tree")
    dom_y, _ = extractor.load_axe_labels(image_report, dom_nodes, graph_source="dom")
    assert int(a11y_y.sum().item()) == 1
    assert int(dom_y.sum().item()) == 0

    list_report = tmp_path / "list.json"
    write_axe_report(list_report, "list", "#bad-list")
    a11y_y, _ = extractor.load_axe_labels(list_report, a11y_nodes, graph_source="a11y-tree")
    dom_y, _ = extractor.load_axe_labels(list_report, dom_nodes, graph_source="dom")
    assert int(a11y_y.sum().item()) == 0
    assert int(dom_y.sum().item()) == 1

    contrast_report = tmp_path / "contrast.json"
    write_axe_report(contrast_report, "color-contrast", "#low-contrast")
    visual_nodes = build_graph(html, "rendered-visual").node_map
    for node in visual_nodes.values():
        node.is_visible = True
    visual_y, _ = extractor.load_axe_labels(
        contrast_report,
        visual_nodes,
        graph_source="rendered-visual",
    )
    assert int(visual_y.sum().item()) == 1


def test_rendered_visual_label_kept_when_visibility_extraction_misses(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        """
        <html><body>
          <p id="low-contrast">Low contrast</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    contrast_report = tmp_path / "contrast.json"
    write_axe_report(contrast_report, "color-contrast", "#low-contrast")

    extractor = FeatureExtractor.__new__(FeatureExtractor)
    visual_nodes = build_graph(html, "rendered-visual").node_map
    for node in visual_nodes.values():
        node.is_visible = False

    visual_y, _ = extractor.load_axe_labels(
        contrast_report,
        visual_nodes,
        graph_source="rendered-visual",
    )

    labelled_nodes = [node for node in visual_nodes.values() if node.axe_violations]
    assert int(visual_y.sum().item()) == 1
    assert labelled_nodes
    assert labelled_nodes[0].visual_label_qa == ["rendered_label_on_nonvisible_or_unmatched_node"]
