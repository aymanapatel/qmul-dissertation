import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from learning_v2.baselines import deterministic_findings
from learning_v2.evidence import collect_static_evidence
from learning_v2.governance import create_grouped_multilabel_split, inventory_corpus
from learning_v2.inference import predict_graph
from learning_v2.losses import sampled_binary_cross_entropy
from learning_v2.schema import FeatureContract


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_controlled_fixture_matches_deterministic_rule_multiset():
    gold = json.loads((FIXTURES / "mixed_issues.gold.json").read_text())
    predicted = deterministic_findings(FIXTURES / "mixed_issues.html", site_id="fixture")
    assert sorted(f.rule_id for f in predicted) == sorted(f["rule_id"] for f in gold["findings"])


def _write_site(root: Path, site: str, html: str, rules: list[str]):
    directory = root / site; directory.mkdir()
    (directory / "0.html").write_text(html)
    (directory / "page-0_home.json").write_text(json.dumps({
        "url": f"https://{site}",
        "violations": [{"id": rule, "nodes": [{"target": ["html"]}]} for rule in rules],
    }))


def test_governed_split_keeps_duplicate_html_together(tmp_path):
    for index in range(12):
        html = "<html><title>same</title></html>" if index < 2 else f"<html><title>{index}</title></html>"
        _write_site(tmp_path, f"site-{index}", html, ["html-has-lang"] if index % 2 else [])
    inventory = inventory_corpus(tmp_path)
    first = create_grouped_multilabel_split(inventory, seed=3)
    second = create_grouped_multilabel_split(inventory, seed=3)
    assert first == second
    locations = {site: name for name in ("train", "val", "test") for site in first[name]}
    assert locations["site-0"] == locations["site-1"]


def test_static_evidence_is_stable_and_separates_axe(tmp_path):
    _write_site(tmp_path, "site", "<html lang='en'><body><button>Save</button></body></html>", [])
    first = collect_static_evidence(tmp_path / "site")
    second = collect_static_evidence(tmp_path / "site")
    assert first.html_sha256 == second.html_sha256
    assert [node.identity.css_path for node in first.nodes] == [node.identity.css_path for node in second.nodes]
    assert not hasattr(first.nodes[0], "axe_label")


def test_invalid_visual_nodes_are_excluded_from_loss():
    logits = torch.zeros((2, 1), requires_grad=True)
    targets = torch.tensor([[1.0], [0.0]])
    loss, counts = sampled_binary_cross_entropy(logits, targets, minimum_negatives=1, valid_mask=torch.tensor([False, True]))
    assert torch.isfinite(loss)
    assert counts == {"positive_pairs": 0, "negative_pairs": 1}


class AllPositive(torch.nn.Module):
    def forward(self, x, edge_index, tag_indices):
        result = torch.full((x.shape[0], 5), -10.0)
        result[:, 1] = 10.0
        return result


def test_display_top_k_does_not_truncate_evaluation_predictions():
    graph = Data(
        x=torch.randn(4, 7), edge_index=torch.tensor([[0, 1], [1, 2]]),
        tag_indices=torch.tensor([1, 1, 1, 1]), rendered_visible_mask=torch.ones(4, dtype=torch.bool),
    )
    graph.graph_source = "rendered-visual"; graph.rendered_visual_feature_version = 2
    nodes = {index: SimpleNamespace(tag="p", text_content="text", attrs={}, dom_path=str(index)) for index in range(4)}
    rules = ["avoid-inline-spacing", "color-contrast", "link-in-text-block", "meta-viewport", "scrollable-region-focusable"]
    report = predict_graph(
        model=AllPositive(), graph=graph, node_map=nodes, checkpoint={"rule_ids": rules},
        calibration={"recommended": {"node_threshold": 0.5, "page_threshold": 0.5, "rule_thresholds": {rule: 0.5 for rule in rules}}},
        contract=FeatureContract.from_data(graph, "rendered-visual"), device="cpu", top_k=1,
    )
    assert report["prediction_count"] == 4
    assert len(report["predictions"]) == 4
    assert len(report["top_candidates"]) == 1

