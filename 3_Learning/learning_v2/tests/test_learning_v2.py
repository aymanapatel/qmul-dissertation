import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from learning_v2.calibration import calibrate_predictions
from learning_v2.data import (
    create_site_split,
    load_cached_graph,
    sanitise_graph,
    validate_split,
)
from learning_v2.inference import predict_graph
from learning_v2.losses import sampled_binary_cross_entropy
from learning_v2.metrics import PredictionArrays
from learning_v2.models import ModelConfig, build_model
from learning_v2.schema import FeatureContract, inference_fingerprint
from learning_v2.trainer import (
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    validation_loss,
)


VISUAL_RULE_INDICES = (17, 20, 31, 37, 41)


class FixedLogitModel(torch.nn.Module):
    def forward(self, x, edge_index, tag_indices):
        return x


def test_validation_loss_records_full_and_repeatable_training_matched_losses():
    logits = torch.tensor([[0.0, 1.0], [4.0, -4.0], [-1.0, 2.0], [0.5, -0.5]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    graph = Data(
        x=logits,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        tag_indices=torch.zeros(4, dtype=torch.long),
        rule_y=targets,
        label_mask=torch.tensor([True, False, True, True]),
    )

    config = TrainingConfig(negative_ratio=0.5, minimum_negatives=0)
    loader = DataLoader([graph], batch_size=1)
    result = validation_loss(FixedLogitModel(), loader, config, "cpu", sampling_seed=17)
    repeated = validation_loss(FixedLogitModel(), loader, config, "cpu", sampling_seed=17)
    allowed = graph.label_mask[:, None].expand_as(targets)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[allowed], targets[allowed], reduction="mean"
    )

    assert abs(result["loss"] - float(expected)) < 1e-7
    assert result["loss_type"] == "full_valid_pair_bce"
    assert result["valid_pairs"] == 6
    assert result["positive_pairs"] == 2
    assert result["negative_pairs"] == 4
    assert result["sampled_loss"] == repeated["sampled_loss"]
    assert result["sampled_loss_type"] == "fixed_sample_training_matched_bce"
    assert result["sampled_positive_pairs"] == 2
    assert result["sampled_negative_pairs"] == 1
    assert result["sampling_seed"] == 17


def raw_graph(*, positive: bool = True) -> Data:
    labels = torch.zeros(4, 46)
    if positive:
        labels[1, 20] = 1
    graph = Data(
        x=torch.randn(4, 7),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        tag_indices=torch.tensor([1, 2, 3, 4]),
        node_y_multi=labels,
    )
    graph.graph_source = "rendered-visual"
    graph.rendered_visual_feature_version = 2
    graph.rendered_visible_mask = torch.ones(4, dtype=torch.bool)
    return graph


def write_cache(root: Path, site: str, *, positive: bool) -> Path:
    path = root / site / "rendered-visual.pt"
    path.parent.mkdir(parents=True)
    torch.save({"data": raw_graph(positive=positive)}, path)
    return path


def test_sanitise_graph_keeps_features_independent_from_labels():
    raw = raw_graph()
    labelled = sanitise_graph(
        raw,
        graph_source="rendered-visual",
        rule_indices=VISUAL_RULE_INDICES,
        require_labels=True,
    )
    inference = sanitise_graph(
        raw,
        graph_source="rendered-visual",
        rule_indices=VISUAL_RULE_INDICES,
        require_labels=False,
    )

    for labelled_tensor, inference_tensor in zip(
        inference_fingerprint(labelled),
        inference_fingerprint(inference),
    ):
        assert torch.equal(labelled_tensor, inference_tensor)
    assert hasattr(labelled, "rule_y")
    assert not hasattr(inference, "rule_y")
    assert not hasattr(inference, "node_y")


def test_site_split_is_deterministic_and_disjoint(tmp_path):
    graph_paths = {
        f"site-{index}": write_cache(
            tmp_path,
            f"site-{index}",
            positive=index % 2 == 0,
        )
        for index in range(20)
    }
    first = create_site_split(
        graph_paths,
        rule_indices=VISUAL_RULE_INDICES,
        seed=7,
    )
    second = create_site_split(
        graph_paths,
        rule_indices=VISUAL_RULE_INDICES,
        seed=7,
    )

    assert first == second
    validate_split(first, available_sites=set(graph_paths))
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["val"]).isdisjoint(first["test"])


def test_sampled_loss_handles_positive_and_all_negative_batches():
    positive_targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    positive_loss, positive_counts = sampled_binary_cross_entropy(
        torch.zeros_like(positive_targets),
        positive_targets,
        negative_ratio=2,
        minimum_negatives=0,
    )
    negative_targets = torch.zeros(3, 2)
    negative_loss, negative_counts = sampled_binary_cross_entropy(
        torch.zeros_like(negative_targets),
        negative_targets,
        negative_ratio=2,
        minimum_negatives=2,
    )

    assert torch.isfinite(positive_loss)
    assert positive_counts == {"positive_pairs": 1, "negative_pairs": 2}
    assert torch.isfinite(negative_loss)
    assert negative_counts == {"positive_pairs": 0, "negative_pairs": 2}


def test_checkpoint_round_trip_is_deterministic(tmp_path):
    torch.manual_seed(4)
    config = ModelConfig(
        architecture="graphsage",
        feature_dim=7,
        num_tags=116,
        num_rules=5,
        hidden_dim=16,
        num_layers=2,
    )
    model = build_model(config)
    model.eval()
    graph = sanitise_graph(
        raw_graph(),
        graph_source="rendered-visual",
        rule_indices=VISUAL_RULE_INDICES,
        require_labels=False,
    )
    expected = model(graph.x, graph.edge_index, graph.tag_indices)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_metric=0.4,
        model_config=config,
        training_config=TrainingConfig(),
        feature_contract=FeatureContract.from_data(graph, "rendered-visual"),
        rule_ids=["a", "b", "c", "d", "e"],
        rule_indices=list(VISUAL_RULE_INDICES),
    )

    loaded, checkpoint = load_checkpoint(checkpoint_path)
    loaded.eval()
    actual = loaded(graph.x, graph.edge_index, graph.tag_indices)

    assert checkpoint["epoch"] == 3
    assert torch.allclose(expected, actual)


def test_calibration_returns_per_rule_thresholds():
    arrays = PredictionArrays(
        node_probs=torch.tensor([0.9, 0.7, 0.4, 0.1]),
        node_labels=torch.tensor([1, 1, 0, 0]),
        rule_probs=torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.6, 0.3], [0.1, 0.9]]
        ),
        rule_labels=torch.tensor(
            [[1, 0], [1, 0], [0, 0], [0, 1]], dtype=torch.bool
        ),
        page_probs=torch.tensor([0.9, 0.2]),
        page_labels=torch.tensor([1, 0]),
    )
    calibration = calibrate_predictions(
        arrays,
        ["color-contrast", "meta-viewport"],
        precision_floor=0.5,
    )

    assert set(calibration["recommended"]["rule_thresholds"]) == {
        "color-contrast",
        "meta-viewport",
    }
    assert 0 < calibration["recommended"]["node_threshold"] < 1


class FixedModel(torch.nn.Module):
    def forward(self, x, edge_index, tag_indices):
        logits = torch.full((x.shape[0], 5), -10.0)
        logits[1, 1] = 10.0
        return logits


def test_prediction_report_explicitly_excludes_axe():
    graph = sanitise_graph(
        raw_graph(),
        graph_source="rendered-visual",
        rule_indices=VISUAL_RULE_INDICES,
        require_labels=False,
    )
    nodes = {
        index: SimpleNamespace(
            tag="p",
            text_content="Visible text",
            attrs={},
            dom_path=f"html > p:nth-child({index + 1})",
        )
        for index in range(4)
    }
    report = predict_graph(
        model=FixedModel(),
        graph=graph,
        node_map=nodes,
        checkpoint={
            "rule_ids": [
                "avoid-inline-spacing",
                "color-contrast",
                "link-in-text-block",
                "meta-viewport",
                "scrollable-region-focusable",
            ]
        },
        calibration={
            "recommended": {
                "node_threshold": 0.5,
                "page_threshold": 0.5,
                "rule_thresholds": {
                    "avoid-inline-spacing": 0.5,
                    "color-contrast": 0.5,
                    "link-in-text-block": 0.5,
                    "meta-viewport": 0.5,
                    "scrollable-region-focusable": 0.5,
                },
            }
        },
        contract=FeatureContract.from_data(graph, "rendered-visual"),
        device="cpu",
        top_k=4,
    )

    assert report["axe_used_for_prediction"] is False
    assert report["prediction_count"] == 1
    assert report["predictions"][0]["predicted_rules"][0]["rule_id"] == "color-contrast"
