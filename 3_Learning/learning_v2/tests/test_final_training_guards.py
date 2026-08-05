from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from learning_v2.calibration import choose_threshold
from learning_v2.assemble_multiview_bundle import assemble
from learning_v2.experiment import _governed_split, validate_rule_support, validation_rule_average_precision
from learning_v2.final_evaluation_split import build_final_evaluation_split
from learning_v2.metrics import PredictionArrays


def test_final_evaluation_split_preserves_complete_truth_and_documents_exclusions(tmp_path: Path):
    governed_path = tmp_path / "governed.json"
    truth_path = tmp_path / "truth.json"
    governed = {"seed": 42, "train": ["train"], "val": ["val"], "test": ["kept", "blocked"]}
    truth = {
        "pairs": [
            {
                "site_id": "kept", "criterion_id": criterion, "status": "pass",
                "adjudicated": True, "annotator_ids": ["r1", "r2"],
            }
            for criterion in ("1.1.1", "1.4.3")
        ]
    }
    governed_path.write_text(json.dumps(governed), encoding="utf-8")
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    before = truth_path.read_bytes()

    result = build_final_evaluation_split(
        governed, truth, governed_path=governed_path, truth_path=truth_path,
        exclusion_reason_code="blocked", exclusion_reason="Evidence was not ratable.",
    )

    assert truth_path.read_bytes() == before
    assert result["test"] == ["kept"]
    assert result["expected_truth_pair_count"] == 2
    assert result["exclusions"] == [{
        "site_id": "blocked", "partition": "test", "reason_code": "blocked",
        "reason": "Evidence was not ratable.", "label_imputed": False,
        "evidence_paths": [
            "detection_annotation_packet/evidence/blocked/rendered.png",
            "detection_annotation_packet/evidence/blocked/resource_audit.json",
        ],
        "evidence_sha256": {},
    }]


def test_final_evaluation_split_requires_complete_site_criterion_matrix(tmp_path: Path):
    governed_path = tmp_path / "governed.json"
    truth_path = tmp_path / "truth.json"
    governed = {"train": ["train"], "val": ["val"], "test": ["a", "b"]}
    truth = {"pairs": [
        {"site_id": "a", "criterion_id": "1", "status": "pass", "adjudicated": True, "annotator_ids": ["r1", "r2"]},
        {"site_id": "a", "criterion_id": "2", "status": "pass", "adjudicated": True, "annotator_ids": ["r1", "r2"]},
        {"site_id": "b", "criterion_id": "1", "status": "pass", "adjudicated": True, "annotator_ids": ["r1", "r2"]},
    ]}
    governed_path.write_text(json.dumps(governed), encoding="utf-8")
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(ValueError, match="complete site/criterion matrix"):
        build_final_evaluation_split(
            governed, truth, governed_path=governed_path, truth_path=truth_path,
            exclusion_reason_code="blocked", exclusion_reason="Evidence was not ratable.",
        )


def test_governed_mode_never_silently_drops_graphs(tmp_path: Path):
    for site, nodes in (("train", 2), ("val", 20), ("test", 2)):
        target = tmp_path / site / "a11y-tree.pt"
        target.parent.mkdir(parents=True)
        torch.save({"data": Data(x=torch.ones((nodes, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))}, target)
    governed = {"seed": 42, "train": ["train"], "val": ["val"], "test": ["test"]}
    with pytest.raises(ValueError, match="cannot be changed"):
        _governed_split(governed, tmp_path, ["a11y-tree"], max_nodes=10)
    exact = _governed_split(governed, tmp_path, ["a11y-tree"], max_nodes=0)
    assert {name: exact[name] for name in ("train", "val", "test")} == {
        "train": ["train"], "val": ["val"], "test": ["test"]
    }


def test_requested_rule_with_inadequate_validation_support_raises(tmp_path: Path):
    paths = {}
    for site, positive in (("train", True), ("val", False), ("test", True)):
        target = tmp_path / site / "a11y-tree.pt"
        target.parent.mkdir(parents=True)
        labels = torch.zeros((2, 1)); labels[0, 0] = float(positive)
        torch.save({"data": Data(node_y_multi=labels)}, target)
        paths[site] = target
    with pytest.raises(ValueError, match="cannot be trained silently"):
        validate_rule_support(
            paths, {"train": ["train"], "val": ["val"], "test": ["test"]},
            "a11y-tree", ("image-alt",), (0,),
        )


def test_precision_floor_is_explicit_and_uses_exact_score_candidates():
    met = choose_threshold(torch.tensor([0.91, 0.82, 0.2]), torch.tensor([1, 0, 0]), precision_floor=0.8)
    assert met["precision_floor_met"] is True
    assert met["threshold"] == pytest.approx(0.91)
    unmet = choose_threshold(torch.tensor([0.91, 0.82, 0.2]), torch.tensor([0, 1, 0]), precision_floor=0.8)
    assert unmet["precision_floor_met"] is False
    assert "unconstrained_best" in unmet


def test_checkpoint_selection_score_is_threshold_independent_average_precision():
    arrays = PredictionArrays(
        node_probs=torch.tensor([0.0, 0.0]), node_labels=torch.tensor([0.0, 0.0]),
        rule_probs=torch.tensor([[0.9], [0.1]]), rule_labels=torch.tensor([[1.0], [0.0]]),
        page_probs=torch.tensor([0.0]), page_labels=torch.tensor([0.0]),
        valid_node_mask=torch.tensor([True, True]),
    )
    result = validation_rule_average_precision(arrays, ["image-alt"])
    assert result["metric"] == "rule_macro_average_precision"
    assert result["value"] == pytest.approx(1.0)


def test_multiview_assembly_rejects_model_split_mismatch_before_writing(tmp_path: Path):
    requested = {"seed": 42, "train": ["train"], "val": ["val"], "test": ["test"]}
    split_path = tmp_path / "requested.json"
    split_path.write_text(json.dumps(requested), encoding="utf-8")
    cache = tmp_path / "cache"
    for site in ("train", "val", "test"):
        target = cache / site / "a11y-tree.pt"; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"cache")
    models = tmp_path / "models"; (models / "a11y-tree").mkdir(parents=True)
    (models / "comparison.json").write_text(json.dumps({
        "pilot": False, "split_mode": "governed", "split": "governed_split.json", "results": [],
    }), encoding="utf-8")
    (models / "governed_split.json").write_text(json.dumps({
        "seed": 42, "train": ["different"], "val": ["val"], "test": ["test"],
    }), encoding="utf-8")
    output_cache = tmp_path / "output-cache"
    output_phase5 = tmp_path / "output-phase5"
    with pytest.raises(ValueError, match="do not match"):
        assemble(Namespace(
            split=split_path,
            view_cache=[f"a11y-tree={cache}"],
            view_model=[f"a11y-tree={models}"],
            output_cache=output_cache,
            output_phase5=output_phase5,
            allow_pilot_models=False,
        ))
    assert not output_cache.exists()
    assert not output_phase5.exists()
