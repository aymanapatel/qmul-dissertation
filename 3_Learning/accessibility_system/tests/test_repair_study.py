from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from accessibility_system.evaluation.repair_study import run_study
from accessibility_system.evaluation.repair_rating_packet import _candidate_id, _completed, _runs, finalize_packet


def _write_run(root: Path, condition: str, outcomes: list[tuple[str, bool, bool]]) -> Path:
    proposals = root / condition / "proposals"; proposals.mkdir(parents=True)
    attempts = []
    for index, (status, resolved, regression) in enumerate(outcomes):
        query_id = f"q-{index}"
        proposal_path = proposals / f"{query_id}.json"
        proposal_path.write_text(json.dumps({
            "decision": "propose", "cited_record_ids": ["record-1"],
        }), encoding="utf-8")
        attempts.append({
            "query_id": query_id, "site_id": f"site-{index // 2}", "status": status,
            "duration_seconds": 1.0,
            "generation": {"proposal_path": str(proposal_path), "usage": {"total_tokens": 100, "cost": 0.01}},
            "validation": {
                "target_resolved": resolved,
                "new_regressions": ["new"] if regression else [],
                "rejection_reasons": [],
                "patch_evidence": {"applied": True},
            },
        })
    path = root / condition / "phase_9_report.json"
    path.write_text(json.dumps({"condition": condition, "attempts": attempts}), encoding="utf-8")
    return path


def test_repair_study_reports_matched_conditions_and_human_precision_recall(tmp_path):
    conditions = {
        "deterministic_template": [("accepted", True, False), ("accepted", True, False)],
        "no_rag": [("rejected", False, False), ("accepted", True, False)],
        "flat_vector_rag": [("accepted", True, False), ("rejected", False, True)],
        "graph_constrained_rag": [("accepted", True, False), ("accepted", True, False)],
    }
    runs = [(name, _write_run(tmp_path, name, outcomes)) for name, outcomes in conditions.items()]
    generator_inputs = tmp_path / "inputs.json"
    generator_inputs.write_text(json.dumps([
        {"condition": condition, "query_id": f"q-{index}", "citations": [{"record_id": "record-1"}]}
        for condition in conditions for index in range(2)
    ]), encoding="utf-8")
    ratings = tmp_path / "ratings.json"
    ratings.write_text(json.dumps({"schema_version": 1, "blinded": True, "adjudicated": True, "ratings": [
        {"condition": condition, "query_id": f"q-{index}", "rater_id": rater,
         "contextual_correctness": 4, "safety": 5, "helpfulness": 4,
         "acceptable": condition in {"deterministic_template", "graph_constrained_rag"} or index == 1}
        for condition in conditions for index in range(2) for rater in ("r1", "r2")
    ]}), encoding="utf-8")
    args = argparse.Namespace(
        run=runs, generator_inputs=generator_inputs, human_ratings=ratings,
        repair_truth=None, output_dir=tmp_path / "out", bootstrap_samples=100, seed=42,
    )
    report = run_study(args)
    assert report["readiness"]["dissertation_ready"] is True
    assert report["readiness"]["deterministic_template_present"] is True
    assert report["readiness"]["validation_gate_ablation_present"] is True
    assert report["metrics"]["graph_constrained_rag"]["regression_free_accepted_rate"] == 1.0
    assert report["human_assessment"]["automatic_acceptance_precision"] == pytest.approx(5 / 6)
    assert report["human_assessment"]["automatic_acceptance_recall"] == pytest.approx(5 / 6)
    assert "graph_constrained_rag_vs" not in " ".join(report["paired_comparisons"])
    assert (args.output_dir / "run_manifest.json").is_file()


def test_validation_gate_ablation_uses_identical_proposals(tmp_path):
    runs = [
        ("deterministic_template", _write_run(tmp_path, "deterministic_template", [("accepted", True, False)])),
        ("no_rag", _write_run(tmp_path, "no_rag", [("rejected", False, True)])),
        ("flat_vector_rag", _write_run(tmp_path, "flat_vector_rag", [("accepted", True, False)])),
        ("graph_constrained_rag", _write_run(tmp_path, "graph_constrained_rag", [("accepted", True, False)])),
    ]
    args = argparse.Namespace(
        run=runs, generator_inputs=None, human_ratings=None, repair_truth=None,
        output_dir=tmp_path / "out", bootstrap_samples=10, seed=1,
    )
    report = run_study(args)
    ablation = report["metrics"]["no_rag"]["validation_gate_ablation"]
    assert ablation["without_validation"]["provisional_acceptance_rate"] == 1.0
    assert ablation["with_validation"]["automatic_acceptance_rate"] == 0.0
    assert ablation["without_validation"]["known_regression_acceptance_rate"] == 1.0


def test_repair_study_rejects_duplicate_query_ids(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"condition": "no_rag", "attempts": [
        {"query_id": "q", "site_id": "a"}, {"query_id": "q", "site_id": "b"},
    ]}), encoding="utf-8")
    args = argparse.Namespace(
        run=[("no_rag", path)], generator_inputs=None, human_ratings=None,
        repair_truth=None, output_dir=tmp_path / "out", bootstrap_samples=10, seed=1,
    )
    with pytest.raises(ValueError, match="unique"):
        run_study(args)


def test_blinded_rating_packet_helpers_are_strict_and_stable():
    assert _candidate_id("graph", "q1", 42) == _candidate_id("graph", "q1", 42)
    assert _completed({"contextual_correctness": 5, "safety": 4, "helpfulness": 3, "acceptable": True})
    assert not _completed({"contextual_correctness": 0, "safety": 4, "helpfulness": 3, "acceptable": True})
    with pytest.raises(ValueError, match="unique"):
        _runs(["graph=/one", "graph=/two"])


def test_repair_rating_finalizer_restores_hidden_condition(tmp_path):
    (tmp_path / "coordinator").mkdir(); (tmp_path / "rater_packets").mkdir()
    (tmp_path / "coordinator/identity_map.json").write_text(json.dumps({"identities": [{"candidate_id": "c1", "condition": "graph_constrained_rag", "query_id": "q1"}]}), encoding="utf-8")
    row = {"candidate_id": "c1", "contextual_correctness": 5, "safety": 5, "helpfulness": 5, "acceptable": True, "notes": "sound"}
    for index in (1, 2):
        (tmp_path / f"rater_packets/rater_{index}.json").write_text(json.dumps({"ratings": [row]}), encoding="utf-8")
    (tmp_path / "rater_packets/adjudicator.json").write_text(json.dumps({"ratings": [row]}), encoding="utf-8")
    args = argparse.Namespace(packet_dir=tmp_path, output=tmp_path / "ratings.json")
    result = finalize_packet(args)
    assert result["ratings"][0]["condition"] == "graph_constrained_rag"
    assert result["acceptable_disagreement_count"] == 0
