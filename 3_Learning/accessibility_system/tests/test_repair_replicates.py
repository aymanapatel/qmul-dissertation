from __future__ import annotations

import argparse
import json
from pathlib import Path

from accessibility_system.evaluation.repair_replicates import run_study


def _report(root: Path, condition: str, replicate: str, accepted: list[bool]) -> Path:
    directory = root / condition / replicate
    proposals = directory / "proposals"; proposals.mkdir(parents=True)
    attempts = []
    for index, outcome in enumerate(accepted):
        proposal = proposals / f"q-{index}.json"
        proposal.write_text(json.dumps({"decision": "propose", "cited_record_ids": []}), encoding="utf-8")
        attempts.append({
            "query_id": f"q-{index}", "site_id": f"site-{index}",
            "status": "accepted" if outcome else "rejected",
            "generation": {"proposal_path": str(proposal), "usage": {}},
            "validation": {
                "target_resolved": outcome, "new_regressions": [],
                "patch_evidence": {"applied": True, "oracle_match": outcome},
                "rejection_reasons": [],
            },
        })
    path = directory / "phase_9_report.json"
    path.write_text(json.dumps({"condition": condition, "attempts": attempts}), encoding="utf-8")
    return path


def test_replicate_study_requires_balanced_minimum_and_preserves_variation(tmp_path):
    runs = []
    for condition in ("no_rag", "flat_vector_rag", "graph_constrained_rag"):
        for replicate, outcomes in (("r1", [True, False]), ("r2", [True, True]), ("r3", [False, True])):
            runs.append((condition, replicate, _report(tmp_path, condition, replicate, outcomes)))
    args = argparse.Namespace(
        run=runs, generator_inputs=None, repair_truth=None, output_dir=tmp_path / "out",
        minimum_replicates=3, bootstrap_samples=100, seed=42,
    )
    report = run_study(args)
    assert report["readiness"]["replicate_study_complete"] is True
    assert report["metrics"]["graph_constrained_rag"]["replicate_count"] == 3
    assert report["metrics"]["no_rag"]["between_replicate_sample_sd"]["regression_free_accepted_rate"] > 0
    assert (args.output_dir / "run_manifest.json").is_file()


def test_replicate_study_rejects_unbalanced_design(tmp_path):
    runs = []
    for condition in ("no_rag", "flat_vector_rag", "graph_constrained_rag"):
        count = 2 if condition == "flat_vector_rag" else 3
        for index in range(count):
            replicate = f"r{index + 1}"
            runs.append((condition, replicate, _report(tmp_path, condition, replicate, [True])))
    args = argparse.Namespace(
        run=runs, generator_inputs=None, repair_truth=None, output_dir=tmp_path / "out",
        minimum_replicates=3, bootstrap_samples=20, seed=42,
    )
    report = run_study(args)
    assert report["readiness"]["replicate_study_complete"] is False
    assert report["readiness"]["replicate_ids_balanced_across_llm_conditions"] is False
