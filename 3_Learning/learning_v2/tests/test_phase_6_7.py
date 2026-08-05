import pytest

from learning_v2.contracts import Candidate, DetectorObservation
from learning_v2.fusion import FusionPolicy, RegistryRouter, fuse_observations, validate_fused_provenance
from learning_v2.catalog.criteria import CriterionRecord, CriterionRegistry
from learning_v2.study import MethodOutput, _load_independent_truth, bootstrap_ci, evaluate_method


def observation(detector, status, confidence, *, error=None):
    return DetectorObservation(
        observation_id=detector, site_id="site", criterion_id="1.1.1",
        detector_id=detector, status=status, confidence=confidence,
        evidence={"source": detector}, error=error,
    )


def test_fusion_agreement_preserves_every_contributor():
    findings = fuse_observations(
        [observation("axe", "fail", 0.9), observation("gat", "fail", 0.8)],
        FusionPolicy({"axe": 0.5, "gat": 0.6}),
    )
    assert findings[0].status == "fail"
    assert len(findings[0].contributing_observations) == 2
    assert set(findings[0].evidence) == {"axe", "gat"}
    validate_fused_provenance(findings)


def test_fusion_disagreement_abstains():
    finding = fuse_observations(
        [observation("rule", "pass", 0.9), observation("gat", "fail", 0.9)],
        FusionPolicy(),
    )[0]
    assert finding.status == "needs_review"
    assert finding.human_review_required
    assert finding.conflicts == ("detector_disagreement",)


@pytest.mark.parametrize("state", ["unsupported", "collection_failed"])
def test_missing_or_failed_detector_never_becomes_pass(state):
    finding = fuse_observations([observation("browser", state, 0.0, error="missing")], FusionPolicy())[0]
    assert finding.status == state


@pytest.mark.parametrize("state", ["unsupported", "collection_failed", "needs_review"])
def test_pass_cannot_hide_an_unresolved_detector_state(state):
    finding = fuse_observations(
        [observation("rule", "pass", 0.9), observation("other", state, 0.0)],
        FusionPolicy(),
    )[0]
    assert finding.status == "needs_review"
    assert finding.human_review_required


def test_router_allows_multiple_specialists_and_reports_missing():
    record = CriterionRecord(
        criterion_id="1.1.1", name="Non-text", wcag_version="2.2", level="A", status="active", scope="core",
        issue_family_ids=(), candidate_generators=("axe",), primary_detector="structural", secondary_detectors=("semantic",),
        required_evidence=("DOM",), automation="assisted", ground_truth_source="manual", repair_policy="review",
        validation_steps=("retest",), human_review_required=False, axe_rule_ids=("image-alt",),
    )
    decision = RegistryRouter(CriterionRegistry({"1.1.1": record})).route(
        Candidate("candidate", "site", ("1.1.1",)), available_detectors={"axe", "mlp", "gat"},
    )[0]
    assert decision.detector_ids == ("axe", "mlp", "gat")
    assert "semantic" in decision.missing_detectors
    assert decision.status == "needs_review"


def test_phase7_metrics_and_site_bootstrap_are_reproducible():
    universe = {("a", "1"), ("a", "2"), ("b", "1"), ("b", "2")}
    truth = {("a", "1"), ("b", "2")}
    output = MethodOutput({pair: float(pair in truth) for pair in universe}, set(truth), set(universe))
    metrics = evaluate_method(output, truth, universe)
    assert metrics["f1"] == 1.0
    assert metrics["micro_pr_auc"] == 1.0
    assert bootstrap_ci(output, truth, universe, samples=50, seed=7) == bootstrap_ci(output, truth, universe, samples=50, seed=7)


def test_final_claim_rejects_weak_labels():
    # The guard is intentionally tested without touching model artifacts.
    from argparse import Namespace
    from learning_v2.study import run_study
    with pytest.raises(ValueError, match="independent_manual"):
        run_study(Namespace(final=True, truth_source="axe_weak_labels"))


def test_independent_truth_must_cover_every_pair_and_be_dual_adjudicated(tmp_path):
    import json
    universe = {("a", "1"), ("b", "1")}
    path = tmp_path / "truth.json"
    path.write_text(json.dumps({"pairs": [
        {"site_id": "a", "criterion_id": "1", "status": "fail", "adjudicated": True, "annotator_ids": ["x", "y"]},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        _load_independent_truth(path, universe)
    path.write_text(json.dumps({"annotation_protocol": "dual", "pairs": [
        {"site_id": site, "criterion_id": "1", "status": "fail" if site == "a" else "pass", "adjudicated": True, "annotator_ids": ["x", "y"]}
        for site in ("a", "b")
    ]}), encoding="utf-8")
    truth, metadata = _load_independent_truth(path, universe)
    assert truth == {("a", "1")}
    assert metadata["dual_annotated"] is True
