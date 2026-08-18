"""Build a fail-closed, requirement-by-requirement dissertation readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(*paths: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in paths if path.is_file()
    ]


def _phase(number: int, title: str, status: str, requirement: str, finding: str, *paths: Path) -> dict[str, Any]:
    return {
        "phase": number, "title": title, "status": status,
        "requirement": requirement, "finding": finding,
        "evidence": _evidence(*paths),
    }


def _pytest_summary(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    root = ET.parse(path).getroot()
    values = root.attrib if root.tag == "testsuite" else {
        key: sum(int(child.attrib.get(key, 0)) for child in root.findall("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    return {key: int(values.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}


def run(args: argparse.Namespace) -> dict[str, Any]:
    criteria = _load(args.registry).get("criteria", {}) if args.registry.is_file() else {}
    families = _load(args.families).get("families", {}) if args.families.is_file() else {}
    active = sum(item.get("status") == "active" and not str(item.get("criterion_id", "")).startswith("1.2.") for item in criteria.values())
    legacy = sum(item.get("status") == "legacy" for item in criteria.values())
    excluded_media = sum(str(item.get("criterion_id", "")).startswith("1.2.") and item.get("scope") == "excluded" for item in criteria.values())
    registry_ok = active == 77 and legacy == 1 and excluded_media == 9 and len(families) == 10

    tests = _pytest_summary(args.pytest_report)
    tests_ok = tests["tests"] > 0 and tests["failures"] == 0 and tests["errors"] == 0
    phase14 = _load(args.phase14) if args.phase14.is_file() else {}
    reproducible = bool(args.lockfile.is_file() and tests_ok and phase14.get("split_hash"))

    corrected = _load(args.corrected_evidence) if args.corrected_evidence.is_file() else {}
    completion = corrected.get("completion_gates", {})
    collection_ok = bool(
        completion.get("matched_cross_modal_split")
        and completion.get("versioned_rendered_visual_contract")
        and completion.get("live_chromium_accessibility_tree_contract")
    )
    annotation_manifest = _load(args.annotation_manifest) if args.annotation_manifest.is_file() else {}
    annotation_ready = not annotation_manifest.get("capture_failures", []) and annotation_manifest.get("case_count", 0) > 0
    independent_truth = args.independent_truth.is_file()

    baseline_ok = bool(phase14.get("fixture_exact_match") and args.deterministic_baseline.is_file())
    live_comparison = _load(args.live_comparison) if args.live_comparison.is_file() else {}
    architectures = {str(item.get("architecture")) for item in live_comparison.get("results", [])}
    visual_ablation = _load(args.visual_ablation) if args.visual_ablation.is_file() else {}
    specialist_ok = {"mlp", "graphsage", "gat"} <= architectures and bool(visual_ablation.get("paired_comparisons"))

    routing_ok = all(path.is_file() for path in (args.routing, args.fusion, args.fused_findings))
    detection = _load(args.detection) if args.detection.is_file() else {}
    required_methods = {
        "axe_alone", "custom_deterministic", "mlp_specialist", "graphsage_specialist",
        "gat_specialist", "visual_specialist", "interaction_specialist",
        "uncalibrated_union", "calibrated_routed_fusion",
    }
    detection_methods = set(detection.get("methods", {}))
    detection_path_complete = required_methods <= detection_methods and bool(detection.get("paired_f1_difference_95_ci"))
    detection_final = detection.get("truth_source") == "independent_manual" and detection.get("study_status") == "final"

    retrieval = _load(args.retrieval) if args.retrieval.is_file() else {}
    retrieval_metrics = retrieval.get("metrics", {})
    retrieval_ok = bool(
        {"no_rag", "flat_vector_rag", "graph_constrained_rag"} <= set(retrieval_metrics)
        and all(item.get("leakage_count") == 0 for item in retrieval_metrics.values())
        and retrieval_metrics.get("graph_constrained_rag", {}).get("source_correctness") == 1.0
        and args.knowledge_graph.is_file()
    )

    repair = _load(args.repair_study) if args.repair_study.is_file() else {}
    repair_ready = repair.get("readiness", {})
    repair_controlled = bool(
        repair_ready.get("three_required_llm_conditions_present")
        and repair_ready.get("deterministic_template_present")
        and repair_ready.get("validation_gate_ablation_present")
        and repair_ready.get("query_sets_identical")
        and args.html_report.is_file()
    )
    phase9_ok = all(
        path.is_file() and len(_load(path).get("attempts", [])) == 6
        for path in args.phase9_reports
    )
    replicates = _load(args.replicate_study) if args.replicate_study.is_file() else {}
    replicate_ready = bool(replicates.get("readiness", {}).get("replicate_study_complete"))
    ratings_ready = bool(
        repair_ready.get("two_independent_raters_with_complete_coverage")
        and repair_ready.get("automatic_acceptance_precision_recall_available")
    )
    visual_final = args.visual_final.is_file()

    phases = [
        _phase(0, "Scope and WCAG registry", "complete" if registry_ok else "incomplete",
               "77 active non-media criteria, one legacy criterion, nine excluded 1.2.x criteria, and ten issue families.",
               f"Observed active={active}, legacy={legacy}, excluded_media={excluded_media}, families={len(families)}.", args.registry, args.families),
        _phase(1, "Reproducible canonical pipeline", "complete" if reproducible else "incomplete",
               "Locked environment, deterministic split provenance, and a green complete test suite.",
               f"pytest={tests}; split_hash={phase14.get('split_hash')}; lockfile={args.lockfile.is_file()}.", args.lockfile, args.pytest_report, args.phase14),
        _phase(2, "Live aligned evidence collection", "complete" if collection_ok and annotation_ready else "incomplete",
               "Versioned rendered and live Chromium AX evidence with matched identities and fail-closed capture.",
               f"Cross-modal contracts={collection_ok}; annotation capture packet valid={annotation_ready}.", args.corrected_evidence, args.annotation_manifest),
        _phase(3, "Governed data and independent truth", "complete" if independent_truth else "pending_external",
               "Frozen site-held-out benchmark with dual independent labels, adjudication, and agreement.",
               "The frozen evidence packet exists, but completed two-rater truth has not been imported." if not independent_truth else "Independent adjudicated truth exists.", args.annotation_manifest, args.independent_truth),
        _phase(4, "Deterministic and conventional baselines", "complete" if baseline_ok else "incomplete",
               "axe and deterministic baselines with exact fixture tests and provenance.",
               f"Fixture exact match={phase14.get('fixture_exact_match')}; deterministic baseline exists={args.deterministic_baseline.is_file()}.", args.phase14, args.deterministic_baseline),
        _phase(5, "Graph and visual specialists", "complete" if specialist_ok else "incomplete",
               "Feature-matched MLP, GraphSAGE, GAT and controlled visual-cue ablations.",
               f"Architectures={sorted(architectures)}; visual ablation present={bool(visual_ablation.get('paired_comparisons'))}. Negative effects are retained.", args.live_comparison, args.visual_ablation),
        _phase(6, "Routing, fusion, calibration and abstention", "complete" if routing_ok else "incomplete",
               "Traceable routed/fused findings with explicit unsupported and review behavior.",
               f"Routing, policy, and fused outputs present={routing_ok}.", args.routing, args.fusion, args.fused_findings),
        _phase(7, "Held-out detection study", "complete" if detection_final and detection_path_complete else "pending_external",
               "All required baselines with precision/recall and site-bootstrap uncertainty against independent truth.",
               f"Statistical path complete={detection_path_complete}; final independent truth={detection_final}. Current truth={detection.get('truth_source')}.", args.detection),
        _phase(8, "Graph-aware retrieval", "complete" if retrieval_ok else "incomplete",
               "Leakage-free no-RAG, flat RAG, and typed graph-constrained retrieval evaluated before generation.",
               f"Retrieval contract complete={retrieval_ok}; graph source correctness={retrieval_metrics.get('graph_constrained_rag', {}).get('source_correctness')}.", args.retrieval, args.knowledge_graph),
        _phase(9, "Typed sandboxed repair", "complete" if phase9_ok else "incomplete",
               "Structured proposals, immutable typed patching, target re-test, browser/axe/specialist regressions, and preserved evidence.",
               f"Four matched six-query condition reports valid={phase9_ok}.", *args.phase9_reports),
        _phase(10, "Repair study and dissertation package",
               "complete" if repair_controlled and replicate_ready and ratings_ready else "pending_external",
               "Deterministic plus three LLM conditions, paired validation ablation, stochastic replicates, and blinded contextual ratings.",
               f"Controlled and JSON/HTML packaged={repair_controlled}; stochastic replicates={replicates.get('readiness', {}).get('minimum_replicates_observed', 0)}/{replicates.get('readiness', {}).get('minimum_replicates_required', 3)}; blinded ratings={ratings_ready}.",
               args.repair_study, args.replicate_study, args.rating_manifest, args.html_report),
    ]
    research_questions = {
        "RQ1_graph_relational_detection": {
            "status": "complete_null_or_negative" if detection_path_complete else "incomplete",
            "independent_confirmation": detection_final,
            "result": "GraphSAGE did not establish a positive paired site-level advantage over the feature-matched MLP in the weak-label pilot.",
        },
        "RQ2_rendered_visual_evidence": {
            "status": "complete" if visual_final else "pending_external",
            "result": "The controlled pilot is negative; independent labels are required for the final visual-cue interval.",
        },
        "RQ3_specialist_routing": {
            "status": "complete" if detection_final else "pending_external",
            "result": "The full precision/recall path runs, but its current real-page truth is axe-derived weak evidence.",
        },
        "RQ4_graph_rag_repairs": {
            "status": "complete" if replicate_ready and ratings_ready else "pending_external",
            "result": "All controlled repair conditions tie at 6/6; GraphRAG is more citation-selective but has no established success advantage.",
        },
        "RQ5_validation_safety": {
            "status": "complete_controlled_null" if repair_controlled else "incomplete",
            "result": "The paired gate ablation is implemented and null on the six already-correct controlled proposals.",
        },
        "RQ6_ablations": {
            "status": "partial",
            "result": "Visual and validation ablations are complete; independent visual truth and stochastic repair replicates remain open.",
        },
    }
    blocking_actions = []
    if not independent_truth:
        blocking_actions.append("Two independent raters must complete the frozen 28-case detection packet and adjudicate disagreements.")
    if not replicate_ready:
        blocking_actions.append("Run balanced LLM replicates r2 and r3 for all three LLM conditions, then rerun the replicate aggregator.")
    if not ratings_ready:
        blocking_actions.append("Two blinded raters must score all 24 repair candidates and adjudicate acceptability disagreements.")
    if not visual_final:
        blocking_actions.append("Run the independent visual ablation after the detection truth file has been finalized.")
    report = {
        "schema_version": 1, "status": "dissertation_readiness_audit",
        "source_plan": str(args.plan.resolve()),
        "automated_contracts_complete": all(item["status"] in {"complete", "pending_external"} for item in phases),
        "dissertation_ready": all(item["status"] == "complete" for item in phases),
        "pytest": tests, "phases": phases, "research_questions": research_questions,
        "blocking_actions": blocking_actions,
        "claim_boundary": [
            "Axe weak labels do not prove superiority over axe.",
            "Null graph, visual, validation, and repair-success effects are retained.",
            "The six controlled repair cases establish mechanism capability, not general real-page effectiveness.",
            "No WCAG-conformance claim is made beyond the measured bounded criteria and states.",
        ],
    }
    lines = [
        "# Dissertation readiness audit", "",
        f"Overall ready: **{'yes' if report['dissertation_ready'] else 'no'}**. Automated contracts implemented: **{'yes' if report['automated_contracts_complete'] else 'no'}**.", "",
        "| Phase | Requirement | Status | Authoritative finding |", "| ---: | --- | --- | --- |",
        *[f"| {item['phase']} | {item['title']} | `{item['status']}` | {item['finding']} |" for item in phases], "",
        "## Research-question status", "",
        *[f"- **{name}** — `{item['status']}`: {item['result']}" for name, item in research_questions.items()], "",
        "## Remaining blocking actions", "",
        *([f"{index}. {item}" for index, item in enumerate(blocking_actions, 1)] or ["None."]), "",
        "## Claim boundary", "",
        *[f"- {item}" for item in report["claim_boundary"]], "",
        f"Test evidence: {tests['tests']} collected, {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped.", "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dissertation_readiness_audit.json"
    md_path = args.output_dir / "DISSERTATION_READINESS_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "outputs": {json_path.name: _sha256(json_path), md_path.name: _sha256(md_path)},
        "evidence_file_count": sum(len(item["evidence"]) for item in phases),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "learning_v2/artifacts_3107_0015"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root.parent / "Plan_v3.md")
    parser.add_argument("--registry", type=Path, default=root.parent / "configs/wcag_criteria.json")
    parser.add_argument("--families", type=Path, default=root.parent / "configs/wcag_label_families.json")
    parser.add_argument("--lockfile", type=Path, default=root / "learning_v2/requirements.lock")
    parser.add_argument("--pytest-report", type=Path, default=artifacts / "verification/pytest.xml")
    parser.add_argument("--phase14", type=Path, default=artifacts / "phase_1_4/phase_1_4_summary.json")
    parser.add_argument("--deterministic-baseline", type=Path, default=artifacts / "phase_1_4/deterministic_baseline.json")
    parser.add_argument("--corrected-evidence", type=Path, default=artifacts / "dissertation_corrected_pilot/corrected_pilot_evidence.json")
    parser.add_argument("--annotation-manifest", type=Path, default=artifacts / "detection_annotation_packet/packet_manifest.json")
    parser.add_argument("--independent-truth", type=Path, default=artifacts / "detection_annotation_packet/final_independent_detection_truth.json")
    parser.add_argument("--live-comparison", type=Path, default=artifacts / "phase_5_corrected_live_ax_stratified50/comparison.json")
    parser.add_argument("--visual-ablation", type=Path, default=artifacts / "visual_ablation_stratified50/visual_ablation.json")
    parser.add_argument("--routing", type=Path, default=artifacts / "phase_6_7_corrected_weak_pilot/phase_6_routing_decisions.json")
    parser.add_argument("--fusion", type=Path, default=artifacts / "phase_6_7_corrected_weak_pilot/phase_6_fusion_policy.json")
    parser.add_argument("--fused-findings", type=Path, default=artifacts / "phase_6_7_corrected_weak_pilot/fused_findings.json")
    parser.add_argument("--detection", type=Path, default=artifacts / "phase_6_7_corrected_weak_pilot/phase_7_detection_study.json")
    parser.add_argument("--retrieval", type=Path, default=artifacts / "phase_8/phase_8_retrieval_evaluation.json")
    parser.add_argument("--knowledge-graph", type=Path, default=artifacts / "phase_8/knowledge_graph.json")
    parser.add_argument("--repair-study", type=Path, default=artifacts / "repair_benchmark_v3/phase_10/phase_10_repair_study.json")
    parser.add_argument("--html-report", type=Path, default=artifacts / "dissertation_corrected_pilot/CORRECTED_PILOT_RESULTS.html")
    parser.add_argument("--replicate-study", type=Path, default=artifacts / "repair_benchmark_v3/replicate_study/phase_10_replicate_study.json")
    parser.add_argument("--rating-manifest", type=Path, default=artifacts / "repair_benchmark_v3/human_rating_packet/packet_manifest.json")
    parser.add_argument("--visual-final", type=Path, default=artifacts / "final_visual_ablation/final_visual_ablation_study.json")
    parser.add_argument("--phase9-report", dest="phase9_reports", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=artifacts / "dissertation_readiness")
    args = parser.parse_args()
    if not args.phase9_reports:
        args.phase9_reports = [
            artifacts / "repair_benchmark_v3/runs/deterministic_template/phase_9_report.json",
            artifacts / "repair_benchmark_v2/runs/no_rag/phase_9_report.json",
            artifacts / "repair_benchmark_v2/runs/flat_vector_rag/phase_9_report.json",
            artifacts / "repair_benchmark_v2/runs/graph_constrained_rag/phase_9_report.json",
        ]
    return args


def main() -> None:
    report = run(parse_args())
    print(json.dumps({
        "status": report["status"], "automated_contracts_complete": report["automated_contracts_complete"],
        "dissertation_ready": report["dissertation_ready"], "blocking_actions": report["blocking_actions"],
    }, indent=2))


if __name__ == "__main__":
    main()
