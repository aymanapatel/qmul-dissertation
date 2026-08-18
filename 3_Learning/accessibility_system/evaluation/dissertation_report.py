"""Package detection, retrieval, and repair evidence into dissertation tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _detection_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    names = (
        "axe_alone", "custom_deterministic", "mlp_specialist", "graphsage_specialist",
        "gat_specialist", "visual_specialist", "uncalibrated_union", "calibrated_routed_fusion",
    )
    return [
        {"method": name, **{key: report["methods"][name].get(key) for key in (
            "precision", "recall", "f1", "micro_pr_auc", "coverage", "manual_review_rate",
        )}}
        for name in names if name in report.get("methods", {})
    ]


def _retrieval_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"condition": name, **{key: values.get(key) for key in (
            "mean_recall_at_k", "mrr", "mean_ndcg_at_k", "source_correctness", "traceability", "leakage_count",
        )}}
        for name, values in report.get("metrics", {}).items()
    ]


def _repair_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"condition": name, **{key: values.get(key) for key in (
            "generation_success_rate", "patch_applicability_rate", "target_resolution_rate",
            "validated_repair_rate", "regression_free_accepted_rate", "rejection_rate", "human_review_rate",
            "oracle_exact_acceptance_precision", "oracle_exact_acceptance_recall",
        )}}
        for name, values in report.get("metrics", {}).items()
    ]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def build_report(detection: dict[str, Any], retrieval: dict[str, Any], repair: dict[str, Any], cache_audit: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    detection_rows = _detection_rows(detection)
    retrieval_rows = _retrieval_rows(retrieval)
    repair_rows = _repair_rows(repair)
    graph_ci = detection.get("paired_f1_difference_95_ci", {}).get("graphsage_minus_mlp_f1")
    graph_established = bool(graph_ci and graph_ci[0] > 0)
    visual = detection.get("methods", {}).get("visual_specialist", {})
    required_repair = repair.get("readiness", {})
    gates = {
        "independent_detection_truth": detection.get("truth_source") == "independent_manual" and detection.get("study_status") == "final",
        "graph_advantage_established": graph_established,
        "rendered_visual_cache_contract": bool((cache_audit or {}).get("rendered_visual_contract", {}).get("passes")),
        "visual_ablation_complete": bool(detection.get("ablations", {}).get("visual_cues", {}).get("controlled_comparison")),
        "live_accessibility_tree_provenance": bool((cache_audit or {}).get("dissertation_gates", {}).get("live_browser_accessibility_tree_proven")),
        "retrieval_leakage_free": all(row.get("leakage_count") == 0 for row in retrieval_rows),
        "matched_three_condition_repair_study": bool(required_repair.get("three_required_llm_conditions_present") and required_repair.get("query_sets_identical")),
        "blinded_human_repair_assessment": bool(required_repair.get("blinded_human_assessment_present")),
    }
    payload = {
        "schema_version": 1,
        "status": "dissertation_evidence_package",
        "research_units": {
            "detection": detection.get("unit"),
            "retrieval": "retrieval_query",
            "repair": repair.get("unit"),
        },
        "detection": {
            "study_status": detection.get("study_status"),
            "truth_source": detection.get("truth_source"),
            "site_count": detection.get("site_count"),
            "criteria": detection.get("criteria"),
            "results": detection_rows,
            "graphsage_minus_mlp_f1_95_ci": graph_ci,
            "graph_advantage_established": graph_established,
            "visual_specialist": {key: visual.get(key) for key in ("precision", "recall", "f1", "micro_pr_auc", "coverage")},
        },
        "retrieval": {"status": retrieval.get("status"), "query_count": retrieval.get("query_count"), "results": retrieval_rows},
        "repair": {"results": repair_rows, "human_assessment": repair.get("human_assessment"), "readiness": required_repair},
        "cache_audit": cache_audit,
        "completion_gates": gates,
        "dissertation_ready": all(gates.values()),
        "claim_boundary": (
            "Current real-page detection uses axe-derived weak labels and must not be described as superiority over axe. "
            "A graph effect is established only when the site-bootstrap paired confidence interval excludes zero. "
            "An LLM repair counts as automatic success only after target resolution and regression-free sandbox validation."
        ),
    }

    lines = [
        "# Dissertation methodology and current results", "",
        "## Research design", "",
        "The evaluation keeps three units separate: site–criterion pairs for detection, retrieval queries for RAG, and proposed repairs for remediation. Sites and templates are held out before model selection; thresholds are frozen on validation data; paired uncertainty is bootstrapped over sites. Axe-derived labels are treated as weak evidence, not independent ground truth.", "",
        "Graph issues are evaluated with a feature-matched MLP, GraphSAGE, and GAT on the same split and training contract. Visual claims require a versioned rendered/style/geometry cache plus a controlled structure-only versus rendered-cue ablation. Repairs use the same generator and context budget across no-RAG, flat-RAG, and graph-constrained-RAG conditions, followed by typed patching and sandbox validation.", "",
        "## Detection results", "",
        f"Status: `{detection.get('study_status')}`; truth: `{detection.get('truth_source')}`; sites: {detection.get('site_count')}; unit: `{detection.get('unit')}`.", "",
        *_markdown_table(
            ["Method", "Precision", "Recall", "F1", "PR-AUC", "Coverage", "Review"],
            [[row["method"], *(_number(row.get(key)) for key in ("precision", "recall", "f1", "micro_pr_auc", "coverage", "manual_review_rate"))] for row in detection_rows],
        ), "",
        f"GraphSAGE minus MLP paired F1 95% CI: `{graph_ci}`. Graph advantage established: **{'yes' if graph_established else 'no'}**.", "",
        f"The visual-specialist result is reported with its coverage. Rendered cache contract passed: **{'yes' if gates['rendered_visual_cache_contract'] else 'no'}**. It is not interpreted as visual-cue evidence unless both the cache contract and identical-case structure-only ablation pass.", "",
        "## Retrieval results", "",
        *_markdown_table(
            ["Condition", "Recall@k", "MRR", "nDCG@k", "Source correctness", "Traceability", "Leakage"],
            [[row["condition"], *(_number(row.get(key)) for key in ("mean_recall_at_k", "mrr", "mean_ndcg_at_k", "source_correctness", "traceability")), str(row.get("leakage_count"))] for row in retrieval_rows],
        ), "",
        "Retrieval is evaluated before generation. Graph-RAG denotes explicit typed graph traversal and constraint, not merely vector similarity over graph-labelled text.", "",
        "## Repair results", "",
        *_markdown_table(
            ["Condition", "Generation", "Applicable", "Target resolved", "Validated", "Accepted", "Oracle P", "Oracle R", "Rejected", "Review"],
            [[row["condition"], *(_number(row.get(key)) for key in (
                "generation_success_rate", "patch_applicability_rate", "target_resolution_rate", "validated_repair_rate",
                "regression_free_accepted_rate", "oracle_exact_acceptance_precision", "oracle_exact_acceptance_recall",
                "rejection_rate", "human_review_rate",
            ))] for row in repair_rows],
        ), "",
        "Generation success is not repair success. `Validated` requires target resolution with no new in-scope regressions; `Accepted` also satisfies the Phase 9 semantic and evidence policy.", "",
        f"Automatic acceptance precision: `{repair.get('human_assessment', {}).get('automatic_acceptance_precision')}`; recall: `{repair.get('human_assessment', {}).get('automatic_acceptance_recall')}`. These remain unavailable until blinded human labels are imported.", "",
        "## Dissertation-readiness gates", "",
        *[f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}" for name, passed in gates.items()], "",
        f"Overall ready: **{'yes' if all(gates.values()) else 'no'}**.", "",
        "## Claim boundary", "", payload["claim_boundary"], "",
    ]
    return payload, "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    detection = _load(args.detection); retrieval = _load(args.retrieval); repair = _load(args.repair)
    cache_audit = _load(args.cache_audit) if args.cache_audit else None
    report, markdown = build_report(detection, retrieval, repair, cache_audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dissertation_evidence.json"
    md_path = args.output_dir / "DISSERTATION_METHODOLOGY_RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "inputs": {str(path): _sha256(path) for path in (args.detection, args.retrieval, args.repair, args.cache_audit) if path},
        "outputs": {json_path.name: _sha256(json_path), md_path.name: _sha256(md_path)},
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "dissertation_ready": report["dissertation_ready"]}, indent=2))


if __name__ == "__main__":
    main()
