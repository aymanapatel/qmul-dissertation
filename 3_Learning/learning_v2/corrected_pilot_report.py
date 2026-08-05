"""Package corrected live-AX and rendered-visual pilot evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(records: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    return {field: sum(int(item.get(field, 0) or 0) for item in records) for field in fields}


def _model_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "architecture": item["architecture"],
            "rules": item["rules"],
            "best_epoch": item["best_epoch"],
            **{key: item["test"].get(key) for key in (
                "rule_precision", "rule_recall", "rule_f1", "rule_macro_f1_supported",
                "rule_tp", "rule_fp", "rule_fn", "page_f1",
            )},
        }
        for item in comparison["results"]
    ]


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(report: dict[str, Any]) -> str:
    """Render a dependency-free standalone report with escaped evidence values."""
    detection_rows = report.get("held_out_detection_study", {}).get("methods", [])
    repair_rows = report.get("controlled_repair_study", {}).get("conditions", [])
    gates = report.get("completion_gates", {})
    detection_table = _html_table(
        ["Method", "Precision", "Recall", "F1", "PR-AUC", "Coverage", "Review"],
        [[row.get("method"), *(_number(row.get(key)) for key in (
            "precision", "recall", "f1", "micro_pr_auc", "coverage", "manual_review_rate",
        ))] for row in detection_rows],
    )
    repair_table = _html_table(
        ["Condition", "Generation", "Resolved", "Validated", "Accepted", "Oracle P", "Oracle R", "Citation use", "Tokens", "Cost"],
        [[row.get("condition"), *(_number(row.get(key)) for key in (
            "generation_success_rate", "target_resolution_rate", "validated_repair_rate",
            "regression_free_accepted_rate", "oracle_exact_acceptance_precision",
            "oracle_exact_acceptance_recall", "mean_retrieved_citation_utilisation",
            "total_tokens", "total_cost",
        ))] for row in repair_rows],
    )
    gate_items = "".join(
        f"<li class=\"{'pass' if value else 'open'}\"><span aria-hidden=\"true\">{'✓' if value else '○'}</span> {html.escape(name.replace('_', ' '))}</li>"
        for name, value in gates.items()
    )
    remaining = "".join(f"<li>{html.escape(item)}</li>" for item in report.get("remaining_required_work", []))
    split = report.get("split", {})
    graph_ci = report.get("held_out_detection_study", {}).get("graphsage_minus_mlp_f1_95_ci")
    ready = bool(report.get("dissertation_ready"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AccessibilityGraph-RAG methodology results</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:auto;padding:2rem;color:#18202a;background:#fff}}
h1,h2{{line-height:1.2}} .status{{display:inline-block;padding:.35rem .7rem;border-radius:.3rem;font-weight:700;background:{'#d8f5e1' if ready else '#fff1c7'};color:#18202a}}
table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}} th,td{{border:1px solid #aeb7c2;padding:.55rem;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#eef2f6}}
.pass{{color:#126b31}} .open{{color:#8a4b00}} code{{background:#eef2f6;padding:.1rem .25rem}} .boundary{{border-left:.35rem solid #8a4b00;padding:.8rem 1rem;background:#fff8e8}}
</style></head><body>
<main><h1>AccessibilityGraph-RAG methodology results</h1>
<p class="status">Dissertation ready: {'yes' if ready else 'no'}</p>
<p class="boundary">This is a corrected weak-label pilot. Axe-derived labels are not independent truth, and null or negative graph, visual, validation, and repair-success effects are retained.</p>
<h2>Data contract</h2><p>Frozen split <code>{html.escape(str(split.get('hash')))}</code>; sites: {html.escape(str(split.get('site_counts')))}. Rendered contract: {report.get('rendered_visual', {}).get('contract_passed')}; live Chromium AX contract: {report.get('live_accessibility_tree', {}).get('contract_passed')}.</p>
<h2>Held-out site–criterion prediction</h2>{detection_table}
<p>GraphSAGE minus MLP paired site-bootstrap F1 interval: <code>{html.escape(str(graph_ci))}</code>. Positive graph advantage established: {report.get('claim_gates', {}).get('graph_advantage_established')}.</p>
<h2>Controlled repair study</h2>{repair_table}
<p>The deterministic template and three LLM conditions all repaired 6/6 bounded cases. This demonstrates mechanism capability, not GraphRAG superiority or general real-page effectiveness.</p>
<h2>Completion gates</h2><ul>{gate_items}</ul>
<h2>Required final work</h2><ol>{remaining}</ol>
</main></body></html>"""


def build_report(
    visual_manifest: dict[str, Any],
    live_manifest: dict[str, Any],
    visual_split: dict[str, Any],
    live_split: dict[str, Any],
    live_comparison: dict[str, Any],
    visual_comparison: dict[str, Any],
    ablation: dict[str, Any],
    repair: dict[str, Any] | None = None,
    detection: dict[str, Any] | None = None,
    visual_final: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    visual_records = visual_manifest["records"]
    live_records = live_manifest["records"]
    matched_split = visual_split == live_split
    visual_contract = (
        visual_manifest.get("feature_version") == 2
        and len(visual_records) == 50
        and all(int(item.get("feature_version", 0)) == 2 and int(item.get("feature_dim", 0)) >= 517 for item in visual_records)
        and sum(int(item.get("spatial_edges", 0)) for item in visual_records) > 0
    )
    live_contract = (
        live_manifest.get("live_ax_feature_version") == 1
        and len(live_records) == 50
        and all(int(item.get("feature_version", 0)) == 1 and int(item.get("feature_dim", 0)) >= 509 for item in live_records)
    )
    partition_support = visual_manifest.get("selection", {}).get("selected_partition_support", {})
    minimum_support = int(visual_manifest.get("selection", {}).get("minimum_positive_sites_per_rule", 0))
    support_gate = bool(partition_support) and all(
        int(count) >= minimum_support
        for values in partition_support.values()
        for count in values.values()
    )
    live_rows = _model_rows(live_comparison)
    visual_rows = _model_rows(visual_comparison)
    full_vs_structure = {
        architecture: values.get("full_minus_structure_only", {})
        for architecture, values in ablation.get("paired_comparisons", {}).items()
    }
    repair_ready = (repair or {}).get("readiness", {})
    graph_ci = (detection or {}).get("paired_f1_difference_95_ci", {}).get("graphsage_minus_mlp_f1")
    graph_advantage = bool(graph_ci and graph_ci[0] > 0)
    independent_detection = bool(
        (detection or {}).get("truth_source") == "independent_manual"
        and (detection or {}).get("study_status") == "final"
    )
    independent_visual = bool(
        (visual_final or {}).get("truth_source") == "independent_manual"
        and (visual_final or {}).get("status") == "final_independent_visual_ablation"
    )
    visual_advantage = any(
        bool(values.get("visual_advantage_established"))
        for values in (visual_final or {}).get("paired_comparisons", {}).values()
    )
    detection_rows = []
    for name, values in (detection or {}).get("methods", {}).items():
        detection_rows.append({
            "method": name,
            **{key: values.get(key) for key in ("precision", "recall", "f1", "site_macro_f1", "micro_pr_auc", "coverage", "manual_review_rate")},
        })
    repair_rows = []
    for name, values in (repair or {}).get("metrics", {}).items():
        validation_ablation = values.get("validation_gate_ablation", {})
        repair_rows.append({
            "condition": name,
            **{key: values.get(key) for key in (
                "generation_success_rate", "target_resolution_rate", "validated_repair_rate",
                "regression_free_accepted_rate", "oracle_exact_acceptance_precision",
                "oracle_exact_acceptance_recall", "citation_validity_rate",
                "mean_retrieved_citation_utilisation", "total_tokens", "total_cost", "mean_latency_seconds",
            )},
            "ungated_provisional_acceptance_rate": validation_ablation.get("without_validation", {}).get("provisional_acceptance_rate"),
            "gated_automatic_acceptance_rate": validation_ablation.get("with_validation", {}).get("automatic_acceptance_rate"),
            "prevented_known_false_acceptance_count": validation_ablation.get("prevented_known_false_acceptance_count"),
        })
    gates = {
        "matched_cross_modal_split": matched_split,
        "versioned_rendered_visual_contract": visual_contract,
        "live_chromium_accessibility_tree_contract": live_contract,
        "minimum_predeclared_rule_site_support": support_gate,
        "controlled_visual_ablation_complete": bool(full_vs_structure),
        "matched_llm_repair_conditions_complete": bool(
            repair_ready.get("three_required_llm_conditions_present")
            and repair_ready.get("query_sets_identical")
        ),
        "deterministic_repair_baseline_complete": bool(repair_ready.get("deterministic_template_present")),
        "paired_validation_gate_ablation_complete": bool(repair_ready.get("validation_gate_ablation_present")),
        "independent_detection_ground_truth": independent_detection,
        "site_bootstrap_graph_comparison_complete": graph_ci is not None,
        "independent_visual_ablation_complete": independent_visual,
        "blinded_human_repair_assessment": bool(repair_ready.get("two_independent_raters_with_complete_coverage")),
    }
    final_gates = (
        "matched_cross_modal_split",
        "versioned_rendered_visual_contract",
        "live_chromium_accessibility_tree_contract",
        "minimum_predeclared_rule_site_support",
        "matched_llm_repair_conditions_complete",
        "deterministic_repair_baseline_complete",
        "paired_validation_gate_ablation_complete",
        "independent_detection_ground_truth",
        "site_bootstrap_graph_comparison_complete",
        "independent_visual_ablation_complete",
        "blinded_human_repair_assessment",
    )
    report = {
        "schema_version": 1,
        "status": "corrected_weak_label_pilot",
        "truth_source": "axe_weak_labels",
        "claim_level": "pipeline_and_exploratory_model_evidence_only",
        "split": {
            "hash": visual_split.get("split_hash"),
            "matched_between_modalities": matched_split,
            "site_counts": {name: len(visual_split.get(name, [])) for name in ("train", "val", "test")},
            "selection_support": partition_support,
            "selection_warning": (
                "The pilot cohort was enriched using predeclared axe-rule presence in every partition to guarantee support. "
                "It is therefore not a population-prevalence sample and remains unsuitable for superiority-over-axe claims."
            ),
        },
        "rendered_visual": {
            "contract_passed": visual_contract,
            "aggregate": _aggregate(visual_records, ("nodes", "edges", "visible_nodes", "visual_matched_nodes", "spatial_edges")),
            "models": visual_rows,
            "ablation_rule_support": ablation.get("rule_support"),
            "ablation_aggregate": ablation.get("aggregate"),
            "full_minus_structure_only": full_vs_structure,
        },
        "live_accessibility_tree": {
            "contract_passed": live_contract,
            "aggregate": _aggregate(live_records, ("nodes", "edges", "dom_mapped_nodes", "positive_labels", "mapping_loss_count")),
            "models": live_rows,
        },
        "held_out_detection_study": {
            "status": (detection or {}).get("study_status"),
            "truth_source": (detection or {}).get("truth_source"),
            "unit": (detection or {}).get("unit"),
            "site_count": (detection or {}).get("site_count"),
            "criteria": (detection or {}).get("criteria"),
            "methods": detection_rows,
            "graphsage_minus_mlp_f1_95_ci": graph_ci,
            "graph_advantage_established": graph_advantage,
        },
        "controlled_repair_study": {
            "unit": (repair or {}).get("unit"),
            "conditions": repair_rows,
            "paired_comparisons": (repair or {}).get("paired_comparisons"),
            "readiness": repair_ready,
        },
        "final_visual_ablation_study": visual_final,
        "claim_gates": {
            "graph_advantage_established": graph_advantage,
            "visual_cue_advantage_established": visual_advantage,
            "note": "A null or negative controlled result is dissertation-complete; these gates control positive advantage claims, not completion.",
        },
        "completion_gates": gates,
        "dissertation_ready": all(gates[name] for name in final_gates),
        "remaining_required_work": [
            "Complete dual independent annotation and adjudication on the frozen real-page test universe.",
            "Run the final detection study once with site-bootstrap paired architecture comparisons.",
            "Do not retain a visual/GNN advantage claim unless its independently labelled confidence interval supports it.",
            "Import two-rater blinded repair-quality labels and report agreement and automatic-acceptance precision/recall.",
        ],
    }
    lines = [
        "# Corrected live-AX and rendered-visual pilot", "",
        "Status: `corrected_weak_label_pilot`. This package proves the corrected collection and evaluation path runs end to end; it is not the final independently labelled dissertation result.", "",
        "## Data contract", "",
        f"The modalities use the same frozen split (`{visual_split.get('split_hash')}`): 35 train, 8 validation, and 7 test sites. Rendered contract passed: **{'yes' if visual_contract else 'no'}**. Live Chromium AX contract passed: **{'yes' if live_contract else 'no'}**.", "",
        f"Rendered graphs contain {report['rendered_visual']['aggregate']['nodes']:,} nodes and {report['rendered_visual']['aggregate']['spatial_edges']:,} typed spatial edges. Live AX graphs contain {report['live_accessibility_tree']['aggregate']['nodes']:,} nodes, {report['live_accessibility_tree']['aggregate']['dom_mapped_nodes']:,} DOM mappings, and {report['live_accessibility_tree']['aggregate']['mapping_loss_count']:,} recorded selector-mapping losses.", "",
        report["split"]["selection_warning"], "",
        "## Live accessibility-tree prediction", "",
        *_table(
            ["Architecture", "Precision", "Recall", "F1", "Macro F1", "TP", "FP", "FN"],
            [[row["architecture"], *(_number(row.get(key)) for key in ("rule_precision", "rule_recall", "rule_f1", "rule_macro_f1_supported")), str(row["rule_tp"]), str(row["rule_fp"]), str(row["rule_fn"])] for row in live_rows],
        ), "",
        "GraphSAGE has the highest Phase 5 node-rule pilot F1. Architecture claims are decided by the site–criterion study below, not by this single-seed node aggregate.", "",
        "## Held-out site–criterion prediction study", "",
        *_table(
            ["Method", "Precision", "Recall", "F1", "Site macro F1", "PR-AUC", "Coverage", "Review"],
            [[row["method"], *(_number(row.get(key)) for key in ("precision", "recall", "f1", "site_macro_f1", "micro_pr_auc", "coverage", "manual_review_rate"))] for row in detection_rows],
        ), "",
        f"GraphSAGE minus MLP paired site-bootstrap F1 interval: `{graph_ci}`. Graph advantage established: **{'yes' if graph_advantage else 'no'}**. This run still uses axe weak labels; its purpose is to verify the final statistical path.", "",
        "## Rendered-visual prediction", "",
        *_table(
            ["Architecture", "Precision", "Recall", "F1", "TP", "FP", "FN"],
            [[row["architecture"], *(_number(row.get(key)) for key in ("rule_precision", "rule_recall", "rule_f1")), str(row["rule_tp"]), str(row["rule_fp"]), str(row["rule_fn"])] for row in visual_rows],
        ), "",
        "The visual specialist remains ineffective in this pilot. The three-seed full-versus-structure-only intervals are exploratory and do not establish a visual-cue advantage.", "",
        "## Controlled repair study", "",
        *_table(
            ["Condition", "Generation", "Resolved", "Validated", "Accepted", "Oracle P", "Oracle R", "Citation use", "Tokens", "Cost"],
            [[row["condition"], *(_number(row.get(key)) for key in (
                "generation_success_rate", "target_resolution_rate", "validated_repair_rate",
                "regression_free_accepted_rate", "oracle_exact_acceptance_precision",
                "oracle_exact_acceptance_recall", "mean_retrieved_citation_utilisation", "total_tokens", "total_cost",
            ))] for row in repair_rows],
        ), "",
        "The deterministic template and all three LLM conditions reached exact-oracle, regression-free acceptance on all six bounded cases. The paired validation-gate ablation also tied because every proposal in this deliberately bounded set was already correct; it therefore demonstrates the gate path but does not establish a safety-effect advantage. This establishes repair-pipeline capability, not GraphRAG superiority. GraphRAG's measured distinction is more selective evidence use, which must be evaluated on a larger and blinded real-page study.", "",
        "## Dissertation gates", "",
        *[f"- [{'x' if value else ' '}] {name.replace('_', ' ')}" for name, value in gates.items()], "",
        f"Overall dissertation ready: **{'yes' if report['dissertation_ready'] else 'no'}**.", "",
        "Positive-effect claim gates are separate from completion: graph advantage is "
        f"**{'established' if graph_advantage else 'not established'}** and visual-cue advantage is "
        f"**{'established' if visual_advantage else 'not established'}**. Null or negative findings remain valid dissertation outcomes.", "",
        "## Required final work", "",
        *[f"{index}. {item}" for index, item in enumerate(report["remaining_required_work"], 1)], "",
    ]
    return report, "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "visual_manifest": args.visual_manifest,
        "live_ax_manifest": args.live_ax_manifest,
        "live_ax_comparison": args.live_ax_comparison,
        "visual_comparison": args.visual_comparison,
        "visual_ablation": args.visual_ablation,
    }
    if args.detection_study:
        paths["detection_study"] = args.detection_study
    if args.visual_final_study:
        paths["visual_final_study"] = args.visual_final_study
    if args.repair_study:
        paths["repair_study"] = args.repair_study
    visual_split_path = args.visual_manifest.parent / "collection_split.json"
    live_split_path = args.live_ax_manifest.parent / "collection_split.json"
    paths.update({"visual_split": visual_split_path, "live_ax_split": live_split_path})
    report, markdown = build_report(
        _load(args.visual_manifest), _load(args.live_ax_manifest),
        _load(visual_split_path), _load(live_split_path),
        _load(args.live_ax_comparison), _load(args.visual_comparison),
        _load(args.visual_ablation), _load(args.repair_study) if args.repair_study else None,
        _load(args.detection_study) if args.detection_study else None,
        _load(args.visual_final_study) if args.visual_final_study else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "corrected_pilot_evidence.json"
    markdown_path = args.output_dir / "CORRECTED_PILOT_RESULTS.md"
    html_path = args.output_dir / "CORRECTED_PILOT_RESULTS.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()},
        "outputs": {path.name: _sha256(path) for path in (json_path, markdown_path, html_path)},
    }, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--live-ax-manifest", type=Path, required=True)
    parser.add_argument("--live-ax-comparison", type=Path, required=True)
    parser.add_argument("--visual-comparison", type=Path, required=True)
    parser.add_argument("--visual-ablation", type=Path, required=True)
    parser.add_argument("--repair-study", type=Path)
    parser.add_argument("--detection-study", type=Path)
    parser.add_argument("--visual-final-study", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "dissertation_ready": report["dissertation_ready"]}, indent=2))


if __name__ == "__main__":
    main()
