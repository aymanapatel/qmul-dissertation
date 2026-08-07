#!/usr/bin/env python3
"""
Aggregate a GNN batch prediction directory into comparison_summary.json.

The script is intentionally strict by default: every usable site must have a
prediction JSON, otherwise the aggregate report exits with a missing-artifact
error and writes missing_predictions.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wcag_rules import graph_source_for_rule


LEARNING_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = LEARNING_ROOT.parent
DEFAULT_REPORT_DIR = LEARNING_ROOT / "reports/gnn_batch_100"
DEFAULT_AXE_DIR = REPOSITORY_ROOT / "2_Data/browser-use/outputs/axe-core"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def f1_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def axe_rule_counts(axe_path: Path) -> tuple[int, Counter[str], Counter[str]]:
    if not axe_path.exists():
        return 0, Counter(), Counter()
    report = load_json(axe_path)
    rule_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    total = 0
    for violation in report.get("violations", []):
        rule_id = str(violation.get("id", ""))
        nodes = len(violation.get("nodes", []))
        total += nodes
        if rule_id:
            rule_counts[rule_id] += nodes
            view = graph_source_for_rule(rule_id)
            if view:
                view_counts[view] += nodes
    return total, rule_counts, view_counts


def prediction_rule_counts(items: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    rule_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    for item in items:
        rule_id = item.get("axe_rule_id")
        if not rule_id:
            continue
        rule_counts[str(rule_id)] += 1
        source_view = item.get("source_view") or graph_source_for_rule(str(rule_id))
        if source_view:
            view_counts[str(source_view)] += 1
    return rule_counts, view_counts


def site_row(domain: str, prediction: dict[str, Any], axe_dir: Path) -> dict[str, Any]:
    ground_truth = prediction.get("ground_truth", {})
    summary = prediction.get("summary", {})
    axe_path = axe_dir / domain / "page-0_home.json"
    axe_available = axe_path.exists()
    axe_total, raw_rules, _raw_views = axe_rule_counts(axe_path)
    tp = int(ground_truth.get("true_positives", 0) or 0)
    fp = int(ground_truth.get("false_positives", 0) or 0)
    fn = int(ground_truth.get("false_negatives", 0) or 0)
    mapped_true = int(ground_truth.get("true_violations", tp + fn) or 0)
    if not axe_available:
        axe_total = mapped_true
    metrics = f1_counts(tp, fp, fn)
    return {
        "domain": domain,
        "axe": axe_total,
        "raw_axe_available": axe_available,
        "mapped_true": mapped_true,
        "pred": int(summary.get("predicted_violation_count", len(prediction.get("predicted_violations", []))) or 0),
        "cand": int(summary.get("candidate_warning_count", len(prediction.get("candidate_warnings", []))) or 0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "page_prob": float(summary.get("page_violation_probability", 0) or 0),
        "page_prediction": summary.get("page_prediction", "unknown"),
        "rules": dict(raw_rules),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    metrics = f1_counts(tp, fp, fn)
    axe_total = sum(row["axe"] for row in rows)
    mapped_total = sum(row["mapped_true"] for row in rows)
    pred_total = sum(row["pred"] for row in rows)
    cand_total = sum(row["cand"] for row in rows)
    clean_rows = [row for row in rows if row["axe"] == 0]
    violation_rows = [row for row in rows if row["axe"] > 0]
    return {
        "usable": len(rows),
        "axe_total": axe_total,
        "mapped_total": mapped_total,
        "pred_total": pred_total,
        "cand_total": cand_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **metrics,
        "clean_count": len(clean_rows),
        "clean_fp_count": sum(1 for row in clean_rows if row["pred"] > 0),
        "viol_count": len(violation_rows),
        "viol_detected_count": sum(1 for row in violation_rows if row["tp"] > 0),
        "viol_none_count": sum(1 for row in violation_rows if row["tp"] == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GNN batch prediction JSON files.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--axe-dir", type=Path, default=DEFAULT_AXE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    report_dir = args.report_dir
    predictions_dir = report_dir / "predictions"
    usable_sites = read_lines(report_dir / "usable_sites.txt")
    selected_sites = read_lines(report_dir / "sites.txt")
    summary_only_sites = read_lines(report_dir / "summary_only_sites.txt")
    challenge_sites = read_lines(report_dir / "challenge_sites.txt")

    prediction_paths = {path.stem: path for path in predictions_dir.glob("*.json")}
    missing = sorted(set(usable_sites) - set(prediction_paths))
    extra = sorted(set(prediction_paths) - set(usable_sites))
    (report_dir / "missing_predictions.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    if (missing or extra) and not args.allow_missing:
        if missing:
            print(f"Missing prediction JSON files for {len(missing)} usable sites: {', '.join(missing)}")
        if extra:
            print(f"Prediction JSON files not listed as usable: {', '.join(extra)}")
        sys.exit(1)

    domains = usable_sites if usable_sites else sorted(prediction_paths)
    rows = []
    rule_actual: Counter[str] = Counter()
    view_actual: Counter[str] = Counter()
    rule_pred: Counter[str] = Counter()
    view_pred: Counter[str] = Counter()
    rule_candidate: Counter[str] = Counter()

    for domain in domains:
        path = prediction_paths.get(domain)
        if path is None:
            continue
        prediction = load_json(path)
        row = site_row(domain, prediction, args.axe_dir)
        rows.append(row)
        _axe_total, raw_rules, raw_views = axe_rule_counts(args.axe_dir / domain / "page-0_home.json")
        rule_actual.update(raw_rules)
        view_actual.update(raw_views)
        pred_rules, pred_views = prediction_rule_counts(prediction.get("predicted_violations", []))
        cand_rules, _cand_views = prediction_rule_counts(prediction.get("candidate_warnings", []))
        rule_pred.update(pred_rules)
        view_pred.update(pred_views)
        rule_candidate.update(cand_rules)

    aggregate = aggregate_rows(rows)
    non_challenge_rows = [row for row in rows if row["domain"] not in set(challenge_sites)]
    raw_axe_missing = sum(1 for row in rows if not row.get("raw_axe_available"))

    summary = {
        "selected": len(selected_sites) or len(rows),
        "usable": len(usable_sites) or len(rows),
        "summary_only": len(summary_only_sites),
        "summary_only_sites": summary_only_sites,
        "challenge": challenge_sites,
        "raw_axe_files_missing": raw_axe_missing,
        "raw_axe_counts_complete": raw_axe_missing == 0,
        **aggregate,
        "primary_non_challenge": aggregate_rows(non_challenge_rows),
        "rule_actual": rule_actual.most_common(20),
        "rule_pred": rule_pred.most_common(20),
        "rule_candidate": rule_candidate.most_common(20),
        "view_actual": dict(view_actual),
        "view_pred": dict(view_pred),
        "top_axe": sorted(rows, key=lambda row: row["axe"], reverse=True)[:10],
        "best": sorted(
            [row for row in rows if row["mapped_true"] > 0 and row["tp"] > 0],
            key=lambda row: (row["recall"], row["precision"], row["tp"]),
            reverse=True,
        )[:10],
        "low_recall": sorted(
            [row for row in rows if row["mapped_true"] > 0],
            key=lambda row: (row["recall"], -row["mapped_true"]),
        )[:10],
        "worst_gap": sorted(
            [row for row in rows if row["fn"] > 0],
            key=lambda row: (row["fn"], row["mapped_true"]),
            reverse=True,
        )[:10],
    }

    output_path = args.output or report_dir / "comparison_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote aggregate summary: {output_path}")
    print(
        f"Precision={summary['precision']:.3f} "
        f"Recall={summary['recall']:.3f} "
        f"F1={summary['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
