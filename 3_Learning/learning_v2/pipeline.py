"""Prepare Phase 1–4 artifacts from the saved axe-core corpus."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from .baselines import deterministic_findings, run_corpus_baseline
from .evidence import collect_many
from .governance import write_governance_artifacts


PACKAGES = ["torch", "torch-geometric", "beautifulsoup4", "lxml", "playwright", "scikit-learn", "numpy", "pytest"]


def environment_manifest() -> dict:
    values = {}
    for package in PACKAGES:
        try: values[package] = version(package)
        except Exception: values[package] = "missing"
    return {"python": sys.version, "platform": platform.platform(), "packages": values}


def fixture_evaluation(fixtures_dir: Path) -> dict:
    html = fixtures_dir / "mixed_issues.html"; gold = json.loads((fixtures_dir / "mixed_issues.gold.json").read_text())
    predicted = deterministic_findings(html, site_id="mixed_issues")
    predicted_rules = [finding.rule_id for finding in predicted]; gold_rules = [entry["rule_id"] for entry in gold["findings"]]
    return {
        "gold_count": len(gold_rules), "predicted_count": len(predicted_rules),
        "missing_rules": sorted(set(gold_rules) - set(predicted_rules)),
        "unexpected_rules": sorted(set(predicted_rules) - set(gold_rules)),
        "exact_rule_multiset_match": sorted(predicted_rules) == sorted(gold_rules),
        "predictions": [finding.to_dict() for finding in predicted],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--evidence-sites", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment_manifest(); (args.output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    inventory_path, split_path = write_governance_artifacts(args.corpus_dir, args.output_dir, seed=args.seed)
    inventory = json.loads(inventory_path.read_text()); split = json.loads(split_path.read_text())
    sample_ids = split["train"][:args.evidence_sites]
    sample_dirs = [args.corpus_dir / site for site in sample_ids]
    evidence_counts = collect_many(sample_dirs, args.output_dir / "evidence_sample")
    fixture = fixture_evaluation(Path(__file__).parent / "fixtures")
    (args.output_dir / "fixture_evaluation.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    baseline = run_corpus_baseline(args.corpus_dir)
    (args.output_dir / "deterministic_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 1, "phases": [1, 2, 3, 4], "corpus": str(args.corpus_dir),
        "complete_sites": inventory["complete_site_count"], "unique_html": inventory["unique_html_count"],
        "split_hash": split["split_hash"], "evidence_sample": evidence_counts,
        "fixture_exact_match": fixture["exact_rule_multiset_match"], "baseline_metrics": baseline["metrics"],
        "outputs": {
            "environment": "environment.json", "inventory": inventory_path.name, "split": split_path.name,
            "fixture_evaluation": "fixture_evaluation.json", "deterministic_baseline": "deterministic_baseline.json",
        },
    }
    (args.output_dir / "phase_1_4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
