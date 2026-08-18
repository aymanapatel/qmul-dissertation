"""Run the Phase 9 sandbox on the independently known meta-viewport fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .repair.contracts import RepairOperation, RepairProposal
from .repair.validators import validate_repair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--axe-js", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposal = RepairProposal(
        schema_version=1,
        proposal_id="controlled-meta-viewport",
        query_id="controlled-fixture-meta-viewport",
        finding_id="controlled-meta-viewport-finding",
        decision="propose",
        operations=[RepairOperation(
            operation="remove_meta_viewport_restriction",
            selector="meta[name=viewport]",
            attribute_name=None,
            css_property=None,
            new_value=None,
        )],
        rationale="Remove maximum-scale and user-scalable restrictions while retaining the device-width declaration.",
        expected_resolution="The controlled meta-viewport violation no longer reproduces.",
        cited_record_ids=[],
        uncertainty="",
        inspected_visual_elements=[],
        requires_human_review=False,
        human_review_reasons=[],
        validation_steps=[
            "Resolve the target in an isolated copy.", "Rerun deterministic and axe detectors.",
            "Compare accessibility, interaction, visual, and functional evidence.",
        ],
        confidence=1.0,
    )
    finding = {
        "finding_id": "controlled-meta-viewport-finding",
        "site_id": "mixed_issues_controlled_fixture",
        "criterion_id": "1.4.4",
        "rule_id": "meta-viewport",
        "status": "verified_fail",
        "semantic_verified": True,
        "evidence": {"target": ["meta[name=viewport]"]},
    }
    result = validate_repair(
        source_path=args.fixture,
        proposal=proposal,
        original_finding=finding,
        output_dir=args.output_dir,
        axe_js=args.axe_js,
    )
    summary = {
        "schema_version": 1,
        "phase": 9,
        "scope": "controlled_fixture_validation_only_no_llm_call",
        "outcome": result.outcome,
        "target_resolved": result.target_resolved,
        "new_regressions": result.new_regressions,
        "validation_artifact": result.artifact_paths["validation"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "controlled_fixture_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
