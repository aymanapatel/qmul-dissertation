from __future__ import annotations

import json
from pathlib import Path

from accessibility_system.controlled_benchmark import CASES
from accessibility_system.deterministic_repair import build_proposal


def _item(case: dict, index: int) -> dict:
    return {
        "condition": "no_rag",
        "query_id": f"q-{index}",
        "original_finding": {
            "finding_id": f"f-{index}", "site_id": f"s-{index}",
            "criterion_id": case["criterion_id"], "rule_id": case["rule_id"],
            "status": "verified_fail",
            "evidence": {
                "target": [case["selector"]], "html": case["html"],
                "failure_summary": case["evidence"],
            },
        },
    }


def test_deterministic_templates_do_not_need_oracle_and_cover_controlled_cases():
    for index, case in enumerate(CASES):
        item = _item(case, index)
        assert "oracle_operations" not in json.dumps(item)
        proposal = build_proposal(item, case["html"])
        assert proposal.decision == "propose", case["name"]
        assert [operation.model_dump(mode="json") for operation in proposal.operations] == case["oracle"]


def test_deterministic_template_abstains_without_explicit_context():
    item = {
        "condition": "no_rag", "query_id": "q", "original_finding": {
            "finding_id": "f", "site_id": "s", "rule_id": "image-alt",
            "evidence": {"target": ["#hero"], "failure_summary": "Missing alternative."},
        },
    }
    proposal = build_proposal(item, "<html><body><img id='hero'></body></html>")
    assert proposal.decision == "requires_human_review"
    assert proposal.operations == []
