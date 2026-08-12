"""Grounding checks performed before an LLM proposal reaches the sandbox."""

from __future__ import annotations

import re
from typing import Any

from .contracts import RepairProposal


PLACEHOLDER_TOKENS = ("requires_human_review", "todo", "unknown", "provide appropriate", "add descriptive")


def _image_alt_is_asset_identifier(value: str, finding: dict[str, Any]) -> bool:
    """Reject opaque tokens and values copied from the target image filename."""

    candidate = " ".join(str(value or "").split()).strip()
    compact = re.sub(r"[^A-Za-z0-9]", "", candidate)
    if len(compact) >= 16 and re.fullmatch(r"[0-9A-Fa-f]+", compact):
        return True
    if re.fullmatch(
        r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[1-5][0-9A-Fa-f]{3}-"
        r"[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12}",
        candidate,
    ):
        return True
    evidence = finding.get("evidence", {})
    attributes = evidence.get("attributes", {}) if isinstance(evidence, dict) else {}
    source = str(attributes.get("src", "")) if isinstance(attributes, dict) else ""
    filename = source.split("?", 1)[0].split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    def normalise(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

    return bool(stem and normalise(candidate) == normalise(stem))


def proposal_policy_errors(proposal: RepairProposal, generator_input: dict[str, Any]) -> list[str]:
    finding = generator_input.get("original_finding", {})
    errors: list[str] = []
    if proposal.query_id != str(generator_input.get("query_id", "")):
        errors.append("proposal query_id does not match generator input")
    if proposal.finding_id != str(finding.get("finding_id", "")):
        errors.append("proposal finding_id does not match original finding")
    allowed_citations = {str(item.get("record_id")) for item in generator_input.get("citations", [])}
    unknown = sorted(set(proposal.cited_record_ids) - allowed_citations)
    if unknown:
        errors.append(f"proposal cites records not supplied to the model: {unknown}")
    if proposal.decision == "propose" and allowed_citations and not proposal.cited_record_ids:
        errors.append("proposal does not cite any supplied retrieval record")
    allowed_selectors = {
        str(item) for item in finding.get("evidence", {}).get("target", []) if isinstance(item, str)
    }
    for operation in proposal.operations:
        if allowed_selectors and operation.selector not in allowed_selectors:
            errors.append(f"operation selector is outside original evidence: {operation.selector}")
        if not allowed_selectors:
            errors.append("original evidence supplies no selector for the proposed operation")
        if operation.new_value and any(token in operation.new_value.lower() for token in PLACEHOLDER_TOKENS):
            errors.append(f"operation new_value contains instructional or placeholder text: {operation.selector}")
        if (
            finding.get("rule_id") == "image-alt"
            and operation.operation == "set_attribute"
            and operation.attribute_name == "alt"
            and operation.new_value is not None
            and _image_alt_is_asset_identifier(operation.new_value, finding)
        ):
            errors.append(f"image alt value is an asset identifier rather than semantic evidence: {operation.selector}")
    if generator_input.get("safe_action") == "leave_finding_unchanged" and proposal.decision != "leave_unchanged":
        errors.append("retrieval failed but proposal does not leave the finding unchanged")
    return errors
