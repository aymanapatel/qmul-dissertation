"""Registry-driven specialist routing, calibrated fusion, and abstention.

The module deliberately treats detector absence and detector failure as states,
not evidence that a criterion passed.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .catalog import CriterionRegistry
from .contracts import Candidate, DetectorObservation, FusedFinding


DETECTOR_ALIASES = {
    "structural": ("mlp", "graphsage", "gat"),
    "visual": ("visual", "rendered-visual"),
    "interaction": ("interaction",),
    "deterministic": ("deterministic-html",),
    "semantic": ("semantic",),
    "manual": ("human-review",),
}

GENERATOR_DETECTORS = {
    "axe": ("axe",),
    "custom_rules": ("deterministic-html",),
    "graph_patterns": ("mlp", "graphsage", "gat"),
    "visual_checks": ("visual",),
    "interaction_traces": ("interaction",),
}


@dataclass(frozen=True)
class FusionPolicy:
    """Frozen validation-derived thresholds used by the final fusion stage."""

    source_thresholds: dict[str, float] = field(default_factory=dict)
    fail_threshold: float = 0.65
    review_threshold: float = 0.35
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.review_threshold <= self.fail_threshold <= 1:
            raise ValueError("Expected 0 <= review_threshold <= fail_threshold <= 1")
        if any(not 0 <= value <= 1 for value in self.source_thresholds.values()):
            raise ValueError("Source thresholds must be between zero and one")

    def threshold_for(self, detector_id: str) -> float:
        return float(self.source_thresholds.get(detector_id, 0.5))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_thresholds": dict(sorted(self.source_thresholds.items())),
            "fail_threshold": self.fail_threshold,
            "review_threshold": self.review_threshold,
        }


@dataclass(frozen=True)
class RouteDecision:
    candidate_id: str
    criterion_id: str
    detector_ids: tuple[str, ...]
    status: str
    missing_detectors: tuple[str, ...] = ()
    reason: str = ""


class RegistryRouter:
    def __init__(self, registry: CriterionRegistry):
        self.registry = registry

    def route(
        self,
        candidate: Candidate,
        *,
        available_detectors: Iterable[str],
        collection_failed: bool = False,
    ) -> list[RouteDecision]:
        available = set(available_detectors)
        decisions = []
        for criterion_id in candidate.criterion_ids:
            record = self.registry.get(criterion_id)
            if record is None or record.scope == "excluded" or record.status == "legacy":
                decisions.append(RouteDecision(candidate.candidate_id, criterion_id, (), "unsupported", reason="criterion_not_in_active_scope"))
                continue
            if collection_failed:
                decisions.append(RouteDecision(candidate.candidate_id, criterion_id, (), "collection_failed", reason="required_collection_failed"))
                continue
            requested_groups = (record.primary_detector, *record.secondary_detectors)
            requested = []
            for generator in record.candidate_generators:
                requested.extend(GENERATOR_DETECTORS.get(generator, ()))
            for group in requested_groups:
                requested.extend(DETECTOR_ALIASES.get(group, (group,)))
            routed = tuple(dict.fromkeys(detector for detector in requested if detector in available))
            missing = tuple(dict.fromkeys(detector for detector in requested if detector not in available))
            if record.automation == "manual":
                status, reason = "needs_review", "criterion_requires_human_review"
            elif not routed:
                status, reason = "unsupported", "no_registered_specialist_available"
            else:
                status, reason = "needs_review", "awaiting_specialist_observations"
            decisions.append(RouteDecision(candidate.candidate_id, criterion_id, routed, status, missing, reason))
        return decisions


def _normalised_failure_score(observation: DetectorObservation, policy: FusionPolicy) -> float:
    """Map a source score and its decision threshold to a comparable [0, 1] score."""
    threshold = policy.threshold_for(observation.detector_id)
    probability = observation.confidence
    if threshold <= 0:
        return probability
    if threshold >= 1:
        return 1.0 if probability >= 1 else probability
    if probability >= threshold:
        return 0.5 + 0.5 * (probability - threshold) / (1 - threshold)
    return 0.5 * probability / threshold


def fuse_observations(observations: Iterable[DetectorObservation], policy: FusionPolicy) -> list[FusedFinding]:
    groups: dict[tuple[str, str, str], list[DetectorObservation]] = defaultdict(list)
    for observation in observations:
        groups[(observation.site_id, observation.criterion_id, observation.target_id)].append(observation)

    results = []
    for (site_id, criterion_id, target_id), members in sorted(groups.items()):
        statuses = {member.status for member in members}
        failures = [member for member in members if member.status == "collection_failed"]
        unsupported = [member for member in members if member.status == "unsupported"]
        unresolved = [member for member in members if member.status == "needs_review"]
        judged = [member for member in members if member.status in {"pass", "fail"}]
        explicit_fail = [member for member in judged if member.status == "fail"]
        explicit_pass = [member for member in judged if member.status == "pass"]
        conflicts = []
        if explicit_fail and explicit_pass:
            conflicts.append("detector_disagreement")
        if failures and judged:
            conflicts.append("partial_collection_failure")
        if unsupported and judged:
            conflicts.append("partial_detector_unsupported")
        if unresolved and judged:
            conflicts.append("detector_requested_review")

        if failures and not judged:
            status, confidence = "collection_failed", 0.0
        elif unsupported and not judged and not failures:
            status, confidence = "unsupported", 0.0
        elif conflicts:
            status = "needs_review"
            confidence = max((_normalised_failure_score(member, policy) for member in explicit_fail), default=0.0)
        elif explicit_fail:
            scores = [_normalised_failure_score(member, policy) for member in explicit_fail]
            # Noisy-or retains corroboration without discarding individual scores.
            confidence = 1.0
            for score in scores:
                confidence *= 1.0 - score
            confidence = 1.0 - confidence
            status = "fail" if confidence >= policy.fail_threshold else "needs_review"
        elif explicit_pass and not failures:
            confidence = max(member.confidence for member in explicit_pass)
            status = "pass"
        else:
            confidence = max((member.confidence for member in members), default=0.0)
            status = "needs_review" if confidence >= policy.review_threshold or statuses else "unsupported"

        evidence = {
            member.observation_id: member.evidence
            for member in members
            if member.evidence
        }
        raw_id = f"{site_id}|{criterion_id}|{target_id}|fusion-v1"
        results.append(FusedFinding(
            finding_id=hashlib.sha256(raw_id.encode()).hexdigest()[:20],
            site_id=site_id,
            criterion_id=criterion_id,
            target_id=target_id,
            status=status,
            confidence=round(float(confidence), 8),
            contributing_observations=tuple(member.to_dict() for member in members),
            evidence=evidence,
            conflicts=tuple(conflicts),
            human_review_required=status == "needs_review",
        ))
    return results


def validate_fused_provenance(findings: Iterable[FusedFinding]) -> None:
    """Raise if a report invents a pass or drops contributing detector evidence."""
    for finding in findings:
        if not finding.contributing_observations:
            raise ValueError(f"Fused finding {finding.finding_id} has no provenance")
        statuses = {item["status"] for item in finding.contributing_observations}
        if finding.status == "pass" and statuses != {"pass"}:
            raise ValueError("A pass may only be emitted when every usable observation passes")
        if finding.status == "fail" and "fail" not in statuses:
            raise ValueError("A fused failure must retain at least one failing observation")
