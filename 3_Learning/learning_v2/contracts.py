"""JSON-safe evidence, finding, routing, and run contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CONTRACT_SCHEMA_VERSION = 1


@dataclass
class NodeIdentity:
    node_id: int
    css_path: str
    tag: str
    parent_id: int | None = None
    frame: str = "main"


@dataclass
class EvidenceNode:
    identity: NodeIdentity
    attributes: dict[str, Any]
    text: str
    accessible_name: str
    role: str


@dataclass
class ScanArtifact:
    site_id: str
    source_url: str
    html_path: str
    axe_path: str
    html_sha256: str
    axe_sha256: str
    collector_version: str
    nodes: list[EvidenceNode] = field(default_factory=list)
    browser_evidence: dict[str, Any] = field(default_factory=dict)
    collection_failures: list[str] = field(default_factory=list)
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    finding_id: str
    site_id: str
    rule_id: str
    criterion_ids: list[str]
    detector: str
    node: NodeIdentity | None
    status: str = "fail"
    impact: str | None = None
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)
    human_review_required: bool = False
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FINDING_STATES = frozenset({"pass", "fail", "needs_review", "unsupported", "collection_failed"})


@dataclass(frozen=True)
class Candidate:
    """A detector-neutral issue candidate presented to the registry router."""

    candidate_id: str
    site_id: str
    criterion_ids: tuple[str, ...]
    rule_ids: tuple[str, ...] = ()
    generator: str = "unknown"
    evidence_types: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorObservation:
    """One specialist's explicit judgement for a site/criterion/target."""

    observation_id: str
    site_id: str
    criterion_id: str
    detector_id: str
    status: str
    confidence: float
    rule_id: str | None = None
    target_id: str = "page"
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in FINDING_STATES:
            raise ValueError(f"Invalid finding state: {self.status}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusedFinding:
    """Criterion-level result retaining all detector provenance and conflicts."""

    finding_id: str
    site_id: str
    criterion_id: str
    target_id: str
    status: str
    confidence: float
    contributing_observations: tuple[dict[str, Any], ...]
    evidence: dict[str, Any]
    conflicts: tuple[str, ...] = ()
    human_review_required: bool = False
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in FINDING_STATES:
            raise ValueError(f"Invalid finding state: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    phase: str
    config: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    environment: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
