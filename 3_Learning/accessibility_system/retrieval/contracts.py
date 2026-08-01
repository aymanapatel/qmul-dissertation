"""JSON-safe Phase 8 retrieval contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Provenance:
    publisher: str
    title: str
    url: str
    retrieved_at: str
    source_revision: str


@dataclass(frozen=True)
class KnowledgeRecord:
    record_id: str
    version: str
    record_type: str
    title: str
    summary: str
    criterion_ids: tuple[str, ...]
    rule_ids: tuple[str, ...]
    context_patterns: tuple[str, ...]
    repair_pattern_id: str
    repair_summary: str
    validation_requirements: tuple[str, ...]
    provenance: Provenance
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeRecord":
        item = dict(value)
        item["provenance"] = Provenance(**item["provenance"])
        for key in ("criterion_ids", "rule_ids", "context_patterns", "validation_requirements"):
            item[key] = tuple(item[key])
        return cls(**item)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def searchable_text(self) -> str:
        return " ".join((
            self.title, self.summary, " ".join(self.criterion_ids), " ".join(self.rule_ids),
            " ".join(self.context_patterns), self.repair_pattern_id, self.repair_summary,
            " ".join(self.validation_requirements),
        ))


@dataclass(frozen=True)
class ExemplarRecord:
    record_id: str
    source_site: str
    template_hash: str
    rule_id: str
    criterion_ids: tuple[str, ...]
    context_pattern: str
    evidence_text: str
    repair_pattern_id: str
    split: str = "train"
    record_type: str = "training_exemplar"
    schema_version: int = SCHEMA_VERSION

    def searchable_text(self) -> str:
        return " ".join((self.rule_id, " ".join(self.criterion_ids), self.context_pattern, self.evidence_text, self.repair_pattern_id))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    site_id: str
    template_hash: str
    criterion_id: str
    rule_id: str
    context_pattern: str
    evidence_text: str
    relevant_record_ids: tuple[str, ...]
    finding: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def searchable_text(self) -> str:
        return " ".join((self.criterion_id, self.rule_id, self.context_pattern, self.evidence_text))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedRecord:
    record_id: str
    score: float
    rank: int
    record_type: str
    criterion_ids: tuple[str, ...]
    rule_ids: tuple[str, ...]
    repair_pattern_id: str
    citation_url: str | None
    source_site: str | None = None
    template_hash: str | None = None
    graph_path: tuple[str, ...] = ()
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
