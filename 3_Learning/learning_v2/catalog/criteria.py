"""Versioned WCAG criterion registry schema and loader."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CriterionRecord:
    criterion_id: str
    name: str
    wcag_version: str
    level: str
    status: str  # active | legacy
    scope: str  # core | control | stretch | manual_only | excluded
    issue_family_ids: tuple[str, ...]
    candidate_generators: tuple[str, ...]
    primary_detector: str
    secondary_detectors: tuple[str, ...]
    required_evidence: tuple[str, ...]
    automation: str  # automated | assisted | manual
    ground_truth_source: str
    repair_policy: str
    validation_steps: tuple[str, ...]
    human_review_required: bool
    axe_rule_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Convert tuples to lists for JSON serialisation
        for key, value in result.items():
            if isinstance(value, tuple):
                result[key] = list(value)
        return result


@dataclass(frozen=True)
class CriterionRegistry:
    criteria: dict[str, CriterionRecord] = field(default_factory=dict)
    schema_version: int = REGISTRY_SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.criteria)

    def __contains__(self, criterion_id: str) -> bool:
        return criterion_id in self.criteria

    def __getitem__(self, criterion_id: str) -> CriterionRecord:
        return self.criteria[criterion_id]

    def get(self, criterion_id: str) -> CriterionRecord | None:
        return self.criteria.get(criterion_id)

    def active(self) -> list[CriterionRecord]:
        return [c for c in self.criteria.values() if c.status == "active"]

    def excluded(self) -> list[CriterionRecord]:
        return [c for c in self.criteria.values() if c.scope == "excluded"]

    def core(self) -> list[CriterionRecord]:
        return [c for c in self.criteria.values() if c.scope == "core"]

    def by_scope(self, scope: str) -> list[CriterionRecord]:
        return [c for c in self.criteria.values() if c.scope == scope]

    def by_detector(self, detector: str) -> list[CriterionRecord]:
        return [c for c in self.criteria.values() if c.primary_detector == detector]

    def by_family(self, family_id: str) -> list[CriterionRecord]:
        return [c for c in self.criteria.values() if family_id in c.issue_family_ids]

    def validate(self) -> list[str]:
        errors: list[str] = []
        for cid, record in self.criteria.items():
            if not cid:
                errors.append("Empty criterion_id")
            if record.criterion_id != cid:
                errors.append(f"Key mismatch: {cid} != {record.criterion_id}")
            if not record.name:
                errors.append(f"{cid}: missing name")
            if not record.primary_detector:
                errors.append(f"{cid}: missing primary_detector")
            if record.status not in ("active", "legacy"):
                errors.append(f"{cid}: invalid status {record.status}")
            if record.scope not in ("core", "control", "stretch", "manual_only", "excluded"):
                errors.append(f"{cid}: invalid scope {record.scope}")
            if record.automation not in ("automated", "assisted", "manual"):
                errors.append(f"{cid}: invalid automation {record.automation}")
        return errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionRegistry":
        criteria: dict[str, CriterionRecord] = {}
        for cid, cdata in data.get("criteria", {}).items():
            # Convert lists back to tuples
            record_data = dict(cdata)
            for key in ("issue_family_ids", "candidate_generators", "secondary_detectors",
                        "required_evidence", "validation_steps", "axe_rule_ids"):
                if isinstance(record_data.get(key), list):
                    record_data[key] = tuple(record_data[key])
            criteria[cid] = CriterionRecord(**record_data)
        return cls(criteria=criteria, schema_version=data.get("schema_version", REGISTRY_SCHEMA_VERSION))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "criteria": {cid: record.to_dict() for cid, record in self.criteria.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CriterionRegistry":
        with open(path) as f:
            return cls.from_dict(json.load(f))
