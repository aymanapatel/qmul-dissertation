"""WCAG issue-family registry for graph-modelling categories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FamilyRecord:
    family_id: str
    name: str
    wcag_criteria: tuple[str, ...]
    graph_modelling_rationale: str
    schema_version: int = REGISTRY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, tuple):
                result[key] = list(value)
        return result


@dataclass(frozen=True)
class FamilyRegistry:
    families: dict[str, FamilyRecord] = field(default_factory=dict)
    schema_version: int = REGISTRY_SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.families)

    def __contains__(self, family_id: str) -> bool:
        return family_id in self.families

    def __getitem__(self, family_id: str) -> FamilyRecord:
        return self.families[family_id]

    def get(self, family_id: str) -> FamilyRecord | None:
        return self.families.get(family_id)

    def validate(self) -> list[str]:
        errors: list[str] = []
        for fid, record in self.families.items():
            if not fid:
                errors.append("Empty family_id")
            if record.family_id != fid:
                errors.append(f"Key mismatch: {fid} != {record.family_id}")
            if not record.name:
                errors.append(f"{fid}: missing name")
            if not record.wcag_criteria:
                errors.append(f"{fid}: no criteria linked")
        return errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FamilyRegistry":
        families: dict[str, FamilyRecord] = {}
        for fid, fdata in data.get("families", {}).items():
            record_data = dict(fdata)
            if isinstance(record_data.get("wcag_criteria"), list):
                record_data["wcag_criteria"] = tuple(record_data["wcag_criteria"])
            families[fid] = FamilyRecord(**record_data)
        return cls(families=families, schema_version=data.get("schema_version", REGISTRY_SCHEMA_VERSION))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "families": {fid: record.to_dict() for fid, record in self.families.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FamilyRegistry":
        with open(path) as f:
            return cls.from_dict(json.load(f))
