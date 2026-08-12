"""Strict, JSON-safe contracts for Phase 9 repair attempts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Decision = Literal["propose", "requires_human_review", "leave_unchanged"]
OperationName = Literal[
    "set_attribute",
    "remove_attribute",
    "replace_text",
    "insert_label_before",
    "set_style_property",
    "remove_meta_viewport_restriction",
]
ValidationOutcome = Literal["accepted", "rejected", "requires_human_review"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepairOperation(StrictModel):
    """One bounded DOM transformation; unused fields must be null."""

    operation: OperationName
    selector: str = Field(min_length=1, max_length=500)
    attribute_name: str | None = Field(max_length=100)
    css_property: str | None = Field(max_length=100)
    new_value: str | None = Field(max_length=2000)

    @model_validator(mode="after")
    def fields_match_operation(self) -> "RepairOperation":
        if self.operation in {"set_attribute", "remove_attribute"} and not self.attribute_name:
            raise ValueError(f"{self.operation} requires attribute_name")
        if self.operation == "set_attribute" and self.new_value is None:
            raise ValueError("set_attribute requires new_value")
        if self.operation in {"replace_text", "insert_label_before"} and self.new_value is None:
            raise ValueError(f"{self.operation} requires new_value")
        if self.operation == "set_style_property" and (not self.css_property or self.new_value is None):
            raise ValueError("set_style_property requires css_property and new_value")
        return self


class VisualBounds(StrictModel):
    """Rendered CSS-pixel geometry copied from the same-session capture."""

    x: float
    y: float
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)


class VisualObservation(StrictModel):
    """A bounded visual fact inspected while proposing a repair."""

    source: str = Field(min_length=1, max_length=200)
    selector: str = Field(min_length=1, max_length=500)
    tag: str = Field(max_length=100)
    text: str = Field(max_length=500)
    bounds: VisualBounds | None
    foreground_rgb: list[int] | None
    background_rgb: list[int] | None
    contrast_ratio: float | None
    required_contrast_ratio: float | None
    contrast_failure: bool
    contrast_failure_source: str | None = Field(max_length=200)

    @model_validator(mode="after")
    def rgb_values_are_valid(self) -> "VisualObservation":
        for name, value in (
            ("foreground_rgb", self.foreground_rgb),
            ("background_rgb", self.background_rgb),
        ):
            if value is not None and (
                len(value) != 3 or any(channel < 0 or channel > 255 for channel in value)
            ):
                raise ValueError(f"{name} must contain exactly three 0-255 channels")
        return self


class RepairProposal(StrictModel):
    """The only model output accepted by the Phase 9 executor."""

    schema_version: Literal[1]
    proposal_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    query_id: str = Field(min_length=1, max_length=200)
    finding_id: str = Field(min_length=1, max_length=200)
    decision: Decision
    operations: list[RepairOperation] = Field(max_length=8)
    rationale: str = Field(min_length=1, max_length=3000)
    expected_resolution: str = Field(min_length=1, max_length=1500)
    cited_record_ids: list[str] = Field(max_length=20)
    uncertainty: str = Field(max_length=1500)
    inspected_visual_elements: list[VisualObservation] = Field(max_length=12)
    requires_human_review: bool
    human_review_reasons: list[str] = Field(max_length=20)
    validation_steps: list[str] = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def decision_is_consistent(self) -> "RepairProposal":
        if self.decision == "propose" and not self.operations:
            raise ValueError("A proposed repair must contain at least one operation")
        if self.decision == "leave_unchanged" and self.operations:
            raise ValueError("leave_unchanged must not contain operations")
        if self.decision == "requires_human_review" and not self.requires_human_review:
            raise ValueError("requires_human_review decision must set the matching flag")
        if self.requires_human_review and not self.human_review_reasons:
            raise ValueError("Human review requires at least one reason")
        return self


class GenerationResult(StrictModel):
    proposal: RepairProposal
    response_id: str | None
    model: str
    usage: dict[str, Any]
    refusal: str | None
    request_trace: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(StrictModel):
    schema_version: Literal[1] = 1
    attempt_id: str
    outcome: ValidationOutcome
    target_resolved: bool
    new_regressions: list[str]
    human_review_reasons: list[str]
    rejection_reasons: list[str]
    before_sha256: str
    after_sha256: str
    source_unchanged: bool
    detector_results: dict[str, Any]
    accessibility_tree_diff: dict[str, Any]
    interaction_diff: dict[str, Any]
    visual_diff: dict[str, Any]
    functional_diff: dict[str, Any]
    patch_evidence: dict[str, Any]
    collection_failures: list[str]
    artifact_paths: dict[str, str]
    specialist_scope: list[str]
