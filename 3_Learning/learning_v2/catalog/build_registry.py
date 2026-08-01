"""Build validated WCAG criterion and issue-family registries from the CSV source.

This script parses `misc/WCAG_All_ExceptTimebased.csv` and produces:
  - `configs/wcag_criteria.json`   (one record per criterion)
  - `configs/wcag_label_families.json` (one record per issue family)

The registry build asserts the active non-media scope matches Plan_v3.md §4.1.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from .criteria import CriterionRecord, CriterionRegistry
from .families import FamilyRecord, FamilyRegistry

# WCAG 2.2 level mapping (from the W3C specification)
LEVEL_MAP: dict[str, str] = {
    "1.1.1": "A", "1.2.1": "A", "1.2.2": "A", "1.2.3": "A", "1.2.4": "AA",
    "1.2.5": "AA", "1.2.6": "AAA", "1.2.7": "AAA", "1.2.8": "AAA", "1.2.9": "AAA",
    "1.3.1": "A", "1.3.2": "A", "1.3.3": "A", "1.3.4": "AA", "1.3.5": "AA", "1.3.6": "AAA",
    "1.4.1": "A", "1.4.2": "A", "1.4.3": "AA", "1.4.4": "AA", "1.4.5": "AA",
    "1.4.6": "AAA", "1.4.7": "AAA", "1.4.8": "AAA", "1.4.9": "AAA", "1.4.10": "AA",
    "1.4.11": "AA", "1.4.12": "AA", "1.4.13": "AA",
    "2.1.1": "A", "2.1.2": "A", "2.1.3": "AAA", "2.1.4": "A",
    "2.2.1": "A", "2.2.2": "A", "2.2.3": "AAA", "2.2.4": "AAA", "2.2.5": "AAA", "2.2.6": "AAA",
    "2.3.1": "A", "2.3.2": "AAA", "2.3.3": "AAA",
    "2.4.1": "A", "2.4.2": "A", "2.4.3": "A", "2.4.4": "A", "2.4.5": "AA",
    "2.4.6": "AA", "2.4.7": "AA", "2.4.8": "AAA", "2.4.9": "AAA", "2.4.10": "AAA",
    "2.4.11": "AA", "2.4.12": "AAA", "2.4.13": "AAA",
    "2.5.1": "A", "2.5.2": "A", "2.5.3": "A", "2.5.4": "A",
    "2.5.5": "AAA", "2.5.6": "AAA", "2.5.7": "AA", "2.5.8": "AA",
    "3.1.1": "A", "3.1.2": "AA", "3.1.3": "AAA", "3.1.4": "AAA", "3.1.5": "AAA", "3.1.6": "AAA",
    "3.2.1": "A", "3.2.2": "A", "3.2.3": "AA", "3.2.4": "AA", "3.2.5": "AAA", "3.2.6": "AA",
    "3.3.1": "A", "3.3.2": "A", "3.3.3": "AA", "3.3.4": "AA", "3.3.5": "AAA",
    "3.3.6": "AAA", "3.3.7": "AA", "3.3.8": "AA", "3.3.9": "AAA",
    "4.1.1": "A", "4.1.2": "A", "4.1.3": "AA",
}

# axe rule → criterion mapping (from src/wcag_rules.py)
AXE_RULE_MAP: dict[str, tuple[str, ...]] = {
    "area-alt": ("1.1.1", "2.4.4", "4.1.2"),
    "aria-allowed-attr": ("4.1.2",),
    "aria-command-name": ("4.1.2",),
    "aria-hidden-focus": ("4.1.2",),
    "aria-input-field-name": ("4.1.2",),
    "aria-meter-name": ("1.1.1",),
    "aria-progressbar-name": ("1.1.1",),
    "aria-prohibited-attr": ("4.1.2",),
    "aria-required-attr": ("4.1.2",),
    "aria-required-children": ("1.3.1",),
    "aria-required-parent": ("1.3.1",),
    "aria-roles": ("4.1.2",),
    "aria-toggle-field-name": ("4.1.2",),
    "aria-tooltip-name": ("4.1.2",),
    "aria-valid-attr": ("4.1.2",),
    "aria-valid-attr-value": ("4.1.2",),
    "autocomplete-valid": ("1.3.5",),
    "avoid-inline-spacing": ("1.4.12",),
    "blink": ("2.2.2",),
    "button-name": ("4.1.2",),
    "color-contrast": ("1.4.3",),
    "definition-list": ("1.3.1",),
    "dlitem": ("1.3.1",),
    "document-title": ("2.4.2",),
    "frame-title": ("4.1.2",),
    "html-has-lang": ("3.1.1",),
    "html-lang-valid": ("3.1.1",),
    "image-alt": ("1.1.1",),
    "input-button-name": ("4.1.2",),
    "input-image-alt": ("1.1.1", "4.1.2"),
    "label": ("4.1.2",),
    "link-in-text-block": ("1.4.1",),
    "link-name": ("2.4.4", "4.1.2"),
    "list": ("1.3.1",),
    "listitem": ("1.3.1",),
    "marquee": ("2.2.2",),
    "meta-refresh": ("2.2.1",),
    "meta-viewport": ("1.4.4",),
    "nested-interactive": ("4.1.2",),
    "object-alt": ("1.1.1",),
    "role-img-alt": ("1.1.1",),
    "scrollable-region-focusable": ("2.1.1", "2.1.3"),
    "select-name": ("4.1.2",),
    "summary-name": ("4.1.2",),
    "svg-img-alt": ("1.1.1",),
    "valid-lang": ("3.1.2",),
}

# Issue-family definitions (from Plan_v3 §4.4 and the CSV label-family rows)
FAMILY_DEFS = [
    ("missing-accessible-name", "Missing accessible name",
     ("1.1.1", "2.4.6", "3.3.2", "4.1.2"),
     "Needs node + label + ARIA relationship reasoning."),
    ("broken-form-labelling", "Broken form labelling",
     ("1.3.1", "3.3.1", "3.3.2", "3.3.3"),
     "Excellent for DOM/accessibility-tree graphs."),
    ("low-contrast", "Low contrast",
     ("1.4.3", "1.4.11"),
     "Needs visual node + CSS/background relationship."),
    ("keyboard-focus-issue", "Keyboard/focus issue",
     ("2.1.1", "2.1.2", "2.4.3", "2.4.7", "2.4.11", "2.4.13"),
     "Needs sequential/focus-order edges."),
    ("poor-semantic-structure", "Poor semantic structure",
     ("1.3.1", "2.4.1", "2.4.2", "2.4.6"),
     "Needs heading, landmark, list, table, and parent-child relationships."),
    ("unclear-link-button-purpose", "Unclear link/button purpose",
     ("2.4.4", "2.4.9", "2.5.3", "4.1.2"),
     "Needs text, accessible name, visual label, and surrounding context."),
    ("dynamic-content-not-announced", "Dynamic content not announced",
     ("4.1.3", "3.2.1", "3.2.2"),
     "Needs event/state-change graph and ARIA live-region reasoning."),
    ("touch-pointer-target-issue", "Touch/pointer target issue",
     ("2.5.5", "2.5.8", "2.5.1", "2.5.7"),
     "Needs bounding boxes, spatial adjacency, and interaction modelling."),
    ("media-alternative-issue", "Media alternative issue",
     ("1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.2.5", "1.2.6", "1.2.7", "1.2.8", "1.2.9"),
     "Mostly content/media metadata; less graph-heavy unless media is embedded in complex UI."),
    ("authentication-error-prevention-issue", "Authentication/error-prevention issue",
     ("3.3.4", "3.3.7", "3.3.8", "3.3.9"),
     "Good for process-level graph across multiple pages/states."),
]

# Scope classification from Plan_v3 §4.4
SCOPE_MAP: dict[str, str] = {
    # Family 1: accessible names, forms, relational semantics
    "1.1.1": "core", "1.3.1": "core", "2.4.4": "core", "2.4.6": "core",
    "3.3.1": "core", "3.3.2": "core", "4.1.2": "core",
    # Family 2: rendered contrast, focus, target geometry
    "1.4.1": "core", "1.4.3": "core", "1.4.11": "core", "1.4.13": "core",
    "2.4.7": "core", "2.4.11": "core", "2.5.8": "core",
    # Family 3: keyboard, focus order, dynamic announcements
    "2.1.1": "core", "2.1.2": "core", "2.4.3": "core", "4.1.3": "core",
    # Family 4: contextual purpose and bounded remediation
    "2.4.2": "control", "3.1.1": "control", "3.1.2": "control", "1.3.5": "control",
    # Everything else defaults to stretch
}

# Detector assignment heuristics based on CSV columns
def _classify_detector(find_method: str, fix_method: str) -> tuple[str, tuple[str, ...]]:
    """Return (primary_detector, secondary_detectors)."""
    fm = find_method.lower()
    fx = fix_method.lower()
    detectors: list[str] = []

    if "deterministic" in fm or "deterministic" in fx or "contrast formula" in fm or "parser" in fm:
        detectors.append("deterministic")
    if "gnn" in fm or "graph" in fm or "accessibility-tree" in fm:
        detectors.append("structural")
    if "visual" in fm or "pixel" in fm or "ocr" in fm or "contrast" in fm:
        detectors.append("visual")
    if "keyboard" in fm or "playwright" in fm or "interaction" in fm or "focus" in fm or "automation" in fm:
        detectors.append("interaction")
    if "nlp" in fm or "llm" in fm or "semantic" in fm or "nlp" in fx:
        detectors.append("semantic")

    if not detectors:
        detectors = ["manual"]

    primary = detectors[0]
    secondary = tuple(detectors[1:])
    return primary, secondary


def _automation_level(find_method: str, tool_note: str) -> str:
    fm = find_method.lower()
    note = tool_note.lower()
    if "manual" in note and "partly" not in note:
        return "manual"
    if "manual" in fm and "automation" not in fm:
        return "manual"
    if "partly" in note or "assisted" in note:
        return "assisted"
    return "automated"


def build_criterion_registry(csv_path: Path) -> CriterionRegistry:
    criteria: dict[str, CriterionRecord] = {}
    current_area = ""

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            criterion_raw = row["Success Criterion"].strip()
            # Skip label-family header and family rows
            if not criterion_raw or criterion_raw.startswith("Label family"):
                # Check if this is a family definition row
                if row["WCAG Area"].strip().startswith("**"):
                    # This might be a family row — check if it has no criterion but has families
                    pass
                continue

            # Extract criterion ID like "1.1.1" from "**1.1.1 Non-text Content**"
            m = re.search(r"\*\*(\d+\.\d+\.\d+)", criterion_raw)
            if not m:
                continue
            cid = m.group(1)

            # Extract name
            name = re.sub(r"\*\*\d+\.\d+\.\d+\s*", "", criterion_raw).strip().rstrip("*").strip()

            # Get WCAG area
            area = row["WCAG Area"].strip()
            if area:
                current_area = area

            # Level from authoritative map
            level = LEVEL_MAP.get(cid, "unknown")

            # Status: 4.1.1 is legacy in WCAG 2.2
            status = "legacy" if cid == "4.1.1" else "active"

            # Scope
            scope = SCOPE_MAP.get(cid, "stretch")

            # Skip 1.2.x (time-based media) — excluded per Plan_v3 §4.1
            if cid.startswith("1.2."):
                scope = "excluded"
                status = "active"  # still active in WCAG, just excluded from this project

            # axe rule IDs that map to this criterion
            axe_rules = tuple(rid for rid, cids in AXE_RULE_MAP.items() if cid in cids)

            # Detector classification from CSV columns
            primary, secondary = _classify_detector(
                row["Best Method to Find"], row["Best Method to Fix"]
            )

            # Automation level
            automation = _automation_level(
                row["Best Method to Find"],
                row["Possible to Test with axe-core, WAVE, or Other a11y Tools?"],
            )

            # Issue families this criterion belongs to
            family_ids = tuple(
                fid for fid, _, fcriteria, _ in FAMILY_DEFS if cid in fcriteria
            )

            # Required evidence (simplified from CSV)
            evidence_raw = row["Inputs/Details Required to Find and Resolve Issue"]
            required_evidence = tuple(e.strip() for e in evidence_raw.split(",") if e.strip())[:5]

            record = CriterionRecord(
                criterion_id=cid,
                name=name,
                wcag_version="2.2",
                level=level,
                status=status,
                scope=scope,
                issue_family_ids=family_ids,
                candidate_generators=("axe", "custom_rules", "graph_patterns"),
                primary_detector=primary,
                secondary_detectors=secondary,
                required_evidence=required_evidence,
                automation=automation,
                ground_truth_source="axe_weak_labels" if axe_rules else "manual_review",
                repair_policy="sandbox_validated" if scope == "core" else "human_approved",
                validation_steps=("targeted_retest", "regression_check"),
                human_review_required=(automation == "manual"),
                axe_rule_ids=axe_rules,
            )
            criteria[cid] = record

    return CriterionRegistry(criteria=criteria)


def build_family_registry() -> FamilyRegistry:
    families: dict[str, FamilyRecord] = {}
    for fid, name, criteria, rationale in FAMILY_DEFS:
        families[fid] = FamilyRecord(
            family_id=fid,
            name=name,
            wcag_criteria=criteria,
            graph_modelling_rationale=rationale,
        )
    return FamilyRegistry(families=families)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("misc/WCAG_All_ExceptTimebased.csv")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("configs")

    criteria = build_criterion_registry(csv_path)
    families = build_family_registry()

    # Validation assertions from Plan_v3 §4.1
    active_non_media = [c for c in criteria.criteria.values() if c.status == "active" and c.scope != "excluded"]
    legacy = [c for c in criteria.criteria.values() if c.status == "legacy"]
    excluded = [c for c in criteria.criteria.values() if c.scope == "excluded"]
    core = [c for c in criteria.criteria.values() if c.scope == "core"]
    control = [c for c in criteria.criteria.values() if c.scope == "control"]

    print(f"Total criteria: {len(criteria)}")
    print(f"  Active (non-media): {len(active_non_media)}")
    print(f"  Legacy: {len(legacy)}")
    print(f"  Excluded (1.2.x): {len(excluded)}")
    print(f"  Core scope: {len(core)}")
    print(f"  Control scope: {len(control)}")

    # Plan_v3 exit gate: 77 active non-media criteria, 1 legacy, 9 excluded 1.2.x, 10 families
    assert len(active_non_media) == 77, f"Expected 77 active non-media criteria, got {len(active_non_media)}"
    assert len(legacy) == 1, f"Expected 1 legacy criterion (4.1.1), got {len(legacy)}"
    assert len(excluded) == 9, f"Expected 9 excluded 1.2.x criteria, got {len(excluded)}"
    assert len(families) == 10, f"Expected 10 issue families, got {len(families)}"

    # Registry validation
    errors = criteria.validate()
    if errors:
        print(f"Registry validation errors: {errors}")
        sys.exit(1)

    errors = families.validate()
    if errors:
        print(f"Family validation errors: {errors}")
        sys.exit(1)

    criteria.save(output_dir / "wcag_criteria.json")
    families.save(output_dir / "wcag_label_families.json")
    print(f"\nSaved to {output_dir}/wcag_criteria.json and wcag_label_families.json")


if __name__ == "__main__":
    main()
