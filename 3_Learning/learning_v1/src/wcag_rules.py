"""
wcag_rules.py

Constants for axe-core/WCAG rule classification.

The rule registry is intentionally the single source of truth for:
- stable model indices
- the graph/view that is allowed to learn each rule
- WCAG success criteria used in reports
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch


RULE_OWNER_A11Y_TREE = "a11y-tree"
RULE_OWNER_DOM = "dom"
RULE_OWNER_DOM_PAGE = "dom-page"
RULE_OWNER_RENDERED_VISUAL = "rendered-visual"


@dataclass(frozen=True)
class RuleSpec:
    axe_rule_id: str
    rule_index: int
    owner: str
    wcag_ids: Tuple[str, ...]
    category: str


RULE_SPECS: Tuple[RuleSpec, ...] = (
    RuleSpec("area-alt", 0, RULE_OWNER_A11Y_TREE, ("1.1.1", "2.4.4", "4.1.2"), "Text alternatives"),
    RuleSpec("aria-allowed-attr", 1, RULE_OWNER_DOM, ("4.1.2",), "ARIA"),
    RuleSpec("aria-command-name", 2, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Name, role, value"),
    RuleSpec("aria-hidden-focus", 3, RULE_OWNER_DOM, ("4.1.2",), "ARIA"),
    RuleSpec("aria-input-field-name", 4, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Forms"),
    RuleSpec("aria-meter-name", 5, RULE_OWNER_A11Y_TREE, ("1.1.1",), "Name, role, value"),
    RuleSpec("aria-progressbar-name", 6, RULE_OWNER_A11Y_TREE, ("1.1.1",), "Name, role, value"),
    RuleSpec("aria-prohibited-attr", 7, RULE_OWNER_DOM, ("4.1.2",), "ARIA"),
    RuleSpec("aria-required-attr", 8, RULE_OWNER_DOM, ("4.1.2",), "ARIA"),
    RuleSpec("aria-required-children", 9, RULE_OWNER_DOM, ("1.3.1",), "Structure"),
    RuleSpec("aria-required-parent", 10, RULE_OWNER_DOM, ("1.3.1",), "Structure"),
    RuleSpec("aria-roles", 11, RULE_OWNER_DOM, ("4.1.2",), "ARIA"),
    RuleSpec("aria-toggle-field-name", 12, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Name, role, value"),
    RuleSpec("aria-tooltip-name", 13, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Name, role, value"),
    RuleSpec("aria-valid-attr", 14, RULE_OWNER_DOM, ("4.1.2",), "ARIA"),
    RuleSpec("aria-valid-attr-value", 15, RULE_OWNER_DOM, ("4.1.2",), "ARIA"),
    RuleSpec("autocomplete-valid", 16, RULE_OWNER_DOM, ("1.3.5",), "Forms"),
    RuleSpec("avoid-inline-spacing", 17, RULE_OWNER_RENDERED_VISUAL, ("1.4.12",), "Visual"),
    RuleSpec("blink", 18, RULE_OWNER_DOM_PAGE, ("2.2.2",), "Time and media"),
    RuleSpec("button-name", 19, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Name, role, value"),
    RuleSpec("color-contrast", 20, RULE_OWNER_RENDERED_VISUAL, ("1.4.3",), "Visual"),
    RuleSpec("definition-list", 21, RULE_OWNER_DOM, ("1.3.1",), "Structure"),
    RuleSpec("dlitem", 22, RULE_OWNER_DOM, ("1.3.1",), "Structure"),
    RuleSpec("document-title", 23, RULE_OWNER_DOM_PAGE, ("2.4.2",), "Page metadata"),
    RuleSpec("frame-title", 24, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Text alternatives"),
    RuleSpec("html-has-lang", 25, RULE_OWNER_DOM_PAGE, ("3.1.1",), "Language"),
    RuleSpec("html-lang-valid", 26, RULE_OWNER_DOM_PAGE, ("3.1.1",), "Language"),
    RuleSpec("image-alt", 27, RULE_OWNER_A11Y_TREE, ("1.1.1",), "Text alternatives"),
    RuleSpec("input-button-name", 28, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Name, role, value"),
    RuleSpec("input-image-alt", 29, RULE_OWNER_A11Y_TREE, ("1.1.1", "4.1.2"), "Text alternatives"),
    RuleSpec("label", 30, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Forms"),
    RuleSpec("link-in-text-block", 31, RULE_OWNER_RENDERED_VISUAL, ("1.4.1",), "Visual"),
    RuleSpec("link-name", 32, RULE_OWNER_A11Y_TREE, ("2.4.4", "4.1.2"), "Name, role, value"),
    RuleSpec("list", 33, RULE_OWNER_DOM, ("1.3.1",), "Structure"),
    RuleSpec("listitem", 34, RULE_OWNER_DOM, ("1.3.1",), "Structure"),
    RuleSpec("marquee", 35, RULE_OWNER_DOM_PAGE, ("2.2.2",), "Time and media"),
    RuleSpec("meta-refresh", 36, RULE_OWNER_DOM_PAGE, ("2.2.1",), "Page metadata"),
    RuleSpec("meta-viewport", 37, RULE_OWNER_RENDERED_VISUAL, ("1.4.4",), "Visual"),
    RuleSpec("nested-interactive", 38, RULE_OWNER_DOM, ("4.1.2",), "Interaction"),
    RuleSpec("object-alt", 39, RULE_OWNER_A11Y_TREE, ("1.1.1",), "Text alternatives"),
    RuleSpec("role-img-alt", 40, RULE_OWNER_A11Y_TREE, ("1.1.1",), "Text alternatives"),
    RuleSpec("scrollable-region-focusable", 41, RULE_OWNER_RENDERED_VISUAL, ("2.1.1", "2.1.3"), "Keyboard"),
    RuleSpec("select-name", 42, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Forms"),
    RuleSpec("summary-name", 43, RULE_OWNER_A11Y_TREE, ("4.1.2",), "Name, role, value"),
    RuleSpec("svg-img-alt", 44, RULE_OWNER_A11Y_TREE, ("1.1.1",), "Text alternatives"),
    RuleSpec("valid-lang", 45, RULE_OWNER_DOM_PAGE, ("3.1.2",), "Language"),
)


RULE_INDEX: Dict[str, int] = {spec.axe_rule_id: spec.rule_index for spec in RULE_SPECS}
INDEX_TO_RULE: Dict[int, str] = {spec.rule_index: spec.axe_rule_id for spec in RULE_SPECS}
RULE_BY_ID: Dict[str, RuleSpec] = {spec.axe_rule_id: spec for spec in RULE_SPECS}
RULE_BY_INDEX: Dict[int, RuleSpec] = {spec.rule_index: spec for spec in RULE_SPECS}
NUM_RULES = len(RULE_SPECS)

RULE_CATEGORIES = {spec.axe_rule_id: spec.category for spec in RULE_SPECS}

OWNER_TO_RULE_IDS: Dict[str, Tuple[str, ...]] = {
    owner: tuple(spec.axe_rule_id for spec in RULE_SPECS if spec.owner == owner)
    for owner in {
        RULE_OWNER_A11Y_TREE,
        RULE_OWNER_DOM,
        RULE_OWNER_DOM_PAGE,
        RULE_OWNER_RENDERED_VISUAL,
    }
}

GRAPH_SOURCE_TO_RULE_OWNERS: Dict[str, Tuple[str, ...]] = {
    "a11y-tree": (RULE_OWNER_A11Y_TREE,),
    "dom": (RULE_OWNER_DOM, RULE_OWNER_DOM_PAGE),
    "rendered-visual": (RULE_OWNER_RENDERED_VISUAL,),
}


def rule_spec(rule_id: str) -> RuleSpec | None:
    return RULE_BY_ID.get(rule_id)


def rule_owner(rule_id: str) -> str | None:
    spec = rule_spec(rule_id)
    return spec.owner if spec else None


def rule_wcag_ids(rule_id: str) -> Tuple[str, ...]:
    spec = rule_spec(rule_id)
    return spec.wcag_ids if spec else ()


def rule_ids_for_owners(owners: Iterable[str]) -> Tuple[str, ...]:
    owner_set = set(owners)
    return tuple(spec.axe_rule_id for spec in RULE_SPECS if spec.owner in owner_set)


def rule_indices_for_owners(owners: Iterable[str]) -> Tuple[int, ...]:
    owner_set = set(owners)
    return tuple(spec.rule_index for spec in RULE_SPECS if spec.owner in owner_set)


def rule_indices_for_graph_source(graph_source: str) -> Tuple[int, ...]:
    return rule_indices_for_owners(GRAPH_SOURCE_TO_RULE_OWNERS.get(graph_source, ()))


def rule_ids_for_graph_source(graph_source: str) -> Tuple[str, ...]:
    return rule_ids_for_owners(GRAPH_SOURCE_TO_RULE_OWNERS.get(graph_source, ()))


def rule_mask_for_indices(indices: Iterable[int], *, num_rules: int = NUM_RULES) -> torch.Tensor:
    mask = torch.zeros(num_rules, dtype=torch.bool)
    for idx in indices:
        mask[int(idx)] = True
    return mask


def rule_mask_for_graph_source(graph_source: str) -> torch.Tensor:
    return rule_mask_for_indices(rule_indices_for_graph_source(graph_source))


def graph_source_for_rule(rule_id: str) -> str | None:
    owner = rule_owner(rule_id)
    if owner == RULE_OWNER_A11Y_TREE:
        return "a11y-tree"
    if owner in {RULE_OWNER_DOM, RULE_OWNER_DOM_PAGE}:
        return "dom"
    if owner == RULE_OWNER_RENDERED_VISUAL:
        return "rendered-visual"
    return None

