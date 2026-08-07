import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wcag_rules import (  # noqa: E402
    NUM_RULES,
    RULE_INDEX,
    RULE_SPECS,
    graph_source_for_rule,
    rule_indices_for_graph_source,
    rule_wcag_ids,
)


def test_every_rule_has_one_owner_and_wcag_id():
    assert len(RULE_SPECS) == NUM_RULES
    assert len(RULE_INDEX) == NUM_RULES

    seen_indices = set()
    for spec in RULE_SPECS:
        assert spec.owner
        assert spec.wcag_ids
        assert spec.rule_index not in seen_indices
        seen_indices.add(spec.rule_index)


def test_rendered_rules_do_not_leak_into_a11y_tree_mask():
    a11y_indices = set(rule_indices_for_graph_source("a11y-tree"))
    for rule_id in ("color-contrast", "link-in-text-block", "avoid-inline-spacing"):
        assert RULE_INDEX[rule_id] not in a11y_indices
        assert graph_source_for_rule(rule_id) == "rendered-visual"


def test_key_rules_have_expected_wcag_ids():
    assert rule_wcag_ids("area-alt") == ("1.1.1", "2.4.4", "4.1.2")
    assert rule_wcag_ids("image-alt") == ("1.1.1",)
    assert rule_wcag_ids("link-name") == ("2.4.4", "4.1.2")
    assert rule_wcag_ids("list") == ("1.3.1",)
    assert rule_wcag_ids("color-contrast") == ("1.4.3",)
