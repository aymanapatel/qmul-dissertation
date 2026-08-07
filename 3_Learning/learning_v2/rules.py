"""Source-owned rule metadata shared by training and reporting."""

from learning_v1.src.wcag_rules import (
    INDEX_TO_RULE,
    RULE_BY_ID,
    rule_ids_for_graph_source,
    rule_indices_for_graph_source,
)


def rules_for_source(graph_source: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    rule_ids = tuple(rule_ids_for_graph_source(graph_source))
    rule_indices = tuple(rule_indices_for_graph_source(graph_source))
    if not rule_ids or len(rule_ids) != len(rule_indices):
        raise ValueError(f"No consistent rule registry for graph source: {graph_source}")
    return rule_ids, rule_indices


def rule_metadata(rule_id: str) -> dict:
    spec = RULE_BY_ID[rule_id]
    return {
        "rule_id": rule_id,
        "global_rule_index": spec.rule_index,
        "owner": spec.owner,
        "wcag_ids": list(spec.wcag_ids),
        "category": spec.category,
    }


__all__ = ["INDEX_TO_RULE", "rule_metadata", "rules_for_source"]
