from pathlib import Path

import pytest

from accessibility_system.retrieval.contracts import ExemplarRecord, KnowledgeRecord, Provenance, RetrievalQuery
from accessibility_system.retrieval.evaluation import evaluate_retriever
from accessibility_system.retrieval.index import RetrievalIndex
from accessibility_system.retrieval.knowledge import load_knowledge
from accessibility_system.retrieval.prompts import build_generator_input
from accessibility_system.retrieval.retrievers import FlatVectorRetriever, GraphConstrainedRetriever, NoRetrieval, RetrievalBudget


KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge" / "records.v1.json"


def record(record_id, criterion, rule, text, repair="repair"):
    return KnowledgeRecord(
        record_id=record_id, version="1", record_type="technique", title=text, summary=text,
        criterion_ids=(criterion,), rule_ids=(rule,), context_patterns=("img",),
        repair_pattern_id=repair, repair_summary=text, validation_requirements=("retest",),
        provenance=Provenance("W3C WAI", text, f"https://www.w3.org/{record_id}", "2026-08-01", "1"),
    )


def query(**overrides):
    values = dict(
        query_id="q", site_id="test", template_hash="test-hash", criterion_id="1.1.1",
        rule_id="image-alt", context_pattern="img", evidence_text="missing image alternative",
        relevant_record_ids=("correct",), finding={"finding_id": "f"},
    )
    values.update(overrides)
    return RetrievalQuery(**values)


def test_versioned_w3c_knowledge_is_complete_and_traceable():
    version, records = load_knowledge(KNOWLEDGE)
    assert version
    assert len(records) >= 9
    assert all(item.provenance.url.startswith("https://www.w3.org/") for item in records)
    assert all(item.validation_requirements for item in records)


def test_index_rejects_non_training_exemplar():
    bad = ExemplarRecord("ex", "site", "hash", "image-alt", ("1.1.1",), "img", "evidence", "repair", split="test")
    with pytest.raises(ValueError, match="training-only"):
        RetrievalIndex.build("v1", [record("correct", "1.1.1", "image-alt", "alt")], [bad])


def test_retrievers_exclude_query_site_and_template():
    knowledge = [record("correct", "1.1.1", "image-alt", "image alternative")]
    exemplars = [
        ExemplarRecord("same-site", "test", "other", "image-alt", ("1.1.1",), "img", "missing image alternative", "repair"),
        ExemplarRecord("same-template", "train", "test-hash", "image-alt", ("1.1.1",), "img", "missing image alternative", "repair"),
    ]
    index = RetrievalIndex.build("v1", knowledge, exemplars)
    for retriever in (FlatVectorRetriever(index), GraphConstrainedRetriever(index)):
        ids = {item.record_id for item in retriever.retrieve(query(), RetrievalBudget(top_k=5))}
        assert "same-site" not in ids
        assert "same-template" not in ids


def test_graph_condition_is_a_real_candidate_constraint():
    knowledge = [
        record("correct", "1.1.1", "image-alt", "alternative text for image"),
        record("distractor", "9.9.9", "other-rule", "missing image alternative missing image alternative missing image alternative"),
    ]
    index = RetrievalIndex.build("v1", knowledge, [])
    budget = RetrievalBudget(top_k=1)
    flat = FlatVectorRetriever(index).retrieve(query(), budget)
    graph = GraphConstrainedRetriever(index).retrieve(query(), budget)
    assert flat[0].record_id == "distractor"
    assert graph[0].record_id == "correct"
    assert graph[0].graph_path[0] in {"criterion:1.1.1", "rule:image-alt", "context:img"}


def test_retrieval_metrics_and_safe_prompt_failure():
    index = RetrievalIndex.build("v1", [record("correct", "1.1.1", "image-alt", "missing image alternative")], [])
    budget = RetrievalBudget(top_k=1)
    metrics, outputs = evaluate_retriever(GraphConstrainedRetriever(index), [query()], budget, {"test"})
    assert metrics["mean_recall_at_k"] == 1.0
    assert metrics["leakage_count"] == 0
    assert metrics["traceability"] == 1.0
    grounded = build_generator_input("graph_constrained_rag", query(), outputs["q"])
    assert grounded["citations"][0]["record_id"] == "correct"
    assert "Finding evidence (exact)" in grounded["prompt"]
    failed = build_generator_input("graph_constrained_rag", query(), [])
    assert failed["safe_action"] == "leave_finding_unchanged"
    assert failed["original_finding"] == query().finding
    no_rag_metrics, _ = evaluate_retriever(NoRetrieval(), [query()], budget, {"test"})
    assert no_rag_metrics["mean_recall_at_k"] == 0.0
