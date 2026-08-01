"""Retrieval-first evaluation, including source, diversity, and leakage checks."""

from __future__ import annotations

import math
import time

from .contracts import RetrievalQuery, RetrievedRecord
from .retrievers import RetrievalBudget


def _query_metrics(query: RetrievalQuery, records: list[RetrievedRecord], heldout_sites: set[str]) -> dict:
    relevant = set(query.relevant_record_ids)
    retrieved = [record.record_id for record in records]
    hits = [int(record_id in relevant) for record_id in retrieved]
    recall = sum(hits) / len(relevant) if relevant else 0.0
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), len(records)) + 1))
    source_correct = [
        int(query.criterion_id in record.criterion_ids and query.rule_id in record.rule_ids)
        for record in records
    ]
    traceable = [int(bool(record.citation_url) or bool(record.source_site and record.template_hash)) for record in records]
    leakage = [
        record.record_id for record in records
        if record.source_site == query.site_id
        or (record.template_hash is not None and record.template_hash == query.template_hash)
        or (record.source_site is not None and record.source_site in heldout_sites)
    ]
    return {
        "query_id": query.query_id, "retrieved_ids": retrieved,
        "recall_at_k": recall, "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
        "source_correctness": sum(source_correct) / len(source_correct) if source_correct else 0.0,
        "traceability": sum(traceable) / len(traceable) if traceable else 1.0,
        "record_type_diversity": len({record.record_type for record in records}) / len(records) if records else 0.0,
        "repair_pattern_diversity": len({record.repair_pattern_id for record in records}) / len(records) if records else 0.0,
        "leakage_record_ids": leakage,
        "empty": not records,
    }


def evaluate_retriever(retriever, queries: list[RetrievalQuery], budget: RetrievalBudget, heldout_sites: set[str]) -> tuple[dict, dict[str, list[RetrievedRecord]]]:
    per_query = []; outputs = {}; started = time.perf_counter()
    for query in queries:
        records = retriever.retrieve(query, budget); outputs[query.query_id] = records
        per_query.append(_query_metrics(query, records, heldout_sites))
    latency = time.perf_counter() - started
    mean = lambda key: sum(item[key] for item in per_query) / len(per_query) if per_query else 0.0
    metrics = {
        "condition": retriever.name, "query_count": len(queries), "top_k": budget.top_k,
        "context_character_budget": budget.context_characters,
        "mean_recall_at_k": mean("recall_at_k"), "mrr": mean("reciprocal_rank"),
        "mean_ndcg_at_k": mean("ndcg_at_k"), "source_correctness": mean("source_correctness"),
        "traceability": mean("traceability"), "record_type_diversity": mean("record_type_diversity"),
        "repair_pattern_diversity": mean("repair_pattern_diversity"),
        "empty_retrieval_rate": sum(item["empty"] for item in per_query) / max(1, len(per_query)),
        "leakage_count": sum(len(item["leakage_record_ids"]) for item in per_query),
        "latency_seconds": latency, "mean_latency_ms": latency * 1000 / max(1, len(queries)),
        "per_query": per_query,
    }
    return metrics, outputs
