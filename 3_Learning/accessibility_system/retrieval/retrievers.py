"""No-retrieval, flat-vector, and graph-constrained retrieval conditions."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics.pairwise import cosine_similarity

from .contracts import ExemplarRecord, KnowledgeRecord, RetrievalQuery, RetrievedRecord
from .index import RetrievalIndex


@dataclass(frozen=True)
class RetrievalBudget:
    top_k: int = 5
    context_characters: int = 5000


class NoRetrieval:
    name = "no_rag"

    def retrieve(self, query: RetrievalQuery, budget: RetrievalBudget) -> list[RetrievedRecord]:
        return []


class _BaseRetriever:
    def __init__(self, index: RetrievalIndex):
        self.index = index

    def candidate_paths(self, query: RetrievalQuery) -> dict[str, tuple[str, ...]]:
        raise NotImplementedError

    def retrieve(self, query: RetrievalQuery, budget: RetrievalBudget) -> list[RetrievedRecord]:
        paths = self.candidate_paths(query)
        candidate_ids = [record_id for record_id in self.index.document_ids if record_id in paths]
        # Defence in depth: query sites and near-template exemplars are filtered
        # even though they should never enter a valid training-only index.
        candidate_ids = [
            record_id for record_id in candidate_ids
            if not isinstance(self.index.documents[record_id], ExemplarRecord)
            or (
                self.index.documents[record_id].source_site != query.site_id
                and self.index.documents[record_id].template_hash != query.template_hash
            )
        ]
        if not candidate_ids:
            return []
        query_vector = self.index.vectorizer.transform([query.searchable_text()])
        rows = [self.index.row_by_id[record_id] for record_id in candidate_ids]
        similarities = cosine_similarity(query_vector, self.index.matrix[rows]).flatten()
        ranked = sorted(zip(candidate_ids, similarities), key=lambda item: (-float(item[1]), item[0]))[:budget.top_k]
        remaining = budget.context_characters; results = []
        for rank, (record_id, score) in enumerate(ranked, 1):
            record = self.index.documents[record_id]
            if isinstance(record, KnowledgeRecord):
                content = f"{record.title}. {record.summary} Repair: {record.repair_summary} Validation: {', '.join(record.validation_requirements)}"
                citation = record.provenance.url; source_site = None; template_hash = None
                criteria, rules = record.criterion_ids, record.rule_ids
            else:
                content = f"Training exemplar {record.record_id}: {record.evidence_text}"
                citation = None; source_site = record.source_site; template_hash = record.template_hash
                criteria, rules = record.criterion_ids, (record.rule_id,)
            content = content[:remaining]; remaining -= len(content)
            if not content:
                break
            results.append(RetrievedRecord(
                record_id=record_id, score=round(float(score), 8), rank=rank,
                record_type=record.record_type, criterion_ids=criteria, rule_ids=rules,
                repair_pattern_id=record.repair_pattern_id, citation_url=citation,
                source_site=source_site, template_hash=template_hash,
                graph_path=paths.get(record_id, ()), content=content,
            ))
        return results


class FlatVectorRetriever(_BaseRetriever):
    name = "flat_vector_rag"

    def candidate_paths(self, query: RetrievalQuery) -> dict[str, tuple[str, ...]]:
        return {record_id: () for record_id in self.index.document_ids}


class GraphConstrainedRetriever(_BaseRetriever):
    name = "graph_constrained_rag"

    def candidate_paths(self, query: RetrievalQuery) -> dict[str, tuple[str, ...]]:
        criterion_paths = self.index.graph.paths_to_documents([f"criterion:{query.criterion_id}"], max_hops=1)
        rule_paths = self.index.graph.paths_to_documents([f"rule:{query.rule_id}"], max_hops=1)
        context_paths = self.index.graph.paths_to_documents([f"context:{query.context_pattern}"], max_hops=1)
        # Criterion and detector rule are hard graph constraints. Context
        # narrows that intersection when possible, never substituting for it.
        candidates = set(criterion_paths) & set(rule_paths)
        if not candidates:
            candidates = set(criterion_paths) or set(rule_paths)
        preferred = candidates & set(context_paths)
        if preferred:
            candidates = preferred
        return {
            record_id: context_paths.get(record_id, criterion_paths.get(record_id, rule_paths.get(record_id, ())))
            for record_id in sorted(candidates)
        }
