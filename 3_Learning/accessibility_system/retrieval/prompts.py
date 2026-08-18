"""Evidence-grounded generator inputs; Phase 8 does not generate or apply patches."""

from __future__ import annotations

from .contracts import RetrievalQuery, RetrievedRecord


def build_generator_input(condition: str, query: RetrievalQuery, records: list[RetrievedRecord]) -> dict:
    citations = [
        {"record_id": record.record_id, "url": record.citation_url, "source_site": record.source_site}
        for record in records
    ]
    retrieval_failed = condition != "no_rag" and not records
    safe_action = "leave_finding_unchanged" if retrieval_failed else "propose_only_do_not_apply"
    context = "\n".join(
        f"[{record.record_id}] {record.content}" for record in records
    ) or "No retrieved context."
    prompt = (
        "You are preparing a typed accessibility repair proposal. Do not apply a patch.\n"
        f"Finding evidence (exact): {query.evidence_text}\n"
        f"Criterion: {query.criterion_id}; detector rule: {query.rule_id}; context: {query.context_pattern}.\n"
        f"Retrieval condition: {condition}.\nRetrieved records:\n{context}\n"
        "Cite record IDs for every recommendation. If purpose, language, visual state, or intent is uncertain, require human review. "
        "Return the original finding unchanged when evidence or retrieved support is insufficient."
    )
    return {
        "schema_version": 1, "query_id": query.query_id, "condition": condition,
        "prompt": prompt, "citations": citations, "safe_action": safe_action,
        "original_finding": query.finding, "retrieval_failed": retrieval_failed,
    }
