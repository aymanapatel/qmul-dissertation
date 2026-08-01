"""Build and evaluate the Phase 8 knowledge/retrieval layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from learning_v2.baselines import axe_findings

from .retrieval.contracts import RetrievalQuery
from .retrieval.evaluation import evaluate_retriever
from .retrieval.index import RetrievalIndex, sha256_file
from .retrieval.knowledge import build_training_exemplars, load_knowledge
from .retrieval.prompts import build_generator_input
from .retrieval.retrievers import FlatVectorRetriever, GraphConstrainedRetriever, NoRetrieval, RetrievalBudget


def _context(finding) -> str:
    html = str(finding.evidence.get("html", ""))
    match = re.search(r"<\s*([a-zA-Z0-9-]+)", html); tag = match.group(1).lower() if match else "unknown"
    return {"a": "link", "img": "img", "input": "form-control", "textarea": "form-control", "select": "form-control", "html": "document-metadata"}.get(
        tag, "rendered-text" if finding.rule_id == "color-contrast" else tag,
    )


def build_queries(corpus_dir: Path, test_sites: list[str], inventory: dict, knowledge, supported_rules: set[str], max_queries: int) -> list[RetrievalQuery]:
    site_rows = {item["site_id"]: item for item in inventory["sites"]}; queries = []; seen = set()
    for site in sorted(test_sites):
        for finding in axe_findings(corpus_dir / site):
            if finding.rule_id not in supported_rules:
                continue
            context = _context(finding)
            evidence = " ".join((
                str(finding.evidence.get("failure_summary", "")), str(finding.evidence.get("html", "")),
                " ".join(str(value) for value in finding.evidence.get("target", [])),
            )).strip()[:1600]
            for criterion in finding.criterion_ids:
                gold = tuple(sorted(
                    record.record_id for record in knowledge
                    if criterion in record.criterion_ids and finding.rule_id in record.rule_ids
                ))
                if not gold:
                    continue
                identity = (site, criterion, finding.rule_id, context)
                if identity in seen:
                    continue
                seen.add(identity)
                query_id = "q-" + hashlib.sha256("|".join(identity).encode()).hexdigest()[:20]
                queries.append(RetrievalQuery(
                    query_id=query_id, site_id=site, template_hash=site_rows[site]["html_sha256"],
                    criterion_id=criterion, rule_id=finding.rule_id, context_pattern=context,
                    evidence_text=evidence, relevant_record_ids=gold,
                    finding={"finding_id": finding.finding_id, "site_id": site, "criterion_id": criterion,
                             "rule_id": finding.rule_id, "status": "weak_label_fail", "evidence": finding.evidence},
                ))
    return queries[:max_queries]


def run(args: argparse.Namespace) -> dict:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    comparison = json.loads((args.phase5_dir / "comparison.json").read_text(encoding="utf-8"))
    runs = comparison.get("results", comparison.get("runs", []))
    supported_rules = {rule for model_run in runs for rule in model_run["rules"]}
    if not supported_rules:
        raise ValueError("Phase 5 comparison contains no supported rules")
    knowledge_version, knowledge = load_knowledge(args.knowledge)
    exemplars = build_training_exemplars(
        args.corpus_dir, split["train"], inventory, knowledge, allowed_rules=supported_rules,
    )
    heldout_sites = set(split["val"]) | set(split["test"])
    leaked_sites = sorted({item.source_site for item in exemplars} & heldout_sites)
    if leaked_sites:
        raise ValueError(f"Held-out sites entered the retrieval index: {leaked_sites[:3]}")
    queries = build_queries(args.corpus_dir, split["test"], inventory, knowledge, supported_rules, args.max_queries)
    query_hashes = {query.template_hash for query in queries}
    template_leaks = sorted({item.template_hash for item in exemplars} & query_hashes)
    if template_leaks:
        raise ValueError("Near-template leakage detected between retrieval index and queries")
    if not queries:
        raise ValueError("No held-out retrieval queries were generated")

    index = RetrievalIndex.build(knowledge_version, knowledge, exemplars)
    inputs = {
        "knowledge": str(args.knowledge), "knowledge_sha256": sha256_file(args.knowledge),
        "split": str(args.split), "split_sha256": sha256_file(args.split), "split_hash": split.get("split_hash"),
        "inventory": str(args.inventory), "inventory_sha256": sha256_file(args.inventory),
        "phase5": str(args.phase5_dir), "corpus": str(args.corpus_dir),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True); index.save(args.output_dir, inputs=inputs)
    budget = RetrievalBudget(args.top_k, args.context_characters)
    retrievers = [NoRetrieval(), FlatVectorRetriever(index), GraphConstrainedRetriever(index)]
    metrics = {}; all_outputs = {}; prompts = []
    for retriever in retrievers:
        result, outputs = evaluate_retriever(retriever, queries, budget, set(split["test"]))
        metrics[retriever.name] = result
        all_outputs[retriever.name] = {
            query_id: [record.to_dict() for record in records] for query_id, records in outputs.items()
        }
        for query in queries:
            prompts.append(build_generator_input(retriever.name, query, outputs[query.query_id]))

    flat_ids = {query_id: tuple(item["record_id"] for item in records) for query_id, records in all_outputs["flat_vector_rag"].items()}
    graph_ids = {query_id: tuple(item["record_id"] for item in records) for query_id, records in all_outputs["graph_constrained_rag"].items()}
    differing = sum(flat_ids[query_id] != graph_ids[query_id] for query_id in flat_ids)
    exit_gate = {
        "heldout_absent_from_index": not leaked_sites,
        "near_templates_absent_from_index": not template_leaks,
        "all_retrieved_sources_traceable": all(metrics[name]["traceability"] == 1.0 for name in ("flat_vector_rag", "graph_constrained_rag")),
        "zero_retrieval_leakage": all(metrics[name]["leakage_count"] == 0 for name in metrics),
        "graph_condition_meaningfully_different": differing > 0,
        "same_corpus_and_budget": True,
    }
    report = {
        "schema_version": 1, "phase": 8, "status": "weak_label_retrieval_pilot",
        "terminology": {"flat_vector_rag": "TF-IDF exemplar/evidence RAG", "graph_constrained_rag": "Graph-RAG using explicit typed-edge traversal before the same TF-IDF scorer"},
        "knowledge_version": knowledge_version, "split_hash": split.get("split_hash"),
        "query_count": len(queries), "knowledge_record_count": len(knowledge),
        "training_exemplar_count": len(exemplars), "index_site_count": len({item.source_site for item in exemplars}),
        "supported_rules": sorted(supported_rules), "budget": {"top_k": args.top_k, "context_characters": args.context_characters},
        "metrics": metrics, "retrieval_lists_differing": differing,
        "exit_gate": exit_gate,
        "limitations": [
            "Queries are axe-derived weak-label findings rather than independently verified repair cases.",
            "Gold relevance is defined from the versioned rule/criterion links in the curated knowledge records.",
            "Phase 8 evaluates retrieval and prompt grounding only; it does not claim generated or validated repair success.",
        ],
    }
    (args.output_dir / "queries.json").write_text(json.dumps([query.to_dict() for query in queries], indent=2), encoding="utf-8")
    (args.output_dir / "retrieval_results.json").write_text(json.dumps(all_outputs, indent=2), encoding="utf-8")
    (args.output_dir / "generator_inputs.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    report_path = args.output_dir / "phase_8_retrieval_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1, "phase": 8, "inputs": inputs,
        "config": {"top_k": args.top_k, "context_characters": args.context_characters, "max_queries": args.max_queries},
        "outputs": {path.name: sha256_file(path) for path in sorted(args.output_dir.glob("*.json")) if path.name != "run_manifest.json"},
        "exit_gate": exit_gate,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--knowledge", type=Path, default=base / "knowledge" / "records.v1.json")
    parser.add_argument("--phase5-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-characters", type=int, default=5000)
    parser.add_argument("--max-queries", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "queries": report["query_count"], "exit_gate": report["exit_gate"]}, indent=2))


if __name__ == "__main__":
    main()
