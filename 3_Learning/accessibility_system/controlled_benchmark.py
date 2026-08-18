"""Build an independently specified, non-leaking controlled LLM repair benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .retrieval.contracts import RetrievalQuery
from .retrieval.evaluation import evaluate_retriever
from .retrieval.index import RetrievalIndex
from .retrieval.knowledge import load_knowledge
from .retrieval.prompts import build_generator_input
from .retrieval.retrievers import FlatVectorRetriever, GraphConstrainedRetriever, NoRetrieval, RetrievalBudget


def _operation(operation: str, selector: str, *, attribute_name=None, css_property=None, new_value=None) -> dict:
    return {"operation": operation, "selector": selector, "attribute_name": attribute_name, "css_property": css_property, "new_value": new_value}


CASES = (
    {
        "name": "english-language", "criterion_id": "3.1.1", "rule_id": "html-has-lang", "context": "document-metadata",
        "selector": "html", "html": "<!doctype html><html><head><title>English help</title></head><body><main><h1>Account help</h1><p>This page is written in English.</p></main></body></html>",
        "evidence": "The page heading and all prose are independently verified as English; html has no lang attribute.",
        "oracle": [_operation("set_attribute", "html", attribute_name="lang", new_value="en")], "semantic_verified": True,
    },
    {
        "name": "save-button", "criterion_id": "4.1.2", "rule_id": "button-name", "context": "button",
        "selector": "#save", "html": "<!doctype html><html lang='en'><head><title>Editor</title></head><body><main><h1>Editor</h1><p id='purpose'>Use the following control to save changes.</p><button id='save'><svg aria-hidden='true'></svg></button></main></body></html>",
        "evidence": "The independently verified control purpose is Save changes; button #save has no accessible name.",
        "oracle": [_operation("set_attribute", "#save", attribute_name="aria-label", new_value="Save changes")], "semantic_verified": True,
    },
    {
        "name": "email-label", "criterion_id": "4.1.2", "rule_id": "label", "context": "form-control",
        "selector": "#email", "html": "<!doctype html><html lang='en'><head><title>Contact</title></head><body><main><h1>Contact</h1><p>Enter your email address.</p><input id='email' type='email'></main></body></html>",
        "evidence": "The independently verified visible label is Email address; input #email has no programmatic label.",
        "oracle": [_operation("insert_label_before", "#email", new_value="Email address")], "semantic_verified": True,
    },
    {
        "name": "documentation-link", "criterion_id": "2.4.4", "rule_id": "link-name", "context": "link",
        "selector": "#docs", "html": "<!doctype html><html lang='en'><head><title>Product</title></head><body><main><h1>Product</h1><p>The following icon opens the documentation.</p><a id='docs' href='/documentation'><svg aria-hidden='true'></svg></a></main></body></html>",
        "evidence": "The independently verified link purpose is Documentation; a#docs has no accessible name.",
        "oracle": [_operation("set_attribute", "#docs", attribute_name="aria-label", new_value="Documentation")], "semantic_verified": True,
    },
    {
        "name": "informative-image", "criterion_id": "1.1.1", "rule_id": "image-alt", "context": "informative-image",
        "selector": "#status", "html": "<!doctype html><html lang='en'><head><title>System status</title></head><body><main><h1>System status</h1><figure><img id='status' src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22/%3E'><figcaption>All systems operational</figcaption></figure></main></body></html>",
        "evidence": "The independently verified image meaning, also stated by its figure caption, is All systems operational.",
        "oracle": [_operation("set_attribute", "#status", attribute_name="alt", new_value="All systems operational")], "semantic_verified": True,
    },
    {
        "name": "text-contrast", "criterion_id": "1.4.3", "rule_id": "color-contrast", "context": "rendered-text",
        "selector": "#low", "html": "<!doctype html><html lang='en'><head><title>Contrast</title></head><body><main><h1>Contrast</h1><p id='low' style='color: #777777; background-color: #888888'>Important account status</p></main></body></html>",
        "evidence": "Computed foreground #777777 on background #888888 fails 4.5:1; #000000 is independently verified against the fixed #888888 background.",
        "oracle": [_operation("set_style_property", "#low", css_property="color", new_value="#000000")], "visual_verified": True,
    },
)


def run(args: argparse.Namespace) -> dict:
    version, knowledge = load_knowledge(args.knowledge)
    index = RetrievalIndex.build(version, knowledge, [])
    budget = RetrievalBudget(args.top_k, args.context_characters)
    retrievers = [NoRetrieval(), FlatVectorRetriever(index), GraphConstrainedRetriever(index)]
    corpus = args.output_dir / "corpus"; corpus.mkdir(parents=True, exist_ok=True)
    queries = []
    truth_cases = []
    for case in CASES:
        site_id = f"controlled-{case['name']}"; site_dir = corpus / site_id; site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "0.html").write_text(case["html"], encoding="utf-8")
        query_id = "controlled-" + hashlib.sha256(case["name"].encode()).hexdigest()[:16]
        relevant = tuple(record.record_id for record in knowledge if case["criterion_id"] in record.criterion_ids and case["rule_id"] in record.rule_ids)
        finding_id = hashlib.sha256(f"{site_id}|{case['rule_id']}|{case['selector']}".encode()).hexdigest()[:20]
        finding = {
            "finding_id": finding_id, "site_id": site_id, "criterion_id": case["criterion_id"],
            "rule_id": case["rule_id"], "status": "verified_fail",
            "evidence": {"target": [case["selector"]], "html": case["html"], "failure_summary": case["evidence"]},
        }
        queries.append(RetrievalQuery(
            query_id=query_id, site_id=site_id, template_hash=hashlib.sha256(case["html"].encode()).hexdigest(),
            criterion_id=case["criterion_id"], rule_id=case["rule_id"], context_pattern=case["context"],
            evidence_text=case["evidence"], relevant_record_ids=relevant, finding=finding,
        ))
        truth_cases.append({
            "query_id": query_id, "status": "verified_fail", "semantic_verified": bool(case.get("semantic_verified")),
            "visual_verified": bool(case.get("visual_verified")), "oracle_operations": case["oracle"],
        })
    metrics = {}; generator_inputs = []; outputs = {}
    for retriever in retrievers:
        result, retrieved = evaluate_retriever(retriever, queries, budget, set())
        metrics[retriever.name] = result
        outputs[retriever.name] = {query.query_id: [record.to_dict() for record in retrieved[query.query_id]] for query in queries}
        generator_inputs.extend(build_generator_input(retriever.name, query, retrieved[query.query_id]) for query in queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "queries.json").write_text(json.dumps([query.to_dict() for query in queries], indent=2), encoding="utf-8")
    (args.output_dir / "generator_inputs.json").write_text(json.dumps(generator_inputs, indent=2), encoding="utf-8")
    (args.output_dir / "repair_truth.json").write_text(json.dumps({"schema_version": 1, "cases": truth_cases}, indent=2), encoding="utf-8")
    (args.output_dir / "retrieval_results.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    report = {
        "schema_version": 1, "status": "independent_controlled_repair_benchmark", "query_count": len(queries),
        "conditions": [retriever.name for retriever in retrievers], "budget": {"top_k": args.top_k, "context_characters": args.context_characters},
        "metrics": metrics, "oracle_separation": "repair_truth.json is passed only to sandbox validation and excluded from generator inputs",
    }
    (args.output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", type=Path, default=Path(__file__).resolve().parent / "knowledge" / "records.v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5); parser.add_argument("--context-characters", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args()); print(json.dumps({"status": report["status"], "queries": report["query_count"]}, indent=2))


if __name__ == "__main__":
    main()
