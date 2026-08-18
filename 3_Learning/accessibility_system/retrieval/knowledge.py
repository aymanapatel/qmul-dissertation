"""Knowledge-record loading, validation, graph construction, and exemplars."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from learning_v2.baselines import axe_findings

from .contracts import ExemplarRecord, KnowledgeRecord


def load_knowledge(path: Path) -> tuple[str, list[KnowledgeRecord]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("knowledge_version"):
        raise ValueError("Unsupported or unversioned knowledge registry")
    records = [KnowledgeRecord.from_dict(item) for item in payload.get("records", [])]
    if not records or len({record.record_id for record in records}) != len(records):
        raise ValueError("Knowledge records must have unique IDs")
    for record in records:
        if not record.criterion_ids or not record.repair_pattern_id or not record.validation_requirements:
            raise ValueError(f"Incomplete graph linkage in {record.record_id}")
        if record.provenance.publisher != "W3C WAI" or not record.provenance.url.startswith("https://www.w3.org/"):
            raise ValueError(f"Untraceable provenance in {record.record_id}")
    return str(payload["knowledge_version"]), records


def _context_from_finding(finding) -> str:
    html = str(finding.evidence.get("html", ""))
    match = re.search(r"<\s*([a-zA-Z0-9-]+)", html)
    tag = match.group(1).lower() if match else (finding.node.tag if finding.node else "unknown")
    return {
        "a": "link", "img": "img", "input": "form-control", "textarea": "form-control",
        "select": "form-control", "html": "document-metadata",
    }.get(tag, "rendered-text" if finding.rule_id == "color-contrast" else tag)


def build_training_exemplars(
    corpus_dir: Path,
    train_sites: Iterable[str],
    inventory: dict,
    knowledge: list[KnowledgeRecord],
    *,
    allowed_rules: set[str] | None = None,
) -> list[ExemplarRecord]:
    site_rows = {row["site_id"]: row for row in inventory["sites"]}
    repair_by_rule_criterion = {}
    for record in knowledge:
        for rule in record.rule_ids:
            for criterion in record.criterion_ids:
                repair_by_rule_criterion.setdefault((rule, criterion), record.repair_pattern_id)
    exemplars = []; seen = set()
    for site in sorted(train_sites):
        if site not in site_rows:
            continue
        for finding in axe_findings(corpus_dir / site):
            if allowed_rules is not None and finding.rule_id not in allowed_rules:
                continue
            context = _context_from_finding(finding)
            evidence = " ".join((
                str(finding.evidence.get("failure_summary", "")),
                str(finding.evidence.get("html", "")),
                " ".join(str(value) for value in finding.evidence.get("target", [])),
            )).strip()[:1200]
            for criterion in finding.criterion_ids:
                repair = repair_by_rule_criterion.get((finding.rule_id, criterion))
                if not repair:
                    continue
                identity = (site, finding.rule_id, criterion, context, evidence)
                if identity in seen:
                    continue
                seen.add(identity)
                raw_id = "|".join(identity)
                exemplars.append(ExemplarRecord(
                    record_id=f"ex-{hashlib.sha256(raw_id.encode()).hexdigest()[:20]}",
                    source_site=site,
                    template_hash=site_rows[site]["html_sha256"],
                    rule_id=finding.rule_id,
                    criterion_ids=(criterion,),
                    context_pattern=context,
                    evidence_text=evidence,
                    repair_pattern_id=repair,
                ))
    return exemplars


class KnowledgeGraph:
    """Small typed property graph represented by explicit nodes and edges."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add_node(self, node_id: str, node_type: str, **attributes) -> None:
        self.nodes[node_id] = {"node_id": node_id, "node_type": node_type, **attributes}

    def add_edge(self, source: str, target: str, relation: str) -> None:
        item = {"source": source, "target": target, "relation": relation}
        if item in self.edges:
            return
        self.edges.append(item)
        self.adjacency[source].append((target, relation)); self.adjacency[target].append((source, relation))

    def paths_to_documents(self, starts: Iterable[str], max_hops: int = 2) -> dict[str, tuple[str, ...]]:
        queue = deque((node, (node,)) for node in starts if node in self.nodes)
        seen = {node for node, _ in queue}; found = {}
        while queue:
            current, path = queue.popleft()
            if self.nodes[current]["node_type"] in {"technique", "failure", "training_exemplar"}:
                found[current.removeprefix("doc:")] = path
                continue
            if len(path) - 1 >= max_hops:
                continue
            for target, _ in sorted(self.adjacency.get(current, [])):
                if target not in seen:
                    seen.add(target); queue.append((target, (*path, target)))
        return found

    def to_dict(self) -> dict:
        return {"schema_version": 1, "nodes": list(sorted(self.nodes.values(), key=lambda item: item["node_id"])), "edges": sorted(self.edges, key=lambda item: (item["source"], item["relation"], item["target"]))}


def build_graph(knowledge: list[KnowledgeRecord], exemplars: list[ExemplarRecord]) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for record in knowledge:
        document = f"doc:{record.record_id}"; graph.add_node(document, record.record_type, record_version=record.version)
        provenance = f"provenance:{record.record_id}"; graph.add_node(provenance, "provenance", url=record.provenance.url, version=record.version)
        graph.add_edge(document, provenance, "cites")
        repair = f"repair:{record.repair_pattern_id}"; graph.add_node(repair, "repair_pattern")
        graph.add_edge(document, repair, "recommends")
        for criterion in record.criterion_ids:
            node = f"criterion:{criterion}"; graph.add_node(node, "criterion"); graph.add_edge(document, node, "supports")
        for rule in record.rule_ids:
            node = f"rule:{rule}"; graph.add_node(node, "detector_rule"); graph.add_edge(document, node, "addresses")
        for context in record.context_patterns:
            node = f"context:{context}"; graph.add_node(node, "context_pattern"); graph.add_edge(document, node, "applies_to")
        for validation in record.validation_requirements:
            node = f"validation:{validation}"; graph.add_node(node, "validation_requirement"); graph.add_edge(repair, node, "requires")
    for exemplar in exemplars:
        document = f"doc:{exemplar.record_id}"; graph.add_node(document, "training_exemplar", source_site=exemplar.source_site)
        graph.add_edge(document, f"rule:{exemplar.rule_id}", "demonstrates")
        for criterion in exemplar.criterion_ids:
            graph.add_edge(document, f"criterion:{criterion}", "labelled_as")
        graph.add_edge(document, f"context:{exemplar.context_pattern}", "observed_in")
        graph.add_edge(document, f"repair:{exemplar.repair_pattern_id}", "routes_to")
    return graph
