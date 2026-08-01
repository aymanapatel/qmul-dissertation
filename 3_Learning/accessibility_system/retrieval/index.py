"""One corpus and one TF-IDF representation shared by both retrievers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from .contracts import ExemplarRecord, KnowledgeRecord
from .knowledge import KnowledgeGraph, build_graph


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RetrievalIndex:
    knowledge_version: str
    knowledge: list[KnowledgeRecord]
    exemplars: list[ExemplarRecord]
    graph: KnowledgeGraph

    def __post_init__(self) -> None:
        self.documents = {record.record_id: record for record in (*self.knowledge, *self.exemplars)}
        self.document_ids = sorted(self.documents)
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([self.documents[record_id].searchable_text() for record_id in self.document_ids])
        self.row_by_id = {record_id: index for index, record_id in enumerate(self.document_ids)}

    @classmethod
    def build(cls, knowledge_version: str, knowledge: list[KnowledgeRecord], exemplars: list[ExemplarRecord]) -> "RetrievalIndex":
        if any(exemplar.split != "train" for exemplar in exemplars):
            raise ValueError("Retrieval exemplars must be training-only")
        return cls(knowledge_version, knowledge, exemplars, build_graph(knowledge, exemplars))

    def save(self, output_dir: Path, *, inputs: dict) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        corpus = {
            "schema_version": 1, "knowledge_version": self.knowledge_version,
            "knowledge": [record.to_dict() for record in self.knowledge],
            "exemplars": [record.to_dict() for record in self.exemplars],
        }
        corpus_path = output_dir / "retrieval_corpus.json"
        corpus_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        graph_path = output_dir / "knowledge_graph.json"
        graph_path.write_text(json.dumps(self.graph.to_dict(), indent=2), encoding="utf-8")
        manifest = {
            "schema_version": 1, "index_type": "tfidf-1-2gram", "knowledge_version": self.knowledge_version,
            "document_count": len(self.document_ids), "knowledge_record_count": len(self.knowledge),
            "training_exemplar_count": len(self.exemplars), "training_sites": sorted({item.source_site for item in self.exemplars}),
            "template_hash_count": len({item.template_hash for item in self.exemplars}),
            "corpus_sha256": sha256_file(corpus_path), "graph_sha256": sha256_file(graph_path), "inputs": inputs,
        }
        (output_dir / "index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
