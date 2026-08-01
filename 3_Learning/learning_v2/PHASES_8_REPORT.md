# Phase 8 Execution Report

Executed: 2026-08-01  
Status: weak-label retrieval pilot  
Split hash: `b2c549d3c46587c028cb71788d410cf98c276f6bc02561d46f353f3c2df1adb6`

## Outcome

Phase 8 is implemented and executed as a retrieval-first experiment. It does
not generate or apply repairs. The implementation uses the term Graph-RAG only
for the condition that traverses a versioned typed graph before ranking.

The index contains:

- 9 versioned W3C knowledge records;
- 857 axe-derived training exemplars from 31 positive training sites;
- explicit criterion, technique/failure, detector-rule, context-pattern,
  repair-pattern, validation-requirement, and provenance links;
- no validation or test sites and no query-template hashes.

The evaluation contains 21 held-out test queries over the five Phase 5 rules,
with a shared top-5 and 5,000-character context budget.

## Retrieval results

| Condition | Recall@5 | MRR | nDCG@5 | Source correctness | Traceability | Leakage |
|---|---:|---:|---:|---:|---:|---:|
| No RAG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Flat TF-IDF RAG | 0.0476 | 0.0095 | 0.0184 | 0.7619 | 1.0000 | 0 |
| Graph-constrained RAG | 0.0952 | 0.0571 | 0.0660 | 1.0000 | 1.0000 | 0 |

Graph-constrained retrieval changed 11 of 21 top-5 lists. It doubled Recall@5, increased MRR and nDCG, and removed criterion/rule-incompatible sources from the retrieved context. This establishes that the graph-aware condition is operationally different from flat vector retrieval.

The absolute Recall@5 values remain low. The large training-exemplar pool often
occupies the small context budget before a curated W3C gold record. This result
is retained rather than tuning retrieval on held-out queries. Future tuning
must use training/validation retrieval queries only.

## Leakage and grounding controls

- every exemplar is labelled `train`;
- query sites are absent from the index;
- matching HTML template hashes are absent from the index;
- retrieval also filters query site/template matches at inference time;
- every knowledge record has a versioned W3C URL;
- every exemplar retains source site and template hash;
- generator inputs reproduce the exact finding evidence and retrieved record
  IDs;
- a failed RAG lookup returns `leave_finding_unchanged`;
- prompts propose only and never apply a patch.

## Exit gate

All Phase 8 structural gates pass:

- held-out test data is absent from the retrieval index;
- sources are traceable;
- graph retrieval is meaningfully different from flat retrieval;
- both retrieval conditions share the same corpus, scorer, top-k, and context
  budget;
- leakage checks report zero violations.

The scientific result remains preliminary because the queries and relevance
labels are derived from axe and the curated graph links. Independent verified
repair cases and blinded contextual-quality assessment are deferred to the
Phase 9–10 repair study.

## Artifacts

- `artifacts_3107_0015/phase_8/index_manifest.json`
- `artifacts_3107_0015/phase_8/retrieval_corpus.json`
- `artifacts_3107_0015/phase_8/knowledge_graph.json`
- `artifacts_3107_0015/phase_8/queries.json`
- `artifacts_3107_0015/phase_8/retrieval_results.json`
- `artifacts_3107_0015/phase_8/generator_inputs.json`
- `artifacts_3107_0015/phase_8/phase_8_retrieval_evaluation.json`
- `artifacts_3107_0015/phase_8/run_manifest.json`
