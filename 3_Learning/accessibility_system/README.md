# Accessibility system — Phase 8 retrieval

This package contains orchestration code that is intentionally separate from
`learning_v2` model training. Phase 8 implements a versioned W3C-backed
knowledge graph, a training-only evidence/exemplar index, and controlled
no-RAG, flat-vector-RAG, and graph-constrained-RAG retrieval conditions.

From `3_Learning`:

```bash
.venv/bin/python -u -m accessibility_system.phase8 \
  --phase5-dir learning_v2/artifacts_3107_0015/phase_5_main \
  --split learning_v2/artifacts_3107_0015/phase_5_main/pilot_split.json \
  --inventory learning_v2/artifacts_3107_0015/phase_1_4/corpus_inventory.json \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --output-dir learning_v2/artifacts_3107_0015/phase_8 \
  --top-k 5 --context-characters 5000 --max-queries 100
```

Both retrieval conditions share the exact same records, TF-IDF representation,
top-k, and context-character budget. Graph-RAG first traverses typed criterion,
detector-rule, and context edges, then applies the same TF-IDF scorer only to
the constrained candidates.

The current queries are axe-derived weak-label findings. Phase 8 evaluates
retrieval and prompt grounding only; it does not generate, apply, or claim the
success of repairs.
