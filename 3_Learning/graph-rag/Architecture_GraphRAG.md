# Graph-RAG: LLM-Validated Per-Node Refinement of GNN Predictions

## Goal

Cut down the GNN's false positives (49 on the airindia page) while preserving true positives, by retrieving similar labeled nodes from the 243-graph corpus and asking a local Ollama LLM to confirm / reject / reassign each predicted violation.

The GNN (DOMAttentionNet) is not retrained. RAG runs purely post-hoc on saved prediction JSON files.

## Architecture (data flow)

```
prediction_airindia.json
        |
        +-> (1) Corpus loader: load all 243 graphs from graphs_multi_a11y_v2/
        |         -> flat tensors: node_emb[N_total, D], labels[N_total], rule_multi[N_total, 46]
        |         -> per-rule bucket index: dict[rule_id -> np.array of global_row_idx]
        |         -> outerHTML cache per node for exemplar snippets
        |
        +-> (2) For each predicted-violation node (52 of them):
        |         a. Parse candidate HTML snippet (re-parse file via html_path saved in JSON)
        |         b. For each predicted_rule (up to 3):
        |              retrieve top-k=8 training nodes whose true_label=1 for that rule
        |              retrieve top-k=4 training nodes whose true_label=0 for same rule (negatives)
        |              All retrieval via cosine sim on (text_emb + attr_features)
        |         c. Build Ollama prompt with:
        |              - axe rule description (from rule_descriptions.py)
        |              - candidate HTML snippet (outerHTML, truncated 1KB)
        |              - GNN's probability + top-3 predicted rules
        |              - k exemplars as: <snippet, true_label, true_rule>
        |         d. Ollama call (qwen2.5:7b-instruct, format=json)
        |              Output: {decision, rule_id, confidence, reasoning}
        |         e. Fuse:
        |              - "confirm"  -> keep prediction as-is
        |              - "reject"   -> demote to candidate_warnings (prob unchanged, status flipped)
        |              - "reassign" -> swap the entry's top predicted_rule_id with LLM's rule_id
        |
        +-> (3) Write refined report: prediction_airindia_rag.json with new fields:
                 - rag_decision, rag_assigned_rule, rag_confidence, rag_reasoning
                 - updated summary metrics (predicted_violation_count, candidate_warning_count)
                 - updated ground_truth metrics (precision, recall) computed against axe_summary
```

## Retrieval design

- **Unit:** per-node.
- **Index:** live PyG k-NN (no external vector DB).
  - Corpus flattened to CPU tensors on first run.
  - Cached to `graphs_multi_a11y_v2/_rag_index.pt` to avoid rebuild.
- **Key:** `(rule_id, node_embedding)` where `node_embedding = concat(attr_features, text_embeddings)`.
- **Negatives:** same-rule retrievals are complemented with k_neg confirmed non-violators for the same rule, so the LLM sees both classes.

## LLM validation design

- **Provider:** local Ollama (`qwen2.5:7b-instruct` default).
- **Candidate scope:** only `status == "predicted_violation"` entries (~52 per page). Candidate_warnings and clean nodes are not sent.
- **Output:** `{decision, rule_id, confidence, reasoning}`
  - `decision` in `{confirm, reject, reassign}`
  - `rule_id` must come from the 46-rule vocabulary (validated client-side)
  - `confidence` in `[0, 1]`
  - `reasoning` short free-text (<=200 chars)
- **Failure policy:** if the Ollama call fails or returns invalid JSON after 2 retries, default to `decision=confirm` so a transient LLM error never hides a real violation.

## Fusion rules

| LLM decision | Action on prediction entry |
|---|---|
| `confirm` | keep `status=predicted_violation`; copy `rag_assigned_rule` from LLM if rule_id differs from top predicted rule |
| `reject` | demote `status=candidate_warning`; set `predicted_violation=false`; keep original probability; record `rag_*` fields |
| `reassign` | keep `status=predicted_violation`; replace `predicted_rules[0].rule_id` with `rag_assigned_rule` |

After all entries are processed, `summary.predicted_violation_count`, `summary.candidate_warning_count`, and `ground_truth` (precision, recall) are recomputed against the same axe ground truth used by the baseline report.

## CLI

```bash
python graph-rag/scripts/rag_validate.py \
  --prediction reports/prediction_airindia.json \
  --corpus-dir graphs_multi_a11y_v2 \
  --graph-source a11y-tree \
  --model qwen2.5:7b-instruct \
  --k-pos 8 --k-neg 4 \
  --max-candidates 60 \
  --output reports/prediction_airindia_rag.json
```

Flags:
- `--dry-run`: build exemplar lists but skip the Ollama call (for prompt inspection)
- `--only-rule link-name`: restrict validation to a single rule (iterative debugging)
- `--concurrency N`: parallel Ollama calls (default 1)
- `--cache-index`: force re-build of `_rag_index.pt`

## Evaluation

`scripts/compare_baseline_rag.py` takes baseline JSON + RAG JSON + axe ground truth and prints:

- Baseline precision / recall / F1 vs RAG precision / recall / F1
- Per-rule confusion deltas
- LLM decision distribution (confirmed / rejected / reassigned)
- Optional CSV row for sweeps over `--k-pos --k-neg --threshold`

## Files to add

All new files live under `graph-rag/`. No existing files (predict_site.py, train.py, models.py, etc.) are modified.

```
graph-rag/
  src/
    __init__.py
    corpus_loader.py        # CorpusLoader, Exemplar, query()
    snippet_extractor.py    # re-parse HTML for outerHTML per node_id (cached parse)
    rule_descriptions.py    # RULE_DESCRIPTIONS dict (axe help texts)
    ollama_client.py        # Ollama API client with JSON schema + retries
    prompts.py              # System + user prompt builders
    rag_fusion.py           # Update prediction JSON with RAG decisions
  scripts/
    rag_validate.py         # CLI entry
    compare_baseline_rag.py # Evaluation
  Architecture_GraphRAG.md  # this file
```

## Known risks & open questions

1. **ProcessedPage.load() drops node_map.** Checkpoints store only `data` + `html_path`. To get HTML snippets for exemplars we must re-parse 243 HTML files at corpus build time. The CorpusLoader caches a `corpus_index.pt` (embeddings + labels + outerHTML strings) once. Q: acceptable to cache the full corpus (~300MB + snippets) to `graphs_multi_a11y_v2/_rag_index.pt`?
2. **Text-only model.** `qwen2.5:7b-instruct` cannot see rendered pages, so color-contrast and image-alt cases are weaker. Q: skip RAG for color-contrast, or always attempt and let the LLM cite low confidence?
3. **Recall ceiling.** RAG here can only suppress FPs, not recover FNs. With 8 FNs from the GNN, best-case recall stays at 3/11 = 0.273. Q: OK for this experiment, or extend to validate top-N candidate_warnings too?
4. **Prompt budget.** 8 pos + 4 neg per candidate rule * up to 3 rules * 1KB HTML ~ 36KB per prompt, ~52 prompts ~ 1.9MB input total per page. Q: any preference on prompt-size trimming?
5. **Ollama latency.** ~52 candidates * ~3s ~ 3 minutes per page on a Mac. Q: want a `--concurrency` flag to parallelize calls?

These will be resolved before / during implementation.