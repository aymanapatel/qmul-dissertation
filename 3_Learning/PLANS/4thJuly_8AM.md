# Add GraphRAG To The A11y GNN

## Summary
Add GraphRAG as a post-prediction explanation/remediation layer. The GNN still detects likely violating nodes and rules; GraphRAG retrieves WCAG, ARIA, axe-rule, and repair-pattern knowledge to explain why the prediction matters and how to fix it.

V1 should be local and lightweight: no Neo4j, no external LLM dependency, no cloud vector DB. Use a small accessibility knowledge graph stored as JSON plus sentence-transformer embeddings for retrieval.

## Key Changes

- Add a local accessibility knowledge graph:
  - Nodes: `axe_rule`, `wcag_criterion`, `issue_family`, `aria_concept`, `html_pattern`, `repair_pattern`.
  - Edges: `maps_to`, `has_failure`, `fixed_by`, `applies_to`, `requires`, `related_to`.
  - Seed from `wcag_rules.py`, `misc/WCAG_All.md`, and curated fixes for common rules like `image-alt`, `button-name`, `link-name`, `label`, `color-contrast`, `html-has-lang`, `nested-interactive`, `list`, and `listitem`.

- Add a GraphRAG index builder:
  - New script: `3_Learning/scripts/build_graphrag_index.py`.
  - Input: local JSON knowledge graph.
  - Output: `3_Learning/knowledge/graphrag_index.pt`.
  - Use existing `sentence-transformers/all-MiniLM-L6-v2`.
  - Store text chunks, node IDs, edge metadata, and embeddings.

- Add retrieval utilities:
  - New module: `3_Learning/src/graphrag.py`.
  - Given a predicted node and predicted rule, retrieve:
    - exact axe-rule knowledge if rule ID is known,
    - related WCAG criteria,
    - relevant ARIA/HTML concepts,
    - top repair patterns by embedding similarity,
    - neighboring KG nodes for graph context.

- Extend prediction reports:
  - Add `--graphrag-index ./knowledge/graphrag_index.pt` to `predict_site.py`.
  - For each predicted violation, add:
    ```json
    "graphrag": {
      "issue_family": "...",
      "wcag": ["..."],
      "why_it_matters": "...",
      "suggested_fix": "...",
      "evidence_used": ["rule_id", "tag", "attrs", "text", "retrieved_kg_nodes"],
      "retrieved_items": [...]
    }
    ```
  - If GraphRAG is not supplied, keep current prediction output unchanged.

- Use graph context in retrieval:
  - Query text should include predicted rule, tag, role, attributes, accessible name/text preview, graph source, parent/child/sibling tags when available.
  - For `a11y-tree`, include inferred role and accessible name.
  - For `dom`, include tag and attributes.

## Implementation Details

- Keep GraphRAG separate from training at first.
  The model should not depend on the retrieval layer; GraphRAG enriches reports after inference.

- Use deterministic templates for final remediation text in V1.
  Do not require an LLM yet. The retrieved KG evidence should select the best explanation/fix template.

- Minimal first knowledge graph should cover these axe rules:
  - `image-alt`
  - `button-name`
  - `link-name`
  - `label`
  - `select-name`
  - `input-image-alt`
  - `html-has-lang`
  - `color-contrast`
  - `aria-allowed-attr`
  - `aria-valid-attr`
  - `aria-valid-attr-value`
  - `aria-required-children`
  - `aria-required-parent`
  - `nested-interactive`
  - `list`
  - `listitem`
  - `frame-title`

- Add fallback behavior:
  - If rule ID is unknown, retrieve by node context only.
  - If no retrieval item clears a similarity threshold, output a generic “manual review required” remediation.
  - If `--axe` is supplied, prefer axe rule IDs over predicted rule IDs for remediation.

## Test Plan

- Build index:
  ```bash
  cd 3_Learning
  python3 scripts/build_graphrag_index.py \
    --output ./knowledge/graphrag_index.pt
  ```

- Predict with GraphRAG:
  ```bash
  python3 scripts/predict_site.py \
    --html ../2_Data/browser-use/outputs/axe-core/www.aggrid.com/0.html \
    --model ./models_multi_f1_fix_semaphore/best_model.pt \
    --output ./reports/prediction_graphrag.json \
    --graph-source dom \
    --graphrag-index ./knowledge/graphrag_index.pt
  ```

- Verify report entries include:
  - WCAG criterion
  - issue family
  - explanation
  - suggested fix
  - retrieved KG node IDs
  - original GNN probability and predicted rules

- Unit/smoke scenarios:
  - `img` with missing `alt` retrieves `image-alt`.
  - button with empty accessible name retrieves `button-name`.
  - link with no text retrieves `link-name`.
  - input without label retrieves `label`.
  - unknown prediction still produces a fallback remediation.

## Assumptions

- GraphRAG is a report/remediation layer, not part of GNN training yet.
- V1 uses local JSON + embeddings, not Neo4j or a hosted vector DB.
- V1 does not call an LLM; generated text comes from retrieved repair templates.
- Later, an LLM can be added after retrieval to produce richer natural-language explanations.
