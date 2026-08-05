# Phase 1–5 Execution Report

Executed: 2026-07-30  
Corpus: `2_Data/browser-use/outputs/axe-core`  
Canonical package: `3_Learning/learning_v2`

## Outcome

Phases 1–5 have been implemented and executed against the available saved
axe-core corpus. Phases 1, 2, and 4 meet their data-bound gates. Phase 3 has a
governed weak-label corpus and controlled independent fixtures, but still needs
human-adjudicated real-page labels before any claim of superiority over axe.
Phase 5 has a completed governed 90-site comparison run; it is an experimental
pilot, not the final dissertation-scale result.

## Phase 1 — Reproducibility and evaluation contract

Implemented:

- pinned direct dependencies in `requirements.lock`;
- versioned feature, checkpoint, evidence, finding, and run contracts;
- explicit separation of `x`, `edge_index`, and `tag_indices` from labels;
- rendered-node validity masks so invisible positive labels are excluded from
  both loss and evaluation instead of being trained and then zeroed at inference;
- complete evaluation predictions independent of display `top_k`;
- split hashes, per-rule metrics, validation-only calibration, and manifests;
- deterministic seeds and equal feature/budget comparisons across MLP,
  GraphSAGE, and GAT.

Environment materialised in `artifacts/phase_1_4/environment.json`:

- Python 3.14.3;
- PyTorch 2.12.1 and PyTorch Geometric 2.8.0;
- Playwright 1.61.0;
- scikit-learn 1.9.0;
- BeautifulSoup 4.15.0 and lxml 6.1.1.

Verification: 22 tests pass. The remaining warnings are dependency-level Python
3.14 deprecations.

## Phase 2 — Evidence collection and identity

Implemented:

- immutable ingestion of saved `0.html` and `page-0_home.json` pairs;
- stable static DOM identities and SHA-256 provenance;
- local Chromium rendering of saved HTML without revisiting live websites;
- computed layout/style evidence, screenshot capture, Chromium accessibility
  tree through CDP, and bounded keyboard focus trace;
- explicit collection-failure recording;
- materialised static evidence for 12 governed sites with zero failures.

Detailed browser sample: `www.1mg.com`:

- 2,852 static DOM nodes;
- 2,854 rendered DOM nodes;
- 2,850 matched stable paths;
- 99.93% static-path alignment and 99.86% browser-path alignment;
- 2,197 accessibility-tree nodes;
- 99 captured focus steps;
- zero collection failures.

The two unmatched paths remain visible in the alignment statistics rather than
being silently paired.

## Phase 3 — Governed data and ground truth

Full corpus inventory:

- 3,103 site directories inspected;
- 704 complete HTML/report pairs;
- 702 unique HTML hashes;
- 599 sites with at least one axe violation;
- 43 observed axe rule IDs;
- two duplicate HTML groups kept within a single split.

Governed split:

| Partition | Sites | Sites with violations |
|---|---:|---:|
| Train | 494 | 411 |
| Validation | 105 | 94 |
| Test | 105 | 94 |

Split hash:
`60b7e25c62a71990c6e889eddf4c0c6ca4c2cffa12bb70319ebd816dd628c68d`.

The controlled independent fixture contains positive, negative, and exception
examples for document title, language, viewport zoom, image alternative,
button/link name, form label, and nested interaction rules. The deterministic
baseline exactly matches its eight expected rule findings.

Gate still open: axe is the source of the real-site weak labels. A manually
adjudicated real-page benchmark is required before final claims and cannot be
fabricated from this corpus.

## Phase 4 — Deterministic and axe baselines

Implemented a common finding schema and deterministic checks for:

- document title and document language;
- image alternatives;
- button and link accessible names;
- form/select labels;
- restrictive viewport zoom;
- nested interactive elements.

Full 704-site comparison at the site-rule-pair unit:

| Metric | Value |
|---|---:|
| True positives | 597 |
| False positives | 1,104 |
| False negatives | 890 |
| Precision | 0.3510 |
| Recall | 0.4015 |
| F1 | 0.3745 |

This is a deliberately small deterministic implementation, not an axe clone.
Its results show why detector-specific scope and a shared unit of analysis must
be reported explicitly.

## Phase 5 — Specialist comparison

### Post-run cache audit correction (2026-08-03)

The historical cache was subsequently audited rather than trusted by filename. All 678 `rendered-visual` cache files have feature width 497, visual feature version 0, and no visual-match mask. The current versioned rendered collector emits width 517 and visual feature version 2. Therefore, every historical row labelled “Rendered visual” below is retained only as a legacy rendered-graph pilot; it is **not evidence that style/contrast visual cues improved prediction**. The canonical experiment now fails closed on this cache unless `--allow-legacy-rendered` is passed solely for reproduction. A new visual-cue claim requires regenerated version-2 graphs and the controlled `learning_v2.visual_ablation` study.

Executed configuration:

- 90 governed, duplicate-safe sites;
- maximum 2,500 nodes per aligned graph view;
- 5 epochs, identical 64-wide two-layer models;
- identical site partitions, features, optimizer budget, and calibration path
  for MLP, GraphSAGE, and GAT within each view;
- rules selected from train/validation support only;
- archived aligned DOM, static accessibility-tree, and rendered graph caches
  derived from the same axe-core corpus;
- test labels used only for the final metrics.

| View | Architecture | Supported rules | Node F1 | Rule F1 | Macro rule F1 | Test positives |
|---|---|---|---:|---:|---:|---:|
| DOM | MLP | `html-has-lang` | 0.000 | 0.000 | 0.000 | 3 |
| DOM | GraphSAGE | `html-has-lang` | 0.000 | 0.000 | 0.000 | 3 |
| DOM | GAT | `html-has-lang` | 0.000 | 0.000 | 0.000 | 3 |
| Accessibility tree | MLP | `image-alt`, `label`, `link-name` | 0.258 | 0.236 | 0.161 | 96 |
| Accessibility tree | GraphSAGE | same | 0.732 | 0.564 | 0.430 | 96 |
| Accessibility tree | GAT | same | 0.724 | 0.680 | 0.446 | 96 |
| Rendered visual | MLP | `color-contrast` | 0.138 | 0.138 | 0.138 | 98 |
| Rendered visual | GraphSAGE | same | 0.000 | 0.000 | 0.000 | 98 |
| Rendered visual | GAT | same | 0.078 | 0.078 | 0.078 | 98 |

Gate decisions:

1. **Accessibility-tree graph models continue.** Both graph models beat the
   feature-matched MLP on node/rule F1 in this run; GAT has the best rule F1.
2. **Rendered graph models do not pass the gate.** The MLP exceeds both graph
   models, and all visual results remain weak. Deterministic contrast remains
   the primary route pending better pixel/state evidence.
3. **DOM learned detection is unsupported.** Only two training positives and
   one validation positive were available for the selected rule; all models
   failed. Language/title checks remain deterministic.
4. These are pilot decisions only. They require repeated seeds, longer budgets,
   confidence intervals, and the independent benchmark before becoming thesis
   conclusions.

## Artifacts

- `artifacts/phase_1_4/phase_1_4_summary.json`
- `artifacts/phase_1_4/corpus_inventory.json`
- `artifacts/phase_1_4/governed_split.json`
- `artifacts/phase_1_4/deterministic_baseline.json`
- `artifacts/phase_1_4/fixture_evaluation.json`
- `artifacts/phase_1_4/browser_sample_www.1mg.com/evidence.json`
- `artifacts/phase_5_main/comparison.json`
- per-view/per-architecture checkpoints, calibration, histories, manifests,
  and test metrics beneath `artifacts/phase_5_main/`.

## Required work before headline evaluation

1. Add human-adjudicated real-page ground truth for selected visual,
   interaction, semantic, and repair criteria.
2. Repeat Phase 5 over several seeds and a larger predeclared support threshold.
3. Replace the static accessibility-tree approximation with regenerated graphs
   from the captured Chromium accessibility tree.
4. Add interaction fixtures for focus traps, occlusion, live regions, hover
   content, and workflow states.
5. Keep DOM metadata and contrast as deterministic controls rather than forcing
   them through a learned graph model.

---

# Addendum — full-corpus Phase 5 run

Executed: 2026-07-31  
Corpus: all cached aligned graphs under `runs_v1/graphs/graphs_multi_5July_2200`  
Artifacts: `artifacts/phase_5_full/`

## Outcome

The Phase 5 specialist comparison was repeated on the full aligned corpus rather
than the 90-site pilot subset. The protocol is otherwise identical to the pilot:
same 64-wide two-layer architectures, 5 epochs, validation-only calibration, and
test labels used only for final metrics. Rule coverage more than doubled because
more rules pass the train/validation support filter at full scale. The pilot's
three gate decisions are upheld.

## Executed configuration

- 667 sites with all three graph views present and at most 10,000 nodes per
  aligned view (one 45,846-node outlier excluded);
- 468 train / 101 validation / 98 test sites, drawn from the governed split with
  duplicate groups kept together;
- split hash
  `ad91d64a5c96bda42bce5221fe198763cc53d3c3b78d7ed2ceea18421e8b9f7b`;
- 5 epochs, identical features, budget, and calibration path for MLP,
  GraphSAGE, and GAT within each view;
- rules selected from train/validation support only, as in the pilot;
- single seed (42); run time 76 seconds on CPU.

## Rule coverage

| View | Rules trained (pilot) | Rules trained (full) |
|---|---:|---:|
| DOM | 1 | 15 |
| Accessibility tree | 3 | 13 |
| Rendered visual | 1 | 3 |

## Results

| View | Architecture | Supported rules | Node F1 | Rule F1 | Macro rule F1 | Test positives |
|---|---|---:|---:|---:|---:|---:|
| DOM | MLP | 15 | 0.016 | 0.030 | 0.029 | 181 |
| DOM | GraphSAGE | 15 | 0.068 | 0.013 | 0.034 | 181 |
| DOM | GAT | 15 | 0.027 | 0.055 | 0.053 | 181 |
| Accessibility tree | MLP | 13 | 0.780 | 0.489 | 0.283 | 619 |
| Accessibility tree | GraphSAGE | 13 | 0.773 | 0.684 | 0.331 | 619 |
| Accessibility tree | GAT | 13 | 0.665 | 0.731 | 0.301 | 619 |
| Rendered visual | MLP | 3 | 0.050 | 0.043 | 0.023 | 631 |
| Rendered visual | GraphSAGE | 3 | 0.131 | 0.129 | 0.044 | 631 |
| Rendered visual | GAT | 3 | 0.114 | 0.121 | 0.055 | 631 |

Per-rule results for the best accessibility-tree model (GAT):

| Rule | Precision | Recall | F1 | Test positives |
|---|---:|---:|---:|---:|
| `link-name` | 0.751 | 0.975 | 0.848 | 356 |
| `image-alt` | 0.601 | 0.899 | 0.720 | 139 |
| `svg-img-alt` | 0.556 | 0.952 | 0.702 | 21 |
| `button-name` | 0.553 | 0.901 | 0.685 | 81 |
| `aria-command-name` | 0.200 | 1.000 | 0.333 | 2 |
| `label` | 0.011 | 0.125 | 0.021 | 8 |

All other supported rules in all views have F1 at or below 0.13, and most are
zero; rare rules with single-digit test support remain unreliable.

## Gate decisions revisited

1. **Accessibility-tree graph models continue.** GraphSAGE and GAT both beat the
   feature-matched MLP on rule F1 (0.684 and 0.731 vs 0.489) and on macro rule
   F1. The relational benefit observed in the pilot persists at full scale.
2. **Rendered graph models do not pass the gate.** All visual results remain
   weak (rule F1 at or below 0.13). Deterministic contrast remains the primary
   route pending better pixel/state evidence.
3. **DOM learned detection remains unsupported.** All fifteen DOM rules,
   including the ARIA, language, and title checks, fail under every
   architecture. Exact checks remain deterministic.

## New observation: node-level and rule-level metrics diverge

The MLP has the best node F1 on the accessibility tree (0.780 vs 0.773 for
GraphSAGE and 0.665 for GAT) yet the worst rule F1 (0.489 vs 0.684 and 0.731).
The graph advantage is therefore at the rule-aggregation level, not at raw node
classification: graph models produce fewer isolated false positives, so their
per-rule and macro scores dominate even when the MLP flags slightly more
positive nodes. Dissertation claims should be stated at the rule level, with
node-level scores reported alongside for completeness.

## Limitations carried forward

This addendum does not close any of the required work listed above. Labels are
still axe-derived weak labels on the same corpus, so the circularity caveat is
unchanged; the run uses a single seed and a 5-epoch budget; no human-adjudicated
real-page benchmark exists yet; and the accessibility-tree graphs remain the
static approximation rather than regenerated Chromium accessibility trees.
These results refine the pilot's gate decisions but are still not the final
dissertation-scale evaluation.

## Artifacts

- `artifacts/phase_5_full/comparison.json`
- `artifacts/phase_5_full/pilot_split.json` (667-site selection and hash)
- per-view `rule_support.json` and per-view/per-architecture checkpoints,
  calibration, histories, manifests, and test metrics beneath
  `artifacts/phase_5_full/`.
