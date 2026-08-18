# Learning v2 — Phases 1–10

This package is the canonical, leakage-controlled pipeline for the first five
implementation phases in `Plan_v3.md`. It uses saved `0.html` and
`page-0_home.json` pairs beneath `2_Data/browser-use/outputs/axe-core`; it does
not revisit live websites.

## Phase 1–4 preparation

From `3_Learning`:

```bash
.venv/bin/python -m learning_v2.pipeline \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --output-dir learning_v2/artifacts/phase_1_4

### previous run
  .venv/bin/python -u -m learning_v2.pipeline \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --output-dir learning_v2/artifacts/phase_1_4 \
  --seed 42 \
  --evidence-sites 12
```

This writes the environment manifest, full corpus inventory, duplicate-aware
multilabel split, detailed evidence sample, controlled-fixture evaluation, and
full deterministic-vs-axe site/rule baseline.

## Phase 5 specialist pilot

The current corrected 50-site live-AX/rendered pilot, its exact commands, and
its dissertation gate decisions are in `CORRECTED_PILOT_COMMANDS.md` and
`artifacts_3107_0015/dissertation_corrected_pilot/CORRECTED_PILOT_RESULTS.md`.
The older commands below remain useful for full-cache collection and historical
reproduction, but their legacy-cache metrics are not the current visual or
live-AX evidence.

Audit a graph cache before using it for dissertation claims:

```bash
.venv/bin/python -u -m learning_v2.cache_audit \
  --cache-dir runs_v1/graphs/graphs_multi_5July_2200 \
  --output learning_v2/artifacts_3107_0015/cache_audit.json
```

The historical cache fails the rendered-cue contract (497 features, visual version 0). Regenerate the rendered view from the frozen split:

```bash
.venv/bin/python -u -m learning_v2.regenerate_visual_cache \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --split learning_v2/artifacts/phase_1_4/governed_split.json \
  --output-dir runs_v2/graphs/rendered_visual_v2 --resume
```

Regenerate the browser accessibility graph from Chromium's live AX tree:

```bash
.venv/bin/python -u -m learning_v2.regenerate_live_ax_cache \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --split learning_v2/artifacts/phase_1_4/governed_split.json \
  --output-dir runs_v2/graphs/live_ax_v1 --resume
```

After all frozen sites pass the cache audit and the aligned DOM/a11y views are present in the same cache root, run the controlled visual ablation:

```bash
.venv/bin/python -u -m learning_v2.visual_ablation \
  --cache-dir runs_v2/graphs/rendered_visual_v2 \
  --split learning_v2/artifacts/phase_1_4/governed_split.json \
  --output-dir learning_v2/artifacts_3107_0015/visual_ablation \
  --seeds 41 42 43 44 45 --epochs 20 --device cpu
```

The archived aligned graph cache is derived from the same axe-core corpus. The
pilot intentionally uses a bounded, small-graph subset; it verifies the full
experimental path and is not a dissertation headline result.

```bash
.venv/bin/python -m learning_v2.experiment \
  --cache-dir runs_v1/graphs/graphs_multi_5July_2200 \
  --inventory learning_v2/artifacts/phase_1_4/corpus_inventory.json \
  --split learning_v2/artifacts/phase_1_4/governed_split.json \
  --output-dir learning_v2/artifacts/phase_5_main \
  --max-sites 90 --max-nodes 2500 --epochs 5 --device cpu \
  --allow-legacy-rendered --allow-static-a11y

# previous run
  .venv/bin/python -u -m learning_v2.experiment \
  --cache-dir runs_v1/graphs/graphs_multi_5July_2200 \
  --inventory learning_v2/artifacts/phase_1_4/corpus_inventory.json \
  --split learning_v2/artifacts/phase_1_4/governed_split.json \
  --output-dir learning_v2/artifacts/phase_5_main \
  --max-nodes 2500 \
  --epochs 20 \
  --batch-size 2 \
  --hidden 64 \
  --layers 2 \
  --device cpu \
  --allow-legacy-rendered --allow-static-a11y
```

Every architecture within a view receives the same sites, features, supported
rules, epoch budget, calibration procedure, and test evaluation. Rendered
labels on invisible nodes are masked from loss and metrics. Axe labels are
structurally separate from the inference fingerprint.

The executed results and gate decisions are recorded in
`PHASES_1_5_REPORT.md`.

## Phase 6 routing and Phase 7 held-out study

Phase 6 adds registry-driven multi-specialist routing, validation-frozen source
thresholds, duplicate fusion, evidence union, conflict recording, and explicit
`pass`, `fail`, `needs_review`, `unsupported`, and `collection_failed` states.

The current real-page truth remains axe-derived, so the executable Phase 7 run
is labelled `weak_label_pilot`. It will not accept `--final` without an
independent manual truth file.

```bash
.venv/bin/python -u -m learning_v2.study \
  --phase5-dir learning_v2/artifacts_3107_0015/phase_5_main \
  --cache-dir runs_v1/graphs/graphs_multi_5July_2200 \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --registry ../configs/wcag_criteria.json \
  --output-dir learning_v2/artifacts_3107_0015/phase_6_7 \
  --bootstrap-samples 1000 --seed 42 --device cpu \
  --allow-legacy-rendered --allow-static-a11y
```

`--allow-legacy-rendered` and `--allow-static-a11y` reproduce the historical weak-label pilot only. They are rejected for `--final`; the final study requires regenerated version-2 visual and Chromium live-AX checkpoints.

The output includes the frozen fusion policy, routing decisions, fully
provenanced fused findings, study metrics, bootstrap confidence intervals,
paired architecture comparisons, ablations, and a hashed run manifest. See
`PHASES_6_7_REPORT.md` for the executed results and limitations.

## Phase 8 knowledge and retrieval

Phase 8 is implemented in the separate `accessibility_system` package so that
retrieval orchestration does not enter the learned-model pipeline. It provides
a versioned W3C-backed knowledge graph, a training-only exemplar index,
no-RAG/flat-vector/graph-constrained conditions with equal budgets, retrieval
metrics, leakage checks, cited generator inputs, and safe retrieval failure.

The executed command and results are documented in `PHASE_8_REPORT.md` and
`accessibility_system/README.md`.

## Phase 9 bounded repair and validation

Phase 9 adds strict OpenAI Structured Output generation, allow-listed typed
patching, immutable source handling, and isolated validation with the
originating detector, axe, deterministic specialists, Chromium accessibility
tree, keyboard focus replay, screenshot comparison, and functional DOM checks.
Automatic acceptance requires a resolved verified finding and zero new
in-scope regressions; semantic and weak-label cases are routed to human review.

See `PHASE_9_REPORT.md` and `accessibility_system/README.md` for the commands,
executed controlled result, and current limitations.

## Phase 10 matched repair evaluation

The independently specified six-case deterministic-template/no-RAG/flat-RAG/GraphRAG controlled run and its precision/recall, target-resolution, regression, citation, token, cost, paired, and validation-gate-ablation results are documented in `PHASE_10_REPORT.md`. Every condition achieved 6/6 exact-oracle, regression-free accepted repairs. This is an important null result: the bounded cases do not require an LLM. GraphRAG used substantially more of its cited evidence than flat RAG but did not improve success over the tied baselines. The final contextual-quality gate still requires completion of the generated 24-candidate condition-blinded rating packet.

`python -m learning_v2.readiness_audit` creates a fail-closed phase-by-phase
JSON/Markdown dossier. The current dossier records all automated contracts as
implemented, while independently identifying the missing dual-human labels and
the currently incomplete `1/3` stochastic LLM replicate design.

## Phase 5 full-corpus run

The same protocol was repeated over every site with all three aligned views and
at most 10,000 nodes per view (667 sites: 468 train, 101 validation, 98 test):

```bash
.venv/bin/python -m learning_v2.experiment \
  --cache-dir runs_v1/graphs/graphs_multi_5July_2200 \
  --inventory learning_v2/artifacts/phase_1_4/corpus_inventory.json \
  --split learning_v2/artifacts/phase_1_4/governed_split.json \
  --output-dir learning_v2/artifacts/phase_5_full \
  --max-sites 704 --max-nodes 10000 --epochs 5 --device cpu \
  --allow-legacy-rendered --allow-static-a11y
```

Results and the revisited gate decisions are in the addendum to
`PHASES_1_5_REPORT.md`.

## Scope and limitations

- The 704 complete snapshot/report pairs are weakly labelled by axe.
- The independent benchmark currently consists of controlled fixtures; real
  page human adjudication remains required before superiority claims.
- Static evidence is available for all complete inputs. Detailed evidence is
  materialised only for a reproducible sample by default to avoid duplicating
  hundreds of megabytes.
- Browser evidence capture operates on saved local HTML and records failures.
- Phase 5 pilot results validate execution; full runs require larger support,
  longer budgets, and the frozen dissertation experiment configuration.
