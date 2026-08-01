# Learning v2 — Phases 1–8

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

The archived aligned graph cache is derived from the same axe-core corpus. The
pilot intentionally uses a bounded, small-graph subset; it verifies the full
experimental path and is not a dissertation headline result.

```bash
.venv/bin/python -m learning_v2.experiment \
  --cache-dir runs_v1/graphs/graphs_multi_5July_2200 \
  --inventory learning_v2/artifacts/phase_1_4/corpus_inventory.json \
  --split learning_v2/artifacts/phase_1_4/governed_split.json \
  --output-dir learning_v2/artifacts/phase_5_main \
  --max-sites 90 --max-nodes 2500 --epochs 5 --device cpu

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
  --device cpu
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
  --bootstrap-samples 1000 --seed 42 --device cpu
```

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

## Phase 5 full-corpus run

The same protocol was repeated over every site with all three aligned views and
at most 10,000 nodes per view (667 sites: 468 train, 101 validation, 98 test):

```bash
.venv/bin/python -m learning_v2.experiment \
  --cache-dir runs_v1/graphs/graphs_multi_5July_2200 \
  --inventory learning_v2/artifacts/phase_1_4/corpus_inventory.json \
  --split learning_v2/artifacts/phase_1_4/governed_split.json \
  --output-dir learning_v2/artifacts/phase_5_full \
  --max-sites 704 --max-nodes 10000 --epochs 5 --device cpu
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
