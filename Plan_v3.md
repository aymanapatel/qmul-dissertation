# AccessibilityGraph-RAG: Repository-Aligned Implementation and Evaluation Plan

Status: proposed implementation plan  
Inputs: `Plan_v2.md`, `3_Learning/`, and `misc/WCAG_All_ExceptTimebased.csv`  
Purpose: turn the architectural vision in `Plan_v2.md` into a bounded, testable dissertation project.

`Plan_v2.md` remains the high-level vision. This document is the delivery and evaluation plan.

## 1. Executive decision

Build the system as a hybrid of specialist analysers, but do not attempt to automate every WCAG criterion with one model.

The project will:

1. use `learning_v2` as the canonical learned-model pipeline;
2. convert the WCAG CSV into a validated routing registry;
3. collect shared DOM, browser accessibility-tree, rendered, and interaction evidence;
4. send exact rules to deterministic analysers, relational issues to graph analysers, visual issues to rendered analysers, dynamic issues to browser-state analysers, and contextual issues to semantic or human review;
5. normalise and fuse all detector outputs into one evidence-backed finding format;
6. compare no retrieval, flat vector RAG, and genuinely graph-aware retrieval for a bounded repair set;
7. apply proposed repairs only in a sandbox and accept them only after targeted and regression validation.

The GNN is an experimental specialist, not the default detector. It is retained only for criteria where relational structure produces a measurable benefit over a feature-matched MLP or deterministic baseline. A negative graph result is still a valid dissertation result.

## 2. Research contract

### 2.1 Aim

Develop and evaluate an evidence-aware web-accessibility system that routes candidate issues to deterministic, structural, visual, interaction, and semantic specialists, then generates and validates contextual repairs.

### 2.2 Main research question

> Does evidence-aware routing and validation improve detection and repair quality over conventional automated testing, single-modality learned models, and flat vector RAG on a site-held-out accessibility benchmark?

### 2.3 Supporting questions and primary comparisons

| Research question | Controlled comparison | Primary outcome |
|---|---|---|
| RQ1: Does graph information help relational detection? | MLP vs GraphSAGE vs GAT using identical node features and splits | Per-criterion macro F1 and PR-AUC on relational criteria |
| RQ2: Does rendered evidence help visual detection? | Structure-only vs structure plus rendered/style/state evidence | Per-criterion recall at a fixed precision and false positives per page |
| RQ3: Does specialist routing help overall detection? | axe/custom rules vs specialist union vs calibrated fusion | Site-macro F1, coverage, calibration, and manual-review rate |
| RQ4: Does graph-aware retrieval improve repairs? | No RAG vs flat vector RAG vs graph-constrained RAG with the same generator and context budget | Validated repair rate and blinded contextual-quality rating |
| RQ5: Does automatic re-validation improve safety? | Generated repairs with vs without the validation gate | Regression rate, rejection rate, and target-resolution rate |
| RQ6: Which evidence and edges matter? | Feature, edge, view, routing, retrieval, and validation ablations | Change in the corresponding primary metric with confidence intervals |

The unit of analysis must be declared for every experiment: node-criterion pair, page-criterion pair, interaction trace, site, retrieval query, or proposed repair. Results from different units must not be combined into one headline score.

## 3. Verified starting point

| Asset | Current state | Decision for this plan |
|---|---|---|
| `learning_v2` | Clean MLP/GraphSAGE/GAT pipeline with site splits, separate labels, checkpoint contracts, validation calibration, held-out evaluation, and axe-free prediction | Make this the canonical learned-model pipeline |
| Rule registry | 46 axe rule IDs assigned to accessibility-tree, DOM/DOM-page, or rendered-visual owners; they touch only 16 WCAG criteria | Preserve the one-to-many rule/criterion mapping, but do not claim full criterion coverage |
| Current v2 corpus | 704 cached `rendered-visual.pt` graphs, about 3.7 GB; current corpus has no matching DOM or accessibility-tree views | Regenerate aligned views from a versioned collector before multi-view experiments |
| Current v2 model bundle | Only a rendered-visual specialist is present | Train all selected specialists on one frozen split and schema |
| Current visual result | Best completed GraphSAGE test node F1 is about 0.049 and rule F1 about 0.052; MLP is close behind | Treat graph benefit as unresolved and add a go/no-go gate |
| Accessibility-tree graph | Existing builder is a static HTML-derived approximation, not the browser's live accessibility tree | Add a real browser accessibility-tree capture and keep the approximation as a separate view |
| Rendered evidence | Computed styles and geometry exist; there is no complete pixel, occlusion, hover, modal, or focus-state pipeline | Extend the collector before making visual-state claims |
| Graph-RAG | `graph-rag/` contains architecture/planning documents and an empty package scaffold | Implement only after the detection benchmark is stable |
| Repair validation | A small heuristic remediation smoke script exists; no isolated patch-and-regression loop exists | Reuse safe templates where possible and build a typed sandbox workflow |
| Verification | Full current suite passes: 17 tests, with dependency deprecation warnings | Keep green and add schema, CLI, live-evidence, retrieval, fusion, and repair tests |
| Reproducibility | No project-local dependency lock, CI workflow, or single canonical artifact layout | Fix before dissertation-scale experiments |

Legacy reports and model directories may be used for diagnosis, but they are not primary evidence unless reproduced through the canonical pipeline. The existing batch reports use different pipelines and incomplete raw axe data, so their scores must not be mixed with new held-out results.

## 4. WCAG scope and registry

### 4.1 CSV audit

The CSV contains 98 data records:

- 87 success-criterion rows;
- 9 of those rows are `1.2.x` time-based media criteria, despite the filename;
- 11 rows are not criteria: an embedded label-family header and 10 summary families;
- 78 criterion rows remain after excluding `1.2.x`;
- `4.1.1 Parsing` is legacy in WCAG 2.2, leaving 77 active non-media criteria;
- WCAG conformance level and version are not structured fields and must not be inferred silently.

The source CSV remains unchanged. Phase 0 will produce two validated artifacts:

- `wcag_criteria.json`: one structured record per criterion;
- `wcag_label_families.json`: the 10 auxiliary issue-family records.

The registry build must assert that the active non-media scope contains 77 criteria, that all `1.2.x` rows are excluded from this project, and that `4.1.1` is present only as a legacy compatibility record.

### 4.2 Non-media distribution

| WCAG area | Criteria retained | Primary evidence routes |
|---|---:|---|
| 1.1 Text Alternatives | 1 | rules, accessibility tree, surrounding context, optional vision, human review |
| 1.3 Adaptable | 6 | DOM/accessibility graph, deterministic attributes, reading-order and responsive evidence |
| 1.4 Distinguishable | 13 | computed styles, pixels, geometry, zoom/reflow, hover/focus states, specialist/manual audio checks |
| 2.1 Keyboard Accessible | 4 | keyboard replay, focus sequence/state graph, deterministic event checks |
| 2.2 Enough Time | 6 | timers, state replay, user controls, workflow/manual review |
| 2.3 Seizures and Physical Reactions | 3 | animation/pixel-frequency analysis and specialist/manual validation |
| 2.4 Navigable | 13 | DOM/accessibility graph, focus order, headings/landmarks, semantic context, occlusion geometry |
| 2.5 Input Modalities | 8 | pointer/touch automation, target geometry, accessible/visible labels, workflow state |
| 3.1 Readable | 6 | metadata rules, NLP, content context, human review |
| 3.2 Predictable | 6 | interaction/state diffs, cross-page consistency, semantic/manual review |
| 3.3 Input Assistance | 9 | forms graph, validation states, workflow replay, semantic/manual review |
| 4.1 Compatible | 3 | parser/ARIA rules, accessibility tree, interaction/live-region checks |
| **Total** | **78** | routes overlap by design |

### 4.3 Registry schema

Each criterion record should contain:

```text
criterion_id
name
wcag_version
level
status: active | legacy
scope: core | control | stretch | manual_only | excluded
issue_family_ids[]
candidate_generators[]
primary_detector
secondary_detectors[]
required_evidence[]
automation: automated | assisted | manual
ground_truth_source
repair_policy
validation_steps[]
human_review_required
axe_rule_ids[]
```

Keep `axe_rule_id`, `criterion_id`, detector ID, finding ID, and repair ID separate. An axe rule can map to several criteria, and several detectors can support one criterion.

### 4.4 Bounded dissertation scope

The registry can route all 77 active non-media criteria, but the primary experiments should focus on four contrasting families:

1. **Accessible names, forms, and relational semantics**: selected cases from 1.1.1, 1.3.1, 2.4.4, 2.4.6, 3.3.1, 3.3.2, and 4.1.2.
2. **Rendered contrast, focus, and target geometry**: selected cases from 1.4.1, 1.4.3, 1.4.11, 1.4.13, 2.4.7, 2.4.11, and 2.5.8.
3. **Keyboard, focus order, and dynamic announcements**: selected cases from 2.1.1, 2.1.2, 2.4.3, and 4.1.3.
4. **Contextual purpose and bounded remediation**: selected accessible-name, link/button-purpose, form-label, ARIA-relationship, and CSS colour/geometry repairs.

Easy deterministic controls such as 1.3.5, 2.4.2, 3.1.1, and 3.1.2 should be implemented to show that the router sends exact checks to rules rather than to a GNN.

Time-based media (`1.2.x`), long-running authentication/error-prevention workflows, full screen-reader equivalence, advanced audio analysis, and broad AAA conformance are stretch or manual-only work. The API and dashboard are also stretch deliverables; the reproducible CLI and structured report come first.

## 5. Target architecture

```text
Scan request or saved fixture
          |
          v
Playwright evidence collector
  DOM + live accessibility tree + styles + pixels + geometry + states
          |
          v
Candidate generators
  axe + custom rules + graph patterns + visual checks + interaction traces
          |
          v
WCAG registry router
  exact -> deterministic        relational -> graph
  rendered -> visual            dynamic -> interaction
  contextual -> semantic/human  mixed -> several specialists
          |
          v
Normalised findings and calibrated fusion
  provenance + confidence + conflicts + abstention/manual review
          |
          +------------------------------+
          |                              |
          v                              v
Detection report                 Knowledge/retrieval layer
                                  criterion -> technique/failure
                                  -> component/context -> repair
                                           |
                                           v
                                  Typed repair proposal
                                           |
                                           v
                                  Isolated validation sandbox
                                  targeted re-test + full regressions
                                           |
                                           v
                                  accepted | rejected | human review
```

The first router should be registry-driven and deterministic. A learned router is optional and only justified if there is labelled routing data.

### 5.1 Shared contracts

| Contract | Required content |
|---|---|
| `ScanArtifact` | scan ID, URL/fixture, viewport, browser/version, capture time, content hashes, collector/schema versions, failures |
| `NodeIdentity` | stable scan-local ID plus DOM path, accessibility-tree reference, frame/shadow root, state and geometry links |
| `EvidenceBundle` | DOM fragment, accessible name/role/state, styles, pixels/crop, bounds, visibility/occlusion, focus/hover/modal state, provenance |
| `Candidate` | candidate ID, possible criterion/rule IDs, generator, node/state locator, raw score and evidence references |
| `Finding` | criterion ID, detector, status, severity, calibrated confidence, evidence, explanation, conflicts, human-review flag |
| `RepairProposal` | typed target, operation, before/after value or patch, rationale, sources/exemplars, uncertainty, approval policy |
| `ValidationResult` | target resolved, regressions, visual/functional differences, relevant traces, outcome and rejection reason |

Axe-derived labels must never enter the inference fingerprint. They are allowed only in training/evaluation artifacts and must remain structurally separate from model inputs.

## 6. Implementation work packages

Durations are relative estimates, not calendar commitments. Several fixture and documentation tasks can run in parallel.

### Phase 0 — Freeze scope, hypotheses, and the WCAG registry (3–5 days)

Deliverables:

- normalized criterion and issue-family registries;
- explicit core/control/stretch/manual/excluded classifications;
- detector, evidence, truth-source, repair, and review routes;
- frozen RQs, units of analysis, baselines, and primary metrics.

Exit gate:

- registry tests confirm 77 active non-media criteria, 1 legacy criterion, 9 excluded `1.2.x` criteria, and 10 separated issue families;
- every active criterion has a declared route and automation status, including `manual_only` where appropriate;
- WCAG level/version metadata is verified against an authoritative source rather than inferred from prose.

### Phase 1 — Make `learning_v2` reproducible and correct the evaluation contract (1 week)

Deliverables:

- dependency/environment lock and documented setup;
- one artifact layout and manifest schema for caches, splits, models, calibration, predictions, and evaluations;
- updated README and removal of stale commands from the canonical path;
- fixes for known evaluation risks: invisible positive labels vs zeroed inference scores, `top_k` report truncation, node-alignment fingerprints, rare-rule split diagnostics, per-rule metrics, PR curves, and calibration metrics;
- CLI and small real-cache smoke tests.

Exit gate:

- all tests pass from a clean environment;
- the same seed and inputs reproduce split hashes and materially identical metrics;
- evaluator recall cannot be changed by report-display limits;
- no held-out test information is used for checkpoint choice or calibration.

### Phase 2 — Implement live evidence collection and shared identity (1–2 weeks)

Deliverables:

- Playwright capture of the post-load DOM and the browser's actual accessibility tree;
- computed styles, viewport, bounds, screenshots and element crops;
- explicit capture of focus, hover, modal, expanded/collapsed, validation, and selected dynamic states;
- stable mappings across DOM, accessibility, rendered, and state views;
- deterministic local fixtures for labels, tables, focus traps, overlays, live regions, contrast, target size, zoom/reflow, and allowed exceptions.

Exit gate:

- repeat captures of every fixture preserve the expected node/state identities;
- expected evidence is present and schema-valid;
- partial capture failures are recorded and cause abstention, not fabricated pass/fail results.

### Phase 3 — Build governed data and independent ground truth (2 weeks)

Deliverables:

- regenerated aligned DOM, browser-accessibility, rendered, and state artifacts;
- site- and template-grouped train/validation/test splits with near-duplicate checks;
- support and mapping-loss statistics for every selected rule and criterion;
- controlled positive, negative, and exception fixtures;
- an independently annotated real-page benchmark for semantic, visual, interaction, and repair-quality claims;
- annotation guide, dual annotation for subjective items, adjudication, and agreement reporting.

Exit gate:

- the untouched test set is frozen before model selection;
- each headline criterion has sufficient positive and negative support or is removed from headline evaluation;
- axe is treated as weak training evidence and a baseline, not as independent proof of superiority over axe.

### Phase 4 — Establish deterministic and conventional-tool baselines (1 week)

Deliverables:

- axe result adapter into the common `Finding` schema;
- deterministic HTML, ARIA, language, metadata, contrast, autocomplete, and geometry checks for the chosen scope;
- candidate-generation coverage measurements separate from candidate-verification measurements;
- positive, negative, boundary, and exception fixture tests.

Exit gate:

- exact criteria are never delegated to a learned model without a documented reason;
- deterministic boundary cases match the specification and fixtures;
- every output includes criterion, location, evidence, detector version, and provenance.

### Phase 5 — Train and evaluate specialist models (2–3 weeks)

Structural track:

- feature-matched MLP, GraphSAGE, and GAT;
- DOM approximation vs live accessibility tree vs combined graph;
- parent/child, sibling, label/reference, landmark, table, reading-order, focus-sequence, and spatial-edge ablations.

Visual track:

- deterministic contrast/geometry baseline first;
- structure-only vs rendered-style/geometry vs pixel/state features;
- multi-viewport and focus/occlusion cases.

Interaction track:

- scripted keyboard and state rules first;
- only add learned sequence/state models if the dataset supports them.

Exit gate:

- all architectures use identical features, splits, training budget, and calibration protocol;
- report per-criterion support and site-macro as well as micro metrics;
- retain a GNN in the final system only where it produces a defensible relational benefit; otherwise use rules/MLP and document the null result.

### Phase 6 — Implement routing, calibration, fusion, and abstention (1 week)

Deliverables:

- registry-driven routing with multiple specialists allowed per candidate;
- source-specific threshold calibration on validation data;
- duplicate finding merge, evidence union, conflict recording, and criterion-level confidence;
- explicit `pass`, `fail`, `needs_review`, `unsupported`, and `collection_failed` states;
- router/fusion tests for agreement, disagreement, missing evidence, and detector failure.

Exit gate:

- no unsupported or failed criterion is silently reported as passing;
- all fused findings preserve the contributing detector scores and evidence;
- confidence and manual-review thresholds are frozen before final testing.

### Phase 7 — Run the held-out detection study (1 week)

Required baselines:

1. axe alone;
2. axe plus custom deterministic rules;
3. MLP specialist;
4. GraphSAGE and GAT specialists;
5. each visual/interaction specialist independently;
6. uncalibrated union;
7. calibrated routed/fused system.

Required reporting:

- per-criterion and macro/micro precision, recall, F1, and PR-AUC;
- site-macro results and false positives per page;
- candidate-generation ceiling;
- calibration error/Brier score where probabilities are used;
- coverage, abstention/manual-review rate, collection failures, and latency;
- bootstrap confidence intervals over sites and paired comparisons;
- all feature, edge, view, routing, and calibration ablations linked to the RQs.

Exit gate:

- final test results are generated once from the frozen configuration;
- negative and inconclusive results are retained;
- detection is sufficiently reliable for the bounded repair study, or repair inputs are restricted to verified findings.

### Phase 8 — Implement and evaluate the knowledge/retrieval layer (2 weeks)

First define the terminology:

- if retrieval is embedding similarity over labelled examples, call it **exemplar/evidence RAG**;
- call it **Graph-RAG** only if a versioned graph explicitly links criterion, technique, failure, detector rule, element/context pattern, repair pattern, validation requirement, and provenance, and retrieval traverses or constrains by those edges.

Deliverables:

- versioned knowledge records and citations;
- training-only exemplar index with query-site and near-template exclusion;
- flat vector and graph-constrained retrievers with the same corpus and budget;
- retrieval evaluation before generation: Recall@k, nDCG/MRR where applicable, source correctness, diversity, and leakage tests;
- generator prompts that cite the exact evidence and retrieved records;
- safe failure behaviour that leaves the original finding unchanged.

Exit gate:

- the held-out test set is absent from every retrieval index;
- retrieved sources can be traced to versioned records;
- the graph-aware condition is meaningfully different from flat vector retrieval.

### Phase 9 — Generate bounded repairs and validate them in a sandbox (1–2 weeks)

Start with typed, locally testable repairs such as accessible-name/label associations, bounded ARIA relationships, language/metadata values, simple semantic structure, and CSS colour or target geometry. Contextual alt text and workflow/authentication changes must require human approval.

Validation sequence:

1. apply the typed patch to an isolated copy;
2. ensure the document still parses and the intended node can be resolved;
3. rerun the originating detector;
4. rerun axe and all in-scope specialists;
5. compare the accessibility tree;
6. replay relevant keyboard, focus, hover, modal, or live-region paths;
7. run visual and functional regression checks;
8. classify the result as `accepted`, `rejected`, or `requires_human_review`.

Exit gate:

- an automatic repair is accepted only if the target finding is resolved and no new in-scope regression is introduced;
- all patch attempts preserve before/after evidence and rejection reasons;
- semantic fixes are never accepted solely because a syntactic detector passes.

### Phase 10 — Run the repair study and package dissertation outputs (1–2 weeks)

Required repair conditions, with generator and context budget held constant:

1. deterministic template;
2. LLM without retrieval;
3. flat vector RAG;
4. graph-constrained RAG;
5. each generative condition with and without automatic re-validation.

Report:

- patch applicability and parse/build success;
- targeted issue-resolution rate;
- regression-free accepted repair rate;
- rejection and human-review rates;
- semantic/contextual correctness from blinded human assessment;
- latency and token/compute cost;
- failure taxonomy and representative accepted/rejected examples.

Final deliverables:

- reproducible CLI and structured JSON/HTML report;
- versioned configs, manifests, split hashes, metrics, plots, and ablation tables;
- architecture and threat-to-validity chapters tied directly to evidence;
- optional API/dashboard only if every core exit gate is complete.

## 7. Recommended module layout

Keep the learned pipeline focused and add orchestration as a separate importable package. Do not place production Python inside the hyphenated `graph-rag` directory.

```text
3_Learning/
  learning_v2/                  # canonical learned-model training/inference
  accessibility_system/
    catalog/                    # WCAG registry loader and validation
    contracts/                  # versioned scan/evidence/finding/repair schemas
    collection/                 # Playwright DOM/a11y/rendered/state capture
    candidates/                 # axe and specialist candidate generators
    detectors/
      deterministic/
      structural/
      visual/
      interaction/
      semantic/
    routing/                    # registry router, calibration, fusion, abstention
    retrieval/                  # vector and graph-constrained retrieval
    repair/                     # typed patches, generation policy, sandbox validation
    evaluation/                 # detection, retrieval, repair, statistics
  configs/
    wcag_criteria.json
    wcag_label_families.json
    experiments/
  scripts/
    build_wcag_registry.py
    collect_evidence.py
    analyse_site.py
    evaluate_detection.py
    build_retrieval_index.py
    evaluate_retrieval.py
    propose_repairs.py
    validate_repairs.py
    evaluate_repairs.py
  tests/
    fixtures/
    test_catalog_*.py
    test_contracts_*.py
    test_collection_*.py
    test_routing_*.py
    test_retrieval_*.py
    test_repair_*.py
    test_cli_e2e.py
```

Large caches, browser captures, indexes, checkpoints, and run outputs should live under versioned artifact directories excluded from source control. Each must have a small tracked or archived manifest containing hashes and provenance.

## 8. Experimental validity rules

1. **No circular superiority claim:** training on axe and testing only against axe measures axe imitation, not better accessibility detection.
2. **No site/template leakage:** group related domains, page templates, and mutations before splitting.
3. **No test-set tuning:** checkpoint choice, thresholds, routes, prompts, retrieval parameters, and repair policies are frozen using train/validation only.
4. **No coverage inflation:** an axe rule touching a criterion does not mean the system fully evaluates that criterion.
5. **No silent automation:** ambiguous semantic, workflow, audio, or exception-dependent cases must abstain or require review.
6. **No unvalidated repair success:** generation success is not repair success; only sandbox-validated, regression-free changes count as automatically accepted.
7. **No unfair ablation:** graph and non-graph models use the same features, splits, tuning budget, and evaluation unit.
8. **No metric hiding:** include class support, mapping/capture loss, failures, per-criterion scores, macro results, confidence intervals, and negative outcomes.

## 9. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Scope expands to all 77 criteria | Core experiments remain unfinished | Freeze four research families; route the rest as stretch/manual |
| Weak or circular axe labels | Invalid superiority claim | Independent manual/mutation benchmark and axe treated as baseline/weak labels |
| Rare positive rules | Unstable thresholds and misleading aggregate scores | Rule-support gates, grouped stratification diagnostics, macro/per-rule reporting |
| GNN does not beat MLP | Graph claim fails | Predeclare go/no-go gate and report a null result honestly |
| Offline/static capture misses dynamic evidence | Visual and interaction findings are invalid | Live browser collector, state fixtures, capture provenance, abstention on failure |
| Cross-view node misalignment | Wrong evidence or repair target | Versioned `NodeIdentity`, fingerprints, alignment assertions, fail closed |
| RAG leaks test examples | Inflated repair results | Train-only index, site/template exclusion, leakage tests |
| LLM hallucinates a fix or source | Unsafe repair | Typed operations, cited records, schema validation, safe fallback, sandbox gate |
| Dynamic sites are non-deterministic | Poor reproducibility | Saved fixtures, content hashes, controlled viewports/states, repeated-capture checks |
| Large artifacts and stale paths | Experiments cannot be reproduced | Canonical manifests, dependency lock, artifact hashes, updated documentation |

## 10. Definition of done

The core project is complete when:

- the validated registry represents 77 active non-media criteria and clearly marks actual automated, assisted, manual, legacy, and excluded coverage;
- the collector produces aligned, versioned evidence for the selected DOM, accessibility, visual, and interaction states;
- the canonical pipeline is reproducible and the complete test suite is green;
- a frozen, independently governed held-out benchmark exists;
- deterministic, MLP, GraphSAGE, GAT, visual, interaction, routed, and fused conditions are compared fairly;
- every research question has a predeclared metric, controlled baseline, ablation, and result with uncertainty;
- flat vector and graph-constrained retrieval are evaluated separately from generation;
- bounded repairs are measured by regression-free validation and human contextual review;
- the CLI emits a traceable report whose findings, evidence, routes, repairs, and validation decisions can be audited;
- limitations and null results are reported without claiming WCAG conformance beyond measured coverage.

## 11. Immediate next actions

1. Build and test the normalized WCAG registries.
2. Freeze `learning_v2` as canonical and add a dependency/artifact manifest.
3. Correct visibility, report truncation, split diagnostics, alignment, and per-rule evaluation issues.
4. Create the controlled evidence fixtures and live Playwright collector contract.
5. Regenerate aligned views and freeze the governed dataset before starting new model or RAG work.

These five actions unblock every later phase and should be completed before implementing the orchestrator or Graph-RAG layer.
