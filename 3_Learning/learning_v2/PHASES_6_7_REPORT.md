# Phase 6–7 Execution Report

## Executed: 2026-08-01  

Plan: `Plan_v3.md`  
Held-out split hash: `b2c549d3c46587c028cb71788d410cf98c276f6bc02561d46f353f3c2df1adb6`

## Outcome

**Post-run cache correction (2026-08-03):** the nominal rendered-visual checkpoint used by this historical pilot has feature version 0 and width 497, so the `visual_specialist` row is not evidence of style/contrast visual cues. It remains in the table for reproducibility. The study runner now rejects this checkpoint by default and always rejects it for `--final`.

Phase 6 is implemented with registry-driven multi-specialist routing,
source/rule thresholds frozen from Phase 5 validation data, evidence-preserving
fusion, conflict recording, and abstention. Phase 7 is implemented and executed
on the 13 held-out Phase 5 pilot sites for the five supported criteria.
 8
The run is deliberately recorded as a **weak-label pilot**, not the final
dissertation detection study. Axe supplies the available real-page labels, so
axe-alone performance and the candidate-generation ceiling are circular. The
runner refuses `--final` unless an independent manual truth file is provided,
and refuses to overwrite an existing final result.

## Phase 6 — Routing, calibration, fusion, and abstention

Implemented:

- registry-driven routing with multiple detectors per criterion;
- validation-frozen, source/rule-specific thresholds;
- duplicate merging at site, criterion, and rule-target level;
- union of evidence while retaining every contributing observation and score;
- explicit disagreement and partial-collection-failure conflicts;
- `pass`, `fail`, `needs_review`, `unsupported`, and `collection_failed` states;
- provenance validation that prevents failed or unsupported checks from being
  silently emitted as passes;
- a frozen fusion failure threshold of 0.65 and review threshold of 0.35.

Across the held-out pilot, 78 rule-target fused records were emitted: 4 fails,
38 `needs_review`, and 36 passes. At the site-criterion unit the manual-review
rate was 49.23%. This conservative result is expected: a disagreement becomes
review rather than an automatic pass or failure.

## Phase 7 — Held-out weak-label study

Unit of analysis: site-criterion pair.  
Sites: 13.  
Criteria: 1.1.1, 1.4.3, 2.4.4, 3.1.1, and 4.1.2.  
Bootstrap: 1,000 site-level resamples with seed 42.

| Method | Precision | Recall | F1 | PR-AUC | Coverage | Review rate |
|---|---:|---:|---:|---:|---:|---:|
| axe alone | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| custom deterministic | 0.591 | 0.722 | 0.650 | 0.504 | 1.000 | 0.000 |
| axe + custom | 0.667 | 1.000 | 0.800 | 0.667 | 1.000 | 0.000 |
| MLP specialists | 0.471 | 0.444 | 0.457 | 0.496 | 1.000 | 0.000 |
| GraphSAGE specialists | 0.560 | 0.778 | 0.651 | 0.518 | 1.000 | 0.000 |
| GAT specialists | 0.452 | 0.778 | 0.571 | 0.596 | 1.000 | 0.000 |
| visual specialist | 0.400 | 1.000 | 0.571 | 0.817 | 0.200 | 0.000 |
| interaction specialist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| uncalibrated union | 0.667 | 1.000 | 0.800 | 0.667 | 1.000 | 0.000 |
| calibrated routed fusion | 1.000 | 0.222 | 0.364 | 0.667 | 1.000 | 0.492 |

The interaction specialist is explicitly unsupported because there is no
Phase 5 interaction/state model or independently labelled trace corpus. It is
not interpreted as a passing detector.

Site-bootstrap 95% F1 intervals were wide: MLP 0.167–0.684, GraphSAGE
0.375–0.821, and GAT 0.316–0.760. Paired F1-difference intervals include zero
for GraphSAGE minus MLP (-0.047 to 0.450) and GAT minus MLP (-0.123 to 0.348),
so this pilot does not establish a statistically defensible graph advantage.
Fusion traded recall for safety: it produced no false positives among automatic
failures, but referred almost half of the study universe for review.

## Artifacts

- `artifacts_3107_0015/phase_6_7/phase_6_fusion_policy.json`
- `artifacts_3107_0015/phase_6_7/phase_6_routing_decisions.json`
- `artifacts_3107_0015/phase_6_7/fused_findings.json`
- `artifacts_3107_0015/phase_6_7/phase_7_detection_study.json`
- `artifacts_3107_0015/phase_6_7/run_manifest.json`

## Exit-gate status

Phase 6's implementation gate passes: unsupported and failed checks cannot
silently pass, all fused findings retain their contributors and evidence, and
the thresholds are frozen in a separate artifact.

Phase 7's final-study gate remains open. To close it, supply an independently
adjudicated truth file, freeze the larger study configuration, and run once
with `--truth-source independent_manual --truth-file ... --final`. Until then,
the generated results are diagnostic pilot evidence only.
