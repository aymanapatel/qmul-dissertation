# Phase 10 repair-study report

Executed: 2026-08-03  
Model: `deepseek/deepseek-v4-flash` through an OpenAI-compatible Chat Completions structured-output endpoint  
Unit: repair query

## Corrected matched four-condition study (current)

The frozen generation policy was corrected to distinguish explicitly independently verified public values from unknown semantic intent, provide rule-to-operation schema guidance, and feed local schema-validation errors into bounded retries. The sandbox policy was also narrowed so an exact hidden-oracle visible-label insertion with independent semantic verification is not rejected merely because the expected label changes pixels. Repairs without independent truth retain the human-review requirement.

The same six queries, model, endpoint mode, context budget, hidden oracle, browser validation, and retry budget were then rerun for all conditions:

| Condition | Generation | Target resolved | Validated/accepted | Oracle precision | Oracle recall | Citation use | Tokens | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Deterministic template | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | 0 | 0.000000 |
| No RAG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | 10,748 | 0.001710 |
| Flat vector RAG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.267 | 14,348 | 0.002148 |
| Graph-constrained RAG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 12,084 | 0.002091 |

All 24 attempts were applicable, exact-oracle, target-resolving, and regression-free. The deterministic template uses only saved finding evidence; hidden oracle operations are merged by the validator after proposal creation. Success therefore ties across all four conditions. The controlled cases are simple enough that an LLM provides no repair-success benefit, and no GraphRAG superiority is claimed. GraphRAG's measured distinction is evidence selectivity: citation utilisation was 0.917 versus 0.267 for flat RAG, with 15.8% fewer tokens and slightly lower cost than flat RAG.

The Phase 10 evaluator now includes a paired validation-gate ablation on each identical generated proposal. The ungated counterfactual provisionally accepts every schema- and policy-valid proposal; the gated result is the observed sandbox outcome. Both were 1.000 in this bounded set, so the validation-effect result is also null. The gate path is operational, but its safety benefit must not be inferred from cases that contained no unsafe generated repair.

The current structured report is `artifacts_3107_0015/repair_benchmark_v3/phase_10/phase_10_repair_study.json`. A 24-candidate condition-blinded human-rating packet covering the deterministic and three LLM conditions is ready under `artifacts_3107_0015/repair_benchmark_v3/human_rating_packet/`.

Stochastic stability is evaluated separately by
`accessibility_system.evaluation.repair_replicates`, which requires three
balanced replicates and hierarchically bootstraps replicates and sites. The
current audit contains only the completed `r1` run (`1/3`), so stochastic
stability is explicitly not yet established. Commands for `r2`, `r3`, and the
balanced aggregation are in `CORRECTED_PILOT_COMMANDS.md`.

The section below is retained as the historical pre-correction run and failure analysis.

## Controlled matched study

Six independently specified fixtures were evaluated under the same no-RAG, flat-vector-RAG, and graph-constrained-RAG context budget. The generator inputs exclude oracle operations; `repair_truth.json` is loaded only by sandbox validation. Each condition used the same model and six query IDs.

| Condition | Generation success | Target resolved among sandboxed | Validated/accepted | Oracle precision | Oracle recall | Review |
|---|---:|---:|---:|---:|---:|---:|
| No RAG | 0.833 | 0.400 | 0.333 | 1.000 | 0.333 | 0.500 |
| Flat vector RAG | 0.667 | 0.500 | 0.167 | 1.000 | 0.167 | 0.500 |
| Graph-constrained RAG | 0.667 | 0.500 | 0.333 | 1.000 | 0.333 | 0.333 |

All accepted repairs exactly matched the hidden oracle, resolved the originating finding, and introduced no detected regression, giving precision 1.0 in this controlled sample. Graph-constrained RAG accepted two of six cases, matching no RAG and exceeding flat RAG by one case. With only six cases, paired McNemar tests are not significant and bootstrap intervals are wide; no GraphRAG repair-superiority claim is made.

Graph-constrained context was more selective: source correctness was 1.0 versus 0.333 for flat retrieval, and mean retrieved-citation utilisation by proposed repairs was 0.75 versus 0.40. It also used 5,493 tokens and recorded cost 0.000770, compared with 8,310 tokens and 0.001702 for flat RAG in this run.

## Failure analysis

The OpenAI-compatible provider returned schema-invalid cross-field output on one no-RAG case and two cases in each RAG condition. Other unsuccessful cases correctly abstained or requested review. The runner now supports bounded retries for schema-invalid output and never retries fatal authentication/configuration errors; retry count must be frozen for all compared conditions.

The six-case controlled benchmark proves typed generation, oracle separation, deterministic and LLM baselines, browser validation, and the executable regression gate. It does not replace blinded assessment on independently adjudicated real pages. Phase 10 is therefore a completed controlled pilot, while the dissertation final repair gate remains open pending predeclared stochastic replicates and blinded human ratings.
