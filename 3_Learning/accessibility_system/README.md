# Accessibility system — Phases 8–10

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

## Phase 9 structured generation and sandbox validation

Phase 9 supports the OpenAI Responses API and OpenAI-compatible Chat Completions with Pydantic Structured Outputs.
The model must return `RepairProposal`; extra fields and raw patches are
rejected. The executor then applies only allow-listed typed operations to an
isolated copy and preserves the before/after HTML, screenshots, detector
results, accessibility-tree difference, interaction replay, visual difference,
functional difference, and final decision.

Bounded repair generation uses a reproducible low-variance configuration by
default: temperature `0.0`, top-p `1.0`, seed `42`, one completion, a 3,000-token
output cap, medium reasoning effort, low verbosity, and provider-side storage
disabled. Configure these with `OPENAI_TEMPERATURE`, `OPENAI_TOP_P`,
`OPENAI_SEED`, `OPENAI_MAX_OUTPUT_TOKENS`, `OPENAI_REASONING_EFFORT`, and
`OPENAI_VERBOSITY`, or override them with the matching Phase 9 CLI/API fields.
The seed is best-effort rather than a determinism guarantee and is sent only in
Chat Completions mode because the installed Responses API has no seed field.
Every run records the effective generation configuration and whether its seed
was applied.

Run the real graph-constrained condition from `3_Learning`:

```bash
.venv/bin/python -u -m accessibility_system.phase9 \
  --generator-inputs learning_v2/artifacts_3107_0015/phase_8/generator_inputs.json \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_3107_0015/phase_9 \
  --condition graph_constrained_rag \
  --model gpt-5.6-sol \
  --temperature 0.0 \
  --top-p 1.0 \
  --generation-seed 42 \
  --max-output-tokens 3000 \
  --reasoning-effort medium \
  --verbosity low \
  --max-proposals 10
```

Progress and redacted root-cause diagnostics are written to
`learning_v2/artifacts_3107_0015/phase_9/phase_9.log`. Authentication,
endpoint/model, quota, connection, request/schema, grounding, and validation
failures have separate categories. A fatal API configuration error is attempted
once; remaining proposals are marked `skipped_after_fatal_error`.

For an OpenAI-compatible provider that implements Structured Outputs through
Chat Completions, pass the API root (not the full `/chat/completions` path):

```bash
.venv/bin/python -u -m accessibility_system.phase9 \
  --generator-inputs learning_v2/artifacts_3107_0015/phase_8/generator_inputs.json \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_3107_0015/phase_9 \
  --condition graph_constrained_rag \
  --api-mode chat_completions \
  --base-url https://provider.example/v1 \
  --model provider/model-name \
  --max-proposals 10
```

The provider must support the `json_schema` response format. Merely exposing a
Chat Completions endpoint does not guarantee Structured Outputs support.

The existing Phase 8 real-page inputs are weak labels. Even when their syntax
is repaired, the validator routes them to human review until the originating
finding and semantic correctness are independently verified.

Run the no-LLM controlled validation fixture:

```bash
.venv/bin/python -u -m accessibility_system.phase9_fixture \
  --fixture learning_v2/fixtures/mixed_issues.html \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_3107_0015/phase_9_controlled
```

## HTTP API

The API exposes live axe-core scans, Phase 8 RAG inputs, asynchronous Phase 9
LLM/validation runs, job status, reports, and generated artifacts. From
`3_Learning`:

```bash
pip install fastapi uvicorn
python -u -m accessibility_system.api --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.
Slow browser and LLM operations return `202 Accepted` with a `job_id`; poll
`GET /v1/jobs/{job_id}` and then fetch `GET /v1/jobs/{job_id}/result`.

### Live-model bundle

The API owns its live-inference configuration. By default it explicitly uses
the training output expected at
`learning_v2/artifacts_evidence-v3.0_1208/phase_5_multiview_final_v2` and the
matching `phase_6_7_final/phase_6_fusion_policy.json`. The suggestion-audit
route will become usable when training has written the six model checkpoints,
their calibration/manifests, and the fusion-policy file. To use a different
completed bundle, set `ACCESSIBILITY_PHASE5_DIR` and
`ACCESSIBILITY_FUSION_POLICY` before starting the API. These settings are
passed explicitly to `run_live_specialists`; the API does not use
`learning_v2.live_inference`'s defaults.

Generate reviewable suggestions for one explicitly supplied webpage:

```bash
curl -X POST http://127.0.0.1:8000/v1/suggestion-audits \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.w3.org/","max_suggestions":5}'
```

The job captures same-session HTML, visual and live-AX evidence, builds the two
`learning_v2` graph views once, and runs the frozen MLP, GraphSAGE, and GAT
checkpoints for both views. Each architecture-specific finding remains separate
and can produce its own structured LLM call after calibrated routing.
Axe is retained as independent scan evidence and is not part of the GNN
inference fingerprint. The result contains a screenshot, checkpoint
probabilities, frozen thresholds, safe system/user prompt traces, token usage,
and ordered suggestions. Polling the job endpoint also returns live stage events
for capture, graph construction, all six specialist runs, routing, and LLM
generation. No proposed operation is applied to the source page. The focused
interface for this route is in `4_UI/learning-v2-demo`.

Scan public sites:

```bash
curl -X POST http://127.0.0.1:8000/v1/scans \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://example.com","https://www.w3.org"]}'
```

Inspect graph-constrained RAG inputs:

```bash
curl 'http://127.0.0.1:8000/v1/rag/inputs?condition=graph_constrained_rag&limit=10'
curl 'http://127.0.0.1:8000/v1/rag/inputs/q-595ebcc81dd2483a1d9f'
```

Start a structured repair and browser-validation run using the API provider
configuration already used by Phase 9:

```bash
curl -X POST http://127.0.0.1:8000/v1/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "condition":"graph_constrained_rag",
    "max_proposals":5,
    "model":"deepseek/deepseek-v4-flash",
    "api_mode":"chat_completions",
    "base_url":"https://openrouter.ai/api/v1",
    "skip_browser":false
  }'
```

For safety, the scan API accepts only public HTTP(S) destinations and rejects
loopback, private, link-local, reserved, and credential-bearing URLs. API keys
are never accepted in request bodies or returned by the service.

## Controlled matched repair study (Phase 10)

Build the six-case benchmark. Its oracle is stored separately and is never put in generator inputs:

```bash
.venv/bin/python -u -m accessibility_system.controlled_benchmark \
  --output-dir learning_v2/artifacts_3107_0015/repair_benchmark
```

Run Phase 9 once per condition with the same provider/model and retry budget, changing only `--condition` and `--output-dir`:

```bash
.venv/bin/python -u -m accessibility_system.phase9 \
  --generator-inputs learning_v2/artifacts_3107_0015/repair_benchmark/generator_inputs.json \
  --repair-truth learning_v2/artifacts_3107_0015/repair_benchmark/repair_truth.json \
  --corpus-dir learning_v2/artifacts_3107_0015/repair_benchmark/corpus \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_3107_0015/repair_benchmark/runs/graph_constrained_rag \
  --condition graph_constrained_rag \
  --api-mode chat_completions --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash --max-proposals 6 \
  --generation-retries 1 --log-level INFO
```

Run the non-LLM deterministic control through the identical validation sandbox.
Its template builder cannot receive the hidden oracle:

```bash
.venv/bin/python -u -m accessibility_system.deterministic_repair \
  --generator-inputs learning_v2/artifacts_3107_0015/repair_benchmark/generator_inputs.json \
  --repair-truth learning_v2/artifacts_3107_0015/repair_benchmark/repair_truth.json \
  --corpus-dir learning_v2/artifacts_3107_0015/repair_benchmark/corpus \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_3107_0015/repair_benchmark_v3/runs/deterministic_template \
  --max-proposals 6
```

Aggregate the matched reports:

```bash
.venv/bin/python -u -m accessibility_system.evaluation.repair_study \
  --run deterministic_template=learning_v2/artifacts_3107_0015/repair_benchmark_v3/runs/deterministic_template/phase_9_report.json \
  --run no_rag=learning_v2/artifacts_3107_0015/repair_benchmark/runs/no_rag/phase_9_report.json \
  --run flat_vector_rag=learning_v2/artifacts_3107_0015/repair_benchmark/runs/flat_vector_rag/phase_9_report.json \
  --run graph_constrained_rag=learning_v2/artifacts_3107_0015/repair_benchmark/runs/graph_constrained_rag/phase_9_report.json \
  --generator-inputs learning_v2/artifacts_3107_0015/repair_benchmark/generator_inputs.json \
  --repair-truth learning_v2/artifacts_3107_0015/repair_benchmark/repair_truth.json \
  --output-dir learning_v2/artifacts_3107_0015/repair_benchmark_v3/phase_10
```

The report contains a paired with/without-validation ablation for every
generative condition. The no-validation arm is a counterfactual over the exact
same proposal, so the gate comparison is not confounded by LLM sampling.
