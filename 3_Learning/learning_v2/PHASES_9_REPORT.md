# Phase 9 implementation report

## Live-run addendum (2026-08-03)

The earlier text below records the implementation state at the time it was written. Live OpenAI-compatible runs have since been executed. The real-page weak-label run produced 0 accepted, 7 rejected, and 3 human-review outcomes. A separate independent six-case matched benchmark was then run under no RAG, flat RAG, and graph-constrained RAG; its accepted and oracle precision/recall results are reported in `PHASE_10_REPORT.md`. Current configuration is loaded from the git-ignored local `.env` or explicit CLI arguments; credentials are never written to reports.

## Outcome

Phase 9 is implemented as a bounded generation-and-validation pipeline. It
uses the OpenAI Responses API `responses.parse` method with a strict Pydantic
`RepairProposal` schema. The default model is `gpt-5.6-sol`. The local API key
is a Python string in the git-ignored `accessibility_system/openai_key.py`; no
environment variable is read.

The model cannot return executable code or raw HTML. It can select only these
typed operations: set/remove an allow-listed attribute, replace text, insert a
label, set an allow-listed CSS property, or remove viewport zoom restrictions.
Identity, citation, and selector grounding are checked before application.

Each valid proposal is applied atomically to an isolated copy. Validation
records parsing and target resolution, the originating detector, axe,
deterministic specialists, accessibility-tree changes, keyboard focus replay,
visual changes, functional changes, before/after hashes, collection failures,
and explicit acceptance/rejection/review reasons. The corpus source is checked
afterward to prove that it was unchanged.

The runner also writes a redacted `phase_9.log` with per-attempt progress and
separate authentication, endpoint/model, connection, quota, schema/request,
grounding, and sandbox outcomes. Fatal API configuration failures stop further
billable attempts and mark the remaining inputs as skipped.

## Acceptance policy

- `accepted`: the independently verified target is resolved, no new in-scope
  regression exists, browser validation completes, and no semantic or visual
  judgement remains.
- `rejected`: the patch/selector/parse fails, the original detector does not
  reproduce, the target remains, a new accessibility issue appears, the source
  changes, or protected links/forms/scripts regress.
- `requires_human_review`: the finding is a weak label, semantic/contextual
  correctness is not independently established, the proposal requests review,
  a CSS visual change is made, or browser validation is incomplete.

This prevents semantic fixes from being accepted solely because a syntactic
rule passes.

## Commands

Install the locked OpenAI and schema dependencies:

```bash
.venv/bin/python -m pip install openai==2.52.0 pydantic==2.13.4
```

Run all tests:

```bash
.venv/bin/python -m pytest -q
```

Run the controlled fixture (no LLM call):

```bash
.venv/bin/python -u -m accessibility_system.phase9_fixture \
  --fixture learning_v2/fixtures/mixed_issues.html \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_3107_0015/phase_9_controlled
```

Run structured LLM generation after replacing the local key string:

```bash
.venv/bin/python -u -m accessibility_system.phase9 \
  --generator-inputs learning_v2/artifacts_3107_0015/phase_8/generator_inputs.json \
  --corpus-dir ../2_Data/browser-use/outputs/axe-core \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_3107_0015/phase_9 \
  --condition graph_constrained_rag \
  --max-proposals 10
```

## Executed evidence

The automated Phase 9 tests cover the strict schema, Responses API parsing,
explicit string key handling, atomic edits, unsafe-operation rejection,
grounding failures, a real Chromium+axe accepted repair, semantic review, and
regression rejection. The full project suite passes.

The controlled fixture execution is stored under
`artifacts_3107_0015/phase_9_controlled`. It is explicitly labelled as a
validation-only run without an LLM call. A live LLM run is not claimed because
the local key remains a placeholder.

## Dissertation limitations still to resolve

- Independently adjudicate real repair inputs; the Phase 8 axe-derived findings
  remain weak labels.
- Add page-specific modal, live-region, hover, authentication, and workflow
  replay scripts wherever those mechanisms occur.
- Define a canonical modified-HTML-to-graph regeneration path before claiming
  that Phase 5 learned specialists were rerun on repaired pages.
- Conduct blinded human assessment of contextual alternative text, accessible
  names, language, semantic structure, and visual correctness in Phase 10.
