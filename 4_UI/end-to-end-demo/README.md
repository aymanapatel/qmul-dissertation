# Live accessibility suggestion demo

This demo accepts one public webpage URL, captures aligned HTML, visual and
accessibility-tree evidence, then runs frozen `learning_v2` MLP, GraphSAGE and
GAT specialists over both graph views. Architecture-specific findings remain
separate through calibrated routing and structured LLM generation. The UI shows
live stages, all six model runs, exact safe prompts, formatted API payloads and
strictly structured remediation suggestions. Suggestions are never applied to
the source page.

## Start the FastAPI backend

The backend reads the existing git-ignored
`3_Learning/accessibility_system/.env`. Keep the API key there; never put it in
the browser or a request body.

```bash
cd 3_Learning
.venv/bin/python -m accessibility_system.api --host 127.0.0.1 --port 8000
```

FastAPI documentation is available at `http://127.0.0.1:8000/docs`.
The first run loads `all-MiniLM-L6-v2`, the same text-feature model used to
construct the training graphs. It has now been cached locally in this
workspace environment.

## Start the webpage

In a second terminal:

```bash
cd 4_UI/learning-v2-demo
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, enter a public HTTP(S) URL, choose how many
suggestions to generate, and select **Generate suggestions**.

## API flow

1. `POST /v1/suggestion-audits` validates the URL and creates a background job.
2. Chromium captures the specified page, computed visual features and live AX
   tree in the same session; axe is retained as separate audit evidence.
3. The accessibility-tree and rendered-visual graphs are built with the exact
   training feature contracts.
4. Frozen MLP, GraphSAGE and GAT checkpoints run on both views using their own
   validation-frozen thresholds; axe is not a model input.
5. Architecture/view findings are routed and ordered independently, so matching
   findings are not hidden by cross-model deduplication.
6. The configured OpenAI-compatible client requests a strict `RepairProposal`
   for each selected finding.
7. The UI polls `/v1/jobs/{job_id}` and displays live stage events, the six-run
   comparison, screenshot, system prompt, user JSON, safe API metadata and
   structured response.

The public job payload contains no run-directory path, API key, authorization
header, cookie, or hidden model reasoning.

Private, loopback, link-local and credential-bearing destinations are rejected.
