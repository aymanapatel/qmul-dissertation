# Live accessibility suggestion demo

This demo accepts one public webpage URL, captures aligned HTML, visual and
accessibility-tree evidence, runs the frozen `learning_v2` GraphSAGE
specialists and routing policy, and calls the configured LLM for each routed
finding. The model returns strictly structured, bounded remediation
suggestions. Suggestions are never applied to the source page.

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
4. Frozen GraphSAGE checkpoints, calibration thresholds and fusion policy
   produce the suggestion candidates without using axe as a model input.
5. The configured OpenAI-compatible client requests a strict `RepairProposal`
   for each routed finding.
6. The UI polls `/v1/jobs/{job_id}` and displays the screenshot, GNN evidence
   and suggestions.

Private, loopback, link-local and credential-bearing destinations are rejected.
