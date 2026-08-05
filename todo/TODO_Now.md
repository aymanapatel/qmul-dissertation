# TODO Now — Final evidence corpus and Phase 3 preparation

This is the execution order for the new HTML, visual sidecars, and axe reports in
`2_Data/browser-use/outputs/dataset_v3.0/axe-core`, based on `Plan_v3.md`.



## [x] Phase 2 core — same-session evidence capture

Implemented and validated in `2_Data/browser-use`:

- [x] Capture Chromium `Accessibility.getFullAXTree` while temporary
      `data-gnn-node-id` markers are still attached.
- [x] Store `backendDOMNodeId → data-gnn-node-id` in `0.ax.json`.
- [x] Capture the audited page's own CDP target as `0.png`. Do not use the
      session-focus screenshot API: it captured browser-use's placeholder tab.
- [x] Add `--output-dir`; fail evidence capture when any required artifact is
      absent; preserve pre-existing marker attributes after capture.
- [x] Add focused capture tests and a real-browser smoke test.

The intended five-file bundle for every page is:

```text
0.html
0.visual.json
0.ax.json
0.png
page-0_home.json
```

The final crawler rerun completed. Do not use `--skip-existing-output` when
recapturing a site with invalid evidence:
```bash
cd /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use
source .venv/bin/activate

python3 -u main.py \
  --no-auth \
  --start-row 1 \
  --end-row 1000 \
  --workers 4 \
  --settle-seconds 2 \
  --capture-ready-timeout 60 \
  --capture-stable-seconds 3 \
  --output-dir outputs/dataset_v3.0/axe-core
```

Verified 2026-08-04: the governed split contains **858 / 858 complete bundles**
(601 train, 128 validation, 129 test). The corpus root also has 442 older or
incomplete directories; they are not in the governed split.

Verify the governed cohort, not every historical directory:

```bash
python3 - <<'PY'
from pathlib import Path

import json
root = Path('outputs/dataset_v3.0/axe-core')
split = json.loads(Path('../../3_Learning/learning_v2/artifacts_dataset_v3.0/phase_1_4/governed_split.json').read_text())
required = {'0.html', '0.visual.json', '0.ax.json', '0.png', 'page-0_home.json'}
sites = [root / site for partition in ('train', 'val', 'test') for site in split[partition]]
for path in sites:
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        print(path.name, 'missing:', ', '.join(missing))
print('Governed sites:', len(sites))
PY
```

Once this succeeds, all Phase 1 onward commands must use:

```text
../2_Data/browser-use/outputs/dataset_v3.0/axe-core
```

instead of the old `outputs/axe-core` directory.

## Phase 0 — WCAG registry

- [ ] Revalidate the registry:

```bash
.venv/bin/python -u -m learning_v2.catalog.build_registry \
  ../misc/WCAG_All_ExceptTimebased.csv ../configs

.venv/bin/python -m pytest -q learning_v2/tests/test_catalog.py
```

Expected: 77 active non-media criteria, 1 legacy criterion, 9 excluded
time-based criteria, and 10 issue families.

## [x] Phase 1 outputs and Phase 3 data split prerequisites

- [x] Build a new inventory and duplicate-aware 70/15/15 split:

```bash
.venv/bin/python -u -m learning_v2.pipeline \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_1_4 \
  --seed 42 --evidence-sites 30
```

- [x] Inspect the frozen split and summary. Current result: 858 complete sites,
      846 unique HTML snapshots, split hash
      `7b890c6737ea57f5b158ef6ef1a1ce928485858a82a03990096c3f56321832f5`,
      and fixture exact match `true`.

```bash
jq '{complete_sites,unique_html,split_hash,fixture_exact_match}' \
  learning_v2/artifacts_dataset_v3.0/phase_1_4/phase_1_4_summary.json

jq '{train:(.train|length),val:(.val|length),test:(.test|length),split_hash}' \
  learning_v2/artifacts_dataset_v3.0/phase_1_4/governed_split.json
```

- [ ] Add a fail-closed aligned-bundle admission check to `learning_v2.pipeline`.
      The current implementation admits HTML+axe pairs without requiring
      `0.visual.json`; do not call this the final governed split until fixed.

- [ ] Treat `deterministic_baseline.json` from this run as the new Phase 4
      baseline.

## [x] Phase 2 static evidence graphs

- [x] Regenerate rendered graphs. `FeatureExtractor` will consume
      `0.visual.json` when present:

```bash
.venv/bin/python -u -m learning_v2.regenerate_visual_cache \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --split learning_v2/artifacts_dataset_v3.0/phase_1_4/governed_split.json \
  --output-dir runs_final_v1/graphs/rendered_visual_v2 \
  --selection-rules image-alt label link-name color-contrast \
  --positive-fraction 0.6 \
  --minimum-positive-sites-per-rule 10 \
  --device cpu
```

- [x] Rendered graph cache completed: **858 / 858 captured**. Record visual
      mapping loss from the manifest before training:

```bash
jq '{requested_sites,outcome_counts,feature_version,selection}' \
  runs_final_v1/graphs/rendered_visual_v2/rendered_visual_cache_manifest.json
```

- [x] Implement `learning_v2.build_same_session_ax_cache`. It consumes the
      crawler's existing `0.ax.json` sidecar and never reconstructs an AX tree
      from local HTML. One-site smoke result: same-session provenance, AX
      feature version 2, 3,357 nodes, 509 features.

Build the final same-session AX cache:

- [x]

```bash
cd /Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning
source .venv/bin/activate

python -u -m learning_v2.build_same_session_ax_cache \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --split learning_v2/artifacts_dataset_v3.0/phase_1_4/governed_split.json \
  --output-dir runs_final_v1/graphs/same_session_ax_v1 \
  --selection-rules image-alt label link-name \
  --positive-fraction 0.6 \
  --minimum-positive-sites-per-rule 10 \
  --minimum-ax-mapping-ratio 0.10 \
  --device mps
```

Then audit the cache:

```bash
python -u -m learning_v2.cache_audit \
  --cache-dir runs_final_v1/graphs/same_session_ax_v1 \
  --views a11y-tree \
  --output learning_v2/artifacts_dataset_v3.0/phase_2_same_session_ax_cache_audit.json
```

### Historical AX limitation

`learning_v2.regenerate_live_ax_cache` opens saved HTML locally and blocks
network resources. It is a reconstructed saved-DOM AX view, not the original
same-session live browser AX tree required by `Plan_v3.md`.

`runs_final_v1/graphs/reconstructed_ax_v1` is a clearly labelled historical
pilot only. Do not use it for final training or Phase 3 selection.

## [x] Phase 2 exit-gate gaps before starting Phase 3 annotation

- [LATER] Add deterministic fixtures and repeat-capture tests for focus, hover,
      modal/overlay, expanded/collapsed, validation, and selected states.

  How to do it:

  1. Add one deterministic fixture per state under
     `3_Learning/learning_v2/fixtures/`, including its expected DOM marker, AX
     role/name/state, visual bounds, and screenshot-visible state.
  2. Extend the collector with actions for keyboard focus, pointer hover,
     opening/closing a modal, toggling `aria-expanded`, invalid form submission,
     and option selection. Save one evidence bundle per named state.
  3. Capture every fixture twice with the same viewport. Compare DOM-marker,
     AX-node, and visual-node mappings; an unexpected identity change fails.
  4. Add pytest cases for expected evidence and partial capture failure. A
     partial capture must be recorded as abstention, never as a pass/fail.

- [x] Record and review low AX-to-DOM mapping coverage. Across the 858 governed
      sites, the median mapped AX-node ratio is 54.98%; the minimum is 0.24%.
      Define an abstention/exclusion threshold before final analysis.

  How to do it:

  1. Retain the final `same_session_ax_cache_manifest.json`.
  2. Calculate `ax_nodes_mapped_to_snapshot / ax_nodes` for every source
     `0.ax.json` and save the distribution in the Phase 2 artifacts.
  3. The builder command above enforces the predeclared 10% policy. It writes
     exclusions to `mapping_coverage_policy.exclusions` and removes those sites
     from its derived `collection_split.json` before GraphSAGE training.
  4. Retain excluded sites for DOM/visual baselines and report every exclusion.

  Distribution command:

  ```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('../2_Data/browser-use/outputs/dataset_v3.0/axe-core')
values = []

for path in sorted(root.glob('*/0.ax.json')):
    stats = json.loads(path.read_text())['mapping_stats']
    total = stats['ax_nodes']
    mapped = stats['ax_nodes_mapped_to_snapshot']
    values.append((path.parent.name, mapped / total if total else 0.0))

print('Sites:', len(values))
print('Below 10%:', sum(ratio < 0.10 for _, ratio in values))

for site, ratio in sorted(values, key=lambda item: item[1])[:20]:
    print(f'{site}\t{ratio:.2%}')
PY
  ```


- [ONGOING] Run the same-session AX cache build and cache audit above. The audit must
      confirm only `ax_capture_provenance=same_session_sidecar` graphs.

  How to do it:

  1. Run `learning_v2.build_same_session_ax_cache` using the Phase 2 command;
     after an interruption, rerun it with `--resume`.
  2. Run `learning_v2.cache_audit --views a11y-tree` immediately afterward.
  3. Require `live_browser_accessibility_tree_proven` and
     `same_session_ax_sidecar_proven` to be `true`, and `invalid_count` to be
     zero in `phase_2_same_session_ax_cache_audit.json`.
  4. In `same_session_ax_cache_manifest.json`, every selected site must be
     `captured` or deliberately `reused`, never `missing_source` or
     `collection_failed`.

  Commands (run from `3_Learning` with `.venv` activated):

  1. [x] Build the threshold-filtered same-session AX cache:

     ```bash
     python -u -m learning_v2.build_same_session_ax_cache \
       --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
       --split learning_v2/artifacts_dataset_v3.0/phase_1_4/governed_split.json \
       --output-dir runs_final_v1/graphs/same_session_ax_v1 \
       --selection-rules image-alt label link-name \
       --positive-fraction 0.6 \
       --minimum-positive-sites-per-rule 10 \
       --minimum-ax-mapping-ratio 0.10 \
       --device mps
     ```

  2. [x] Audit the generated AX cache:

     ```bash
     python -u -m learning_v2.cache_audit \
       --cache-dir runs_final_v1/graphs/same_session_ax_v1 \
       --views a11y-tree \
       --output learning_v2/artifacts_dataset_v3.0/phase_2_same_session_ax_cache_audit.json
     ```

  3. [x] Inspect capture outcomes and the threshold exclusions:

     ```bash
     python - <<'PY'
     import json
     from pathlib import Path

     manifest = json.loads(Path('runs_final_v1/graphs/same_session_ax_v1/same_session_ax_cache_manifest.json').read_text())
     policy = manifest['mapping_coverage_policy']
     print('Requested:', manifest['requested_sites'])
     print('Eligible:', manifest['eligible_sites'])
     print('Outcomes:', manifest['outcome_counts'])
     print('Threshold:', policy['minimum_ax_mapping_ratio'])
     print('Excluded:', policy['excluded_site_count'])
     for item in policy['exclusions'][:20]:
         print(item['site_id'], f"{item.get('mapping_ratio', 0):.2%}")
     PY
     ```

  4. [x] Fail if a selected graph lacks same-session sidecar provenance:

     ```bash
     python - <<'PY'
     import json
     from pathlib import Path
     import torch

     root = Path('runs_final_v1/graphs/same_session_ax_v1')
     split = json.loads((root / 'collection_split.json').read_text())
     failures = []
     for partition in ('train', 'val', 'test'):
         for site in split[partition]:
             path = root / site / 'a11y-tree.pt'
             if not path.is_file():
                 failures.append((site, 'graph_missing'))
                 continue
             graph = torch.load(path, map_location='cpu', weights_only=False)['data']
             if getattr(graph, 'ax_capture_provenance', '') != 'same_session_sidecar':
                 failures.append((site, 'wrong_provenance'))
     print('Verified graphs:', sum(len(split[name]) for name in ('train', 'val', 'test')))
     print('Failures:', len(failures))
     for failure in failures[:20]:
         print(*failure)
     raise SystemExit(1 if failures else 0)
     PY
     ```

```     

**Phase 3 may start only after these three checks are resolved or explicitly
documented as exclusions.**

## [x] Phase 3 — Independent detection truth

- [x] After the final graph cohort and split are frozen, create a styled,
      blinded annotation packet:

```bash
.venv/bin/python -u -m learning_v2.annotation_packet \
  --split runs_final_v1/graphs/same_session_ax_v1/collection_split.json \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --registry ../configs/wcag_criteria.json \
  --output-dir learning_v2/artifacts_dataset_v3.0/detection_annotation_packet \
  --rule-ids image-alt label link-name color-contrast \
  --seed 42 --network-timeout-ms 60000
```

- [x] Complete `rater_1.json` and `rater_2.json` independently.

Example

```
{
  "case_id": "case-8586d5864e9eddddc8ad",
  "criterion_id": "1.1.1",
  "status": "fail",
  "applicable_exception": null,
  "evidence_notes": "Observed [specific non-text element] in rendered.png and source.html. It has [specific accessible-name/alt issue].",
  "confidence": 4
}
```


- [x] Adjudicate every disagreement with evidence.
- [x] Finalize truth:

```bash
.venv/bin/python -u -m learning_v2.annotation_finalize \
  --packet-dir learning_v2/artifacts_dataset_v3.0/detection_annotation_packet \
  --output learning_v2/artifacts_dataset_v3.0/detection_annotation_packet/final_independent_detection_truth.json
```

## Phase 5 — Final exact-split specialist training

The previous `phase_5_live_ax`, `phase_5_visual`, and `phase_5_multiview`
artifacts are bounded 30-site pilot runs. They must not be used for the final
study. Their checkpoints were trained on 21/4/5 sites even though the old
multiview assembly attached the 599/127/129 collector split.

### [x] Freeze the independently annotated evaluation split without modifying
the final truth file:

```bash
.venv/bin/python -u -m learning_v2.final_evaluation_split \
  --governed-split runs_final_v1/graphs/same_session_ax_v1/collection_split.json \
  --truth-file learning_v2/artifacts_dataset_v3.0/detection_annotation_packet/final_independent_detection_truth.json \
  --output learning_v2/artifacts_dataset_v3.0/final_evaluation_split.json
```

Expected: 599 train, 127 validation, 101 test, 404 truth pairs, and 28
documented exclusions with no imputed labels.


### [x] Train feature-matched accessibility-tree models only after the
      same-session cache audit passes:

```bash
.venv/bin/python -u -m learning_v2.experiment \
  --cache-dir runs_final_v1/graphs/same_session_ax_v1 \
  --inventory learning_v2/artifacts_dataset_v3.0/phase_1_4/corpus_inventory.json \
  --split learning_v2/artifacts_dataset_v3.0/final_evaluation_split.json \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_5_live_ax_final_v2 \
  --views a11y-tree --architectures mlp graphsage gat \
  --rule-ids image-alt label link-name \
  --split-mode governed --max-nodes 0 \
  --min-train-positive-sites 5 --min-val-positive-sites 2 \
  --epochs 20 --patience 5 --batch-size 2 \
  --hidden 64 --layers 2 --precision-floor 0.25 \
  --seed 42 --device mps
```

### [x] Train rendered specialists:

```bash
.venv/bin/python -u -m learning_v2.experiment \
  --cache-dir runs_final_v1/graphs/rendered_visual_v2 \
  --inventory learning_v2/artifacts_dataset_v3.0/phase_1_4/corpus_inventory.json \
  --split learning_v2/artifacts_dataset_v3.0/final_evaluation_split.json \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_5_visual_final_v2 \
  --views rendered-visual --architectures mlp graphsage gat \
  --rule-ids color-contrast \
  --split-mode governed --max-nodes 0 \
  --min-train-positive-sites 5 --min-val-positive-sites 2 \
  --epochs 20 --patience 5 --batch-size 1 \
  --hidden 64 --layers 2 --precision-floor 0.8 \
  --seed 42 --device mps
```

The run must stop explicitly if a requested rule lacks support or if no
validation threshold meets the predeclared precision floor. Do not add the
pilot-only override flags for final training.

### [o] Run the visual ablation after the final rendered run succeeds:

The final rendered specialist retains its strict precision-floor gate. An
ablated variant that cannot attain that floor uses its validation-only
maximum-F1 fallback threshold; the unmet floor remains explicitly recorded in
that run's `calibration.json`, `manifest.json`, and the aggregate ablation
report. This is an ablation measurement, not a deployable-model acceptance.

```bash
.venv/bin/python -u -m learning_v2.visual_ablation \
  --cache-dir runs_final_v1/graphs/rendered_visual_v2 \
  --split learning_v2/artifacts_dataset_v3.0/phase_5_visual_final_v2/governed_split.json \
  --output-dir learning_v2/artifacts_dataset_v3.0/visual_ablation_final_v2 \
  --rule-ids color-contrast \
  --variants full without_visual_features without_spatial_edges structure_only \
  --architectures mlp graphsage --seeds 41 42 43 \
  --epochs 20 --patience 5 --batch-size 1 --bootstrap-samples 5000 --device mps \
  --resume
```

### [ ] Assemble aligned views. The assembler now rejects model/split mismatch:

```bash
.venv/bin/python -u -m learning_v2.assemble_multiview_bundle \
  --split learning_v2/artifacts_dataset_v3.0/final_evaluation_split.json \
  --view-cache a11y-tree=runs_final_v1/graphs/same_session_ax_v1 \
  --view-cache rendered-visual=runs_final_v1/graphs/rendered_visual_v2 \
  --view-model a11y-tree=learning_v2/artifacts_dataset_v3.0/phase_5_live_ax_final_v2 \
  --view-model rendered-visual=learning_v2/artifacts_dataset_v3.0/phase_5_visual_final_v2 \
  --output-cache runs_final_v1/graphs/multiview_final_v2 \
  --output-phase5 learning_v2/artifacts_dataset_v3.0/phase_5_multiview_final_v2
```

## Phases 6–7 — Routing, fusion, and detection study

### [ ] Run a weak-label diagnostic, not a final superiority claim:

```bash
.venv/bin/python -u -m learning_v2.study \
  --phase5-dir learning_v2/artifacts_dataset_v3.0/phase_5_multiview_final_v2 \
  --cache-dir runs_final_v1/graphs/multiview_final_v2 \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --registry ../configs/wcag_criteria.json \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_6_7_weak_label \
  --views a11y-tree rendered-visual \
  --architectures mlp graphsage gat \
  --truth-source axe_weak_labels \
  --bootstrap-samples 2000 --seed 42 --device cpu
```

### [x] After independent truth is finalized, consume the test set once:

```bash
.venv/bin/python -u -m learning_v2.study \
  --phase5-dir learning_v2/artifacts_dataset_v3.0/phase_5_multiview_final_v2 \
  --cache-dir runs_final_v1/graphs/multiview_final_v2 \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --registry ../configs/wcag_criteria.json \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_6_7_final \
  --views a11y-tree rendered-visual \
  --architectures mlp graphsage gat \
  --truth-source independent_manual \
  --truth-file learning_v2/artifacts_dataset_v3.0/detection_annotation_packet/final_independent_detection_truth.json \
  --final --bootstrap-samples 5000 --seed 42 --device cpu
```

## Phase 8 — Retrieval

### [x] Rebuild the training-only index, queries, retrieval evaluation, and
      generator inputs:

```bash
.venv/bin/python -u -m accessibility_system.phase8 \
  --phase5-dir learning_v2/artifacts_dataset_v3.0/phase_5_multiview_final_v2 \
  --split learning_v2/artifacts_dataset_v3.0/phase_5_multiview_final_v2/governed_split.json \
  --inventory learning_v2/artifacts_dataset_v3.0/phase_1_4/corpus_inventory.json \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_8 \
  --top-k 5 --context-characters 5000 --max-queries 100

jq '.exit_gate' \
  learning_v2/artifacts_dataset_v3.0/phase_8/phase_8_retrieval_evaluation.json
```

Every Phase 8 exit-gate value must be `true`.

## Phase 9 — Structured repair generation and validation

### [ ] Smoke-test one proposal first:

```bash
.venv/bin/python -u -m accessibility_system.phase9 \
  --generator-inputs learning_v2/artifacts_dataset_v3.0/phase_8/generator_inputs.json \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_9_smoke \
  --condition graph_constrained_rag \
  --api-mode chat_completions \
  --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash \
  --max-proposals 1 --skip-browser --log-level INFO
```

- [ ] Run `no_rag`, `flat_vector_rag`, and `graph_constrained_rag` separately,
      using the same model, proposal count, retries, and browser-validation
      policy. Change only `--condition` and `--output-dir`:

```bash
.venv/bin/python -u -m accessibility_system.phase9 \
  --generator-inputs learning_v2/artifacts_dataset_v3.0/phase_8/generator_inputs.json \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_9/graph_constrained_rag \
  --condition graph_constrained_rag \
  --api-mode chat_completions \
  --base-url https://openrouter.ai/api/v1 \
  --model deepseek/deepseek-v4-flash \
  --max-proposals 30 --generation-retries 1 --log-level INFO
```

## Phase 10 — Matched repair study

### [ ] Run the deterministic control through the same validator:

```bash
.venv/bin/python -u -m accessibility_system.deterministic_repair \
  --generator-inputs learning_v2/artifacts_dataset_v3.0/phase_8/generator_inputs.json \
  --corpus-dir ../2_Data/browser-use/outputs/dataset_v3.0/axe-core \
  --axe-js ../2_Data/browser-use/axe-core.min.js \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_9/deterministic_template \
  --max-proposals 30
```

### [ ] Aggregate matched conditions after all four Phase 9 reports exist:

```bash
.venv/bin/python -u -m accessibility_system.evaluation.repair_study \
  --run deterministic_template=learning_v2/artifacts_dataset_v3.0/phase_9/deterministic_template/phase_9_report.json \
  --run no_rag=learning_v2/artifacts_dataset_v3.0/phase_9/no_rag/phase_9_report.json \
  --run flat_vector_rag=learning_v2/artifacts_dataset_v3.0/phase_9/flat_vector_rag/phase_9_report.json \
  --run graph_constrained_rag=learning_v2/artifacts_dataset_v3.0/phase_9/graph_constrained_rag/phase_9_report.json \
  --generator-inputs learning_v2/artifacts_dataset_v3.0/phase_8/generator_inputs.json \
  --output-dir learning_v2/artifacts_dataset_v3.0/phase_10 \
  --bootstrap-samples 5000 --seed 42
```

### [ ] Create and complete the condition-blinded human repair-rating packet.
### [ ] Rerun the Phase 10 aggregation with finalized human ratings.

## Final verification

### [ ] Run the complete suite and preserve JUnit evidence:

```bash
.venv/bin/python -m pytest -q \
  --junitxml=learning_v2/artifacts_dataset_v3.0/verification/pytest.xml
```

## Current blockers before a dissertation-final run

1. Phase 1 does not yet require `0.visual.json` when admitting a site.
2. Phase 2 still needs controlled dynamic-state fixture/identity tests and a
   declared low-mapping abstention or exclusion threshold.
3. Detection and repair annotation sheets require two independent human raters
   and adjudication.

Until blockers 1–3 are resolved, the new run may be described as a pilot but
not as the final `Plan_v3.md` evidence run.
