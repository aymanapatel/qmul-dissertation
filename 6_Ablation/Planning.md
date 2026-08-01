For v1, run controlled “remove one component” experiments against the same rendered-visual baseline.

Keep these constant:

- The exact `split.json`
- Cached graphs
- Seed initially `42`
- Batch size, epochs, patience, architecture, and selection metric
- Evaluate only once on the test split after checkpoint selection

Recommended v1 ablations:

| Run | Change | Question answered |
|---|---|---|
| Full baseline | No change | Reference performance |
| No graph loss | `--graph-loss-weight 0` | Does page-level supervision help? |
| No consistency losses | Set all three auxiliary weights to `0` | Do node/page consistency constraints help? |
| No rule hard mining | `--hard-neg-weight 1 --hard-pos-weight 1` | Does hard-example weighting help? |
| No node hard-negative mining | `--node-hard-negative-ratio 0` | Does node hard-negative selection help? |
| No message passing | `--layers 0` | Does the GAT graph structure help over node features alone? |

Use a separate model directory for every run. For example:

```bash
cd /Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning

mkdir -p ./ablations_v1/full
cp ./models_20July_2200/split.json ./ablations_v1/full/split.json

.venv/bin/python -u scripts/train_multi_site.py \
  --architecture multi-view \
  --multi-view-sources rendered-visual \
  --reuse-split \
  --data-dir ../2_Data/browser-use/outputs/axe-core \
  --output-dir ./graphs_multi_20July_2200 \
  --model-dir ./ablations_v1/full \
  --resume \
  --epochs 60 \
  --patience 15 \
  --batch-size 2 \
  --device mps \
  --selection-metric node_f1_pos \
  --rule-loss-weight 5 \
  --node-loss-weight 1 \
  --graph-loss-weight 1 \
  --node-hard-negative-ratio 4 \
  --clear-cache-every 1 \
  --seed 42 \
  2>&1 | tee ./ablations_v1/full.console.log
```

Then run an ablation by copying the same split and changing only the relevant argument. For example, removing the three consistency objectives:

```bash
mkdir -p ./ablations_v1/no_consistency
cp ./models_20July_2200/split.json ./ablations_v1/no_consistency/split.json

.venv/bin/python -u scripts/train_multi_site.py \
  --architecture multi-view \
  --multi-view-sources rendered-visual \
  --reuse-split \
  --data-dir ../2_Data/browser-use/outputs/axe-core \
  --output-dir ./graphs_multi_20July_2200 \
  --model-dir ./ablations_v1/no_consistency \
  --resume \
  --epochs 60 \
  --patience 15 \
  --batch-size 2 \
  --device mps \
  --selection-metric node_f1_pos \
  --rule-loss-weight 5 \
  --node-loss-weight 1 \
  --graph-loss-weight 1 \
  --node-hard-negative-ratio 4 \
  --clean-page-node-loss-weight 0 \
  --positive-page-node-evidence-loss-weight 0 \
  --graph-node-consistency-loss-weight 0 \
  --clear-cache-every 1 \
  --seed 42 \
  2>&1 | tee ./ablations_v1/no_consistency.console.log
```

Apply the same pattern for the remaining runs:

```bash
# No graph-level supervision
--graph-loss-weight 0

# No rule hard-example weighting
--hard-neg-weight 1 --hard-pos-weight 1

# No node hard-negative mining
--node-hard-negative-ratio 0

# No GAT message passing: feature-only baseline
--layers 0
```

Results will be available at:

```text
ablations_v1/<run>/rendered-visual/test_metrics.json
```

Report at minimum:

- Node precision, recall and positive-class F1
- Rule micro-F1 and macro-F1
- Graph F1
- Change relative to the full baseline

For dissertation-quality results, repeat each configuration with seeds `42`, `43`, and `44`, retaining the same copied split, and report mean ± standard deviation. A true “no visual features” ablation is not currently exposed by a v1 flag; changing from `rendered-visual` to `dom` would alter features, edges, and rule ownership simultaneously, so it would not isolate the contribution of visual features.