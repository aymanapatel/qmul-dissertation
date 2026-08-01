# Running the 20 July training

The crawling and rendered-visual graph generation are already complete. The cache at
`3_Learning/graphs_multi_20July_2200` contains 702 rendered-visual graphs.

Run the following commands from the dissertation root:

```bash
cd /Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning

# Create the new model bundle once, retaining the trained a11y-tree and DOM specialists.
test -e ./models_20July_2200 || cp -R ./models_5July_2200 ./models_20July_2200

.venv/bin/python scripts/train_multi_site.py \
  --architecture multi-view \
  --multi-view-sources rendered-visual \
  --reuse-split \
  --data-dir ../2_Data/browser-use/outputs/axe-core \
  --output-dir ./graphs_multi_20July_2200 \
  --model-dir ./models_20July_2200 \
  --resume \
  --epochs 100 \
  --patience 20 \
  --batch-size 2 \
  --device mps \
  --selection-metric node_f1_pos \
  --rule-loss-weight 5 \
  --node-loss-weight 1 \
  --graph-loss-weight 1 \
  --node-hard-negative-ratio 4 \
  --clear-cache-every 1
```



`--resume` loads the existing cached graphs, so the completed crawling and graph
generation work is not repeated. `--reuse-split` retains the train, validation, and
test split copied from `models_5July_2200`.

Training outputs are written to `3_Learning/models_20July_2200/rendered-visual`.
The completed multi-view bundle is described by
`3_Learning/models_20July_2200/manifest.json`.

If training is interrupted after a checkpoint has been written, rerun the training
command with `--resume-training` appended. For lower Apple Silicon memory pressure,
also change `--batch-size 4` to `--batch-size 2` and
`--clear-cache-every 20` to `--clear-cache-every 1`.
