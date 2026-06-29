#!/bin/bash
# train_all_sites.sh - Full training on all 700 sites with multi-label GAT

cd /Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning
source .venv/bin/activate

# Use MPS (Apple GPU) for faster training
python scripts/train_multi_site.py \
  --data-dir /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core \
  --output-dir ./graphs_multi \
  --model-dir ./models_multi \
  --epochs 100 \
  --batch-size 8 \
  --hidden 256 \
  --layers 4 \
  --heads 4 \
  --patience 30 \
  --node-loss-weight 1.0 \
  --rule-loss-weight 10.0 \
  --graph-loss-weight 0.5 \
  --focal-gamma 2.0 \
  --hard-neg-weight 10.0 \
  --hard-pos-weight 5.0 \
  --device auto \
  --resume \
  "$@"
