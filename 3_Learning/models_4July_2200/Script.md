

# Training 

```python

python scripts/train_multi_site.py \
  --data-dir ../2_Data/browser-use/outputs/axe-core \
  --output-dir ./graphs_multi_4July_2200 \
  --model-dir ./models_4July_2200 \
  --graph-source a11y-tree \
  --resume \
  --epochs 30 \
  --batch-size 4 \
  --device mps \
  --lr 1e-4 \
  --node-pos-weight-cap 8 \
  --rule-pos-weight 12 \
  --rule-loss-weight 2 \
  --hard-neg-weight 2 \
  --hard-pos-weight 2 \
  --rule-threshold 0.25 \
  --positive-page-node-evidence-loss-weight 0.1 \
  --graph-node-consistency-loss-weight 0.1 \
  --clean-page-node-loss-weight 2.0 \
  --selection-metric node_f1_pos
 ``` 

# Predict

```python

python scripts/predict_site.py \
  --html ../2_Data/browser-use/outputs/axe-core/www.impots.gouv.fr/0.html \
  --axe ../2_Data/browser-use/outputs/axe-core/www.impots.gouv.fr/page-0_home.json \
  --model ./models_4July_2200//best_model.pt \
  --calibration ./models_4July_2200/calibration.json \
  --output ./reports/prediction_impot_a11y_v5.json \
  --graph-source a11y-tree \
  --device mps

```  