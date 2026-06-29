# DOM-to-GNN: HTML Graph Neural Network for Web Accessibility

Original implementation for representing HTML/DOM as PyTorch Geometric graphs for accessibility violation detection.

## Overview

This system converts HTML web pages into graph representations where:
- **Nodes** = HTML elements (`<div>`, `<a>`, `<img>`, etc.)
- **Edges** = Structural relationships (parent-child, sibling, spatial)
- **Node Features** = Tag embeddings + text embeddings + attributes + visual bounding boxes + axe violation labels
- **Tasks** = Node-level (element has violation?) + Graph-level (page has violations?)

## Architecture

### Graph Construction (`src/html_graph_builder.py`)
- Parses HTML via BeautifulSoup4
- Creates directed edges for parent-child and sibling relationships
- Optional spatial edges from Playwright-rendered bounding boxes

### Feature Extraction (`src/feature_extractor.py`)
- **Text**: Sentence-BERT (`all-MiniLM-L6-v2`, 384D) on visible inner text
- **Tags**: Learned embedding layer (~116 tags)
- **Attributes**: Binary flags for IDs, classes, ARIA attributes, semantic roles
- **Visual**: Bounding boxes [x, y, w, h] from Playwright rendering
- **Labels**: axe-core violation mapping via CSS selector matching

### Model (`src/models.py`)
```
DOMGCN:
  Tag Embedding -> Input Projection -> GCNConv x3 -> Global Pooling -> Classifier
```

Supports:
- `GCNConv` (default)
- `GATConv` (attention-based variant)
- Node-level + graph-level dual heads

### Training (`src/train.py`)
- Combined node + graph loss with class weighting for imbalance
- Metrics: Accuracy, F1, Precision, Recall, AUC
- Early stopping on validation F1

## Quick Start

### 1. Process a Page
```bash
source .venv/bin/activate
python scripts/process_page.py \
  --html /path/to/page.html \
  --axe /path/to/axe_report.json \
  --output ./graphs \
  --visual \
  --train
```

This will:
- Parse the HTML into a graph
- Extract text embeddings (BERT)
- Render the page and extract bounding boxes (Playwright)
- Map axe-core violations to DOM nodes
- Save the processed graph to `./graphs/page_graph.pt`
- Run a forward pass demo

### 2. Train on Single Page (with augmentation)
```bash
python scripts/train_single_page.py \
  --graph ./graphs/page_graph.pt \
  --epochs 50 \
  --hidden 64 \
  --layers 2 \
  --save ./models
```

Augmentation strategies (since we only have one page):
- Random edge dropout
- Node feature masking
- Subgraph sampling

### 3. Visualize Predictions
```bash
python scripts/visualize_graph.py \
  --graph ./graphs/page_graph.pt \
  --model ./models/best_model.pt \
  --output ./visualizations \
  --max-nodes 150
```

## Example: bt.com

/Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core/www.1mg.com

```bash
python3 scripts/process_page.py \
  --html /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core/www.1mg.com/0.html \
  --axe /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core/www.1mg.com/page-0_home.json \
  --output ./graphs \
  --visual \
  --train
```

**Results:**
- Total nodes: 1,433
- Total edges: 807,452
- Violating nodes: 9 / 1,433
- Graph label: VIOLATIONS
- Most common tags: div (528), span (258), a (174)

## File Structure

```
3_Learning/
├── src/
│   ├── html_graph_builder.py    # HTML -> PyG Data
│   ├── feature_extractor.py     # BERT, Playwright, axe mapping
│   ├── models.py                # GCN/GAT models
│   ├── train.py                 # Training loop
│   └── __init__.py
├── scripts/
│   ├── process_page.py          # Full pipeline CLI
│   ├── train_single_page.py     # Training CLI
│   └── visualize_graph.py       # Visualization CLI
├── graphs/                      # Saved processed graphs
├── models/                      # Saved model checkpoints
└── visualizations/              # Output plots
```

## Key Design Decisions

1. **Class Imbalance**: Accessibility violations are rare (~0.6% of nodes). We use class-weighted cross-entropy loss.

2. **Single-Page Training**: With only one page, we create augmented variants via edge dropout, feature masking, and subgraph sampling.

3. **No GRASP/AAA Reuse**: This is an original implementation. GRASP was used for research inspiration only.

4. **Modality Choice**: We use sentence-transformers (MiniLM) instead of full BERT for speed, with option to upgrade.

## Future Extensions

- Multi-page site graphs with hyperlink edges
- Visual features via ViT on screenshots
- Heterogeneous graphs with separate element/text/visual nodes
- Self-supervised pre-training (masked element prediction)
- Integration with MLLM copilots (MaC-style)

## Dependencies

- PyTorch + PyTorch Geometric
- BeautifulSoup4 + lxml
- sentence-transformers
- Playwright (for rendering)
- scikit-learn, matplotlib, pandas


# Python running




## Single page training
```
# Crawl bt.com (using your browser-use pipeline)
cd /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use
python main.py --site https://www.bt.com

# Then process the output
cd /Users/aymanpatel/Desktop/Uni/Dissertation/3_Learning
source .venv/bin/activate

```
python3 scripts/process_page.py \
  --html /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core/www.bt.com/0.html \
  --axe /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core/www.bt.com/page-0_home.json \
  --output ./graphs \
  --visual

python3 scripts/train_single_page.py \
  --graph ./graphs/0_graph.pt \
  --epochs 100 \
  --hidden 128 \
  --layers 3 \
  --save ./models

python3 scripts/visualize_graph.py \
  --graph ./graphs/0_graph.pt \
  --model ./models/best_model.pt \
  --output ./visualizations

```

##  Batch script  training


### Step 1: Multi site training:




```python
python3 scripts/train_multi_site.py --data-dir /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core --output-dir ./graphs_multi --model-dir ./models_multi_test --max-sites 20 --epochs 3 --batch-size 4 --device mps
 ```


### 2. Single page prediction


```python
python scripts/predict_site.py \
  --html /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core/www.aggrid.com/0.html \
  --axe /Users/aymanpatel/Desktop/Uni/Dissertation/2_Data/browser-use/outputs/axe-core/www.aggrid.com/page-0_home.json \
  --model ./models_multi/best_model.pt \
  --output ./reports/prediction.json \
  --threshold 0.5
```
