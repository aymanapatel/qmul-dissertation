# GNN Architecture

This project builds PyTorch Geometric graphs from web pages, then trains a multi-task Graph Attention Network to predict accessibility violations at node, rule, and page level.

## 1. Graph Construction

The training entrypoint is `3_Learning/scripts/train_multi_site.py`. For each site directory, it expects:

- `0.html`: the captured page HTML
- `page-0_home.json`: the axe-core report

Each page is processed into a PyG `Data` object by `FeatureExtractor.process_page()` in `3_Learning/src/feature_extractor.py`.

### Graph Sources

The graph source is selected with `--graph-source`.

### DOM Graph Source

Flag value: `dom`.

For the DOM graph, `html_graph_builder.parse_html_to_graph()` parses the HTML with BeautifulSoup and creates:

- One graph node per HTML element.
- Parent-to-child edges from the DOM tree.
- Sibling edges to preserve local reading/order structure.
- Optional spatial edges when visual extraction is enabled.

The resulting `Data` object contains:

```text
x              [num_nodes, attr_dim + text_dim]
edge_index     [2, num_edges]
tag_indices    [num_nodes]
node_y         [num_nodes]
node_y_multi   [num_nodes, num_wcag_rules]
y              [1]
```
### A11y Tree Graph Source

Flag value: `a11y-tree`.

The a11y-tree source is implemented in `3_Learning/src/accessibility_graph_builder.py`.
It builds a static accessibility-tree-style graph from the parsed HTML:

- Nodes are semantically relevant/accessibility-exposed elements rather than every DOM element.
- Roles are inferred from explicit `role`, HTML tag, and input type.
- Accessible names are inferred from `aria-labelledby`, `aria-label`, `alt`, `title`, `placeholder`, `value`, or visible text.
- Hidden/presentational/script/style/template nodes are skipped.
- Edges connect each included node to its nearest included accessibility ancestor.
- Adjacent accessible siblings receive sibling edges.

The a11y graph still keeps the original parsed element on each node so axe selectors can be mapped back to graph nodes for weak labels.

### Node Features

Each node receives three kinds of information:

1. Tag identity
   The HTML tag is stored as `tag_indices`, then embedded inside the model using a learned embedding table.

2. Attribute and accessibility flags
   `DOMNode.get_attribute_features()` creates binary/numeric features such as:
   - `id`, `class`, `href`, `src`, `alt`, `title`
   - ARIA attribute presence
   - semantic role indicators
   - text length and has-text flag
   - image/interactive/heading/landmark/form-related flags
   - optional visual bounding-box and visibility features

3. Text embeddings
   `FeatureExtractor.extract_text_embeddings()` uses `sentence-transformers/all-MiniLM-L6-v2`.
   The text embedding dimension is usually `384`.

The final `data.x` is:

```text
attribute_features + text_embedding
```

The tag embedding is not stored in `data.x`; it is learned by the model and concatenated during the forward pass.

### Labels

Axe-core labels are loaded from the JSON report:

- `node_y`: binary label, `1` if the DOM node has any mapped axe violation.
- `node_y_multi`: multi-hot WCAG/axe rule labels per node.
- `y`: graph-level binary label, `1` if any node on the page violates.

The axe target selectors are matched back to parsed DOM nodes using exact CSS selector matching first, then ID/class/tag heuristics.

## 2. Model Construction

The active model is `DOMAttentionNet` in `3_Learning/src/models.py`.

It is constructed in `train_multi_site.py` as:

```python
model = DOMAttentionNet(
    num_tags=116,
    tag_embed_dim=32,
    attr_dim=attr_dim,
    text_dim=text_dim,
    hidden_dim=args.hidden,
    num_node_classes=2,
    num_graph_classes=2,
    num_rules=NUM_RULES,
    num_layers=args.layers,
    heads=args.heads,
    dropout=args.dropout,
    pooling=args.pooling,
)
```

Current defaults are:

```text
hidden_dim = 256
num_layers = 4
heads      = 4
dropout    = 0.3
pooling    = mean
num_rules  = 46
```

## 3. Forward Pass

The model receives:

```text
x            [N, attr_dim + text_dim]
edge_index   [2, E]
tag_indices  [N]
batch        [N], optional
```

The forward pass is:

1. Embed HTML tags

   ```text
   tag_indices -> Embedding(num_tags, 32)
   ```

2. Concatenate tag, attribute, and text features

   ```text
   [tag_embedding | data.x]
   ```

3. Project into hidden space

   ```text
   Linear(input_dim, hidden_dim)
   BatchNorm1d
   ReLU
   Dropout
   ```

4. Run GAT message passing

   The model uses `num_layers` stacked `GATConv` layers.

   Each layer is:

   ```text
   GATConv(hidden_dim, hidden_dim / heads, heads=heads, concat=True)
   BatchNorm1d(hidden_dim)
   ReLU
   Dropout
   Residual connection
   ```

   With the default `hidden_dim=256` and `heads=4`, each attention head produces `64` channels, then heads are concatenated back to `256`.

5. Produce three outputs

   ```text
   node_logits       [N, 2]
   node_rule_logits  [N, 46]
   graph_logits      [num_graphs, 2]
   ```

## 4. Prediction Heads

The model is multi-task.

### Node Violation Head

Predicts whether each DOM node is violating:

```text
Linear(256, 128)
ReLU
Dropout
Linear(128, 2)
```

Output:

```text
node_logits [N, 2]
```

Training uses cross-entropy with a capped positive class weight.

### Node Rule Head

Predicts which axe/WCAG rules apply to each node:

```text
Linear(256, 256)
ReLU
Dropout
Linear(256, 128)
ReLU
Dropout
Linear(128, num_rules)
```

Output:

```text
node_rule_logits [N, 46]
```

Training uses binary cross-entropy with logits, because each node may have multiple rule labels.

### Graph/Page Head

Predicts whether the whole page has any violation.

The node embeddings are pooled into a graph embedding using one of:

- `mean`
- `max`
- `meanmax`
- `attention`

The default is mean pooling:

```text
global_mean_pool(node_embeddings, batch)
```

Then:

```text
Linear(graph_dim, 256)
ReLU
Dropout
Linear(256, 128)
ReLU
Dropout
Linear(128, 2)
```

Output:

```text
graph_logits [num_graphs, 2]
```

## 5. Losses and Training Objective

Training is handled by `Trainer` in `3_Learning/src/train.py`.

The total loss is a weighted sum:

```text
total_loss =
    node_loss_weight  * node_loss
  + rule_loss_weight  * rule_loss
  + graph_loss_weight * graph_loss
```

Current important training controls:

- `node_pos_weight_cap`: caps the positive weight for rare violating nodes.
- `rule_pos_weight`: controls positive class pressure for rule labels.
- `node_threshold`: threshold used for validation node metrics.
- `rule_threshold`: threshold used for validation rule metrics.
- `selection_metric`: metric used for checkpointing, currently `node_f1_pos`.

The current checkpoint strategy saves:

- `best_model.pt`: best validation checkpoint by `selection_metric`.
- `last_model.pt`: latest epoch checkpoint for crash recovery.

## 6. High-Level Diagram

```text
HTML + axe JSON
      |
      v
FeatureExtractor
      |
      v
PyG Data
  x, edge_index, tag_indices, node_y, node_y_multi, y
      |
      v
DOMAttentionNet
      |
      +--> Node violation logits       [N, 2]
      |
      +--> Node rule logits            [N, 46]
      |
      +--> Graph/page violation logits [B, 2]
```
