## Option A: Homogeneous DOM Tree (Simplest)

**Graph Schema**
- **Nodes**: Every HTML element (`div`, `a`, `img`, `p`, etc.) + text nodes
- **Edges**: Directed parent→child relationships only
- **PyG Class**: `torch_geometric.data.Data`

**Node Features** (per node):
| Feature | Encoding |
|---|---|
| Tag type | One-hot (or learned embedding) |
| Has text? | Boolean |
| Text length | Scalar |
| Attribute count | Scalar |
| Has `id` / `class` | Boolean flags |
| ARIA attributes | Boolean flags (`aria-label`, `role`, etc.) |

**Edge Features**: None, or edge type index

**GNN Model**: `GCNConv` or `GATConv`

**Best for**: Quick prototypes, element classification (e.g., "is this a navigation item?")

---

## Option B: Heterogeneous DOM + Structure (Intermediate)

**Graph Schema**
- **Node Types**:
  - `element`: HTML tags
  - `text`: Text content nodes
  - `virtual`: Single global page node (like Baidu's Virt-GAT)
- **Edge Types**:
  - `element` → `element` (parent-child)
  - `element` → `text` (contains text)
  - `element` → `element` (next-sibling)
  - `virtual` → `element` (global context)

**PyG Class**: `torch_geometric.data.HeteroData`

**Node Features**:
| Node Type | Features |
|---|---|
| `element` | Tag embedding (BERT/XPath-like), class/id embedding, bounding box (if rendered), accessibility role |
| `text` | BERT sentence embedding of text content |
| `virtual` | Learnable embedding |

**GNN Model**: `HGTConv` or `RGCNConv`

**Best for**: Accessibility violation detection (node-level), element relationship reasoning (like AccessFixer)

---

## Option C: Multimodal DOM + Visual + Hyperlink (GRASP-Style, Most Complex)

**Graph Schema**
- **Node Types**:
  - `dom_element`: Parsed HTML elements
  - `visual_region`: Rendered bounding boxes from screenshot (via Playwright)
  - `page`: Top-level web page node
- **Edge Types**:
  - `dom_element` → `dom_element` (parent-child)
  - `dom_element` → `dom_element` (spatial: above, below, left-of)
  - `dom_element` → `visual_region` (DOM-to-rendered mapping)
  - `page` → `dom_element` (page membership)
  - `page` → `page` (hyperlink, for multi-page graphs)

**PyG Class**: `HeteroData`

**Node Features**:
| Node Type | Features |
|---|---|
| `dom_element` | BERT(text_DOM), tag embedding, attribute embeddings |
| `visual_region` | ViT(screenshot_crop), bounding box coords, color histogram |
| `page` | Aggregated DOM features, URL embedding |

**GNN Model**: Multi-layer `HGTConv` with separate projection layers per node type

**Best for**: Page-level classification, representative page sampling (replicating GRASP)

---

## Key Technical Details for PyG

### 1. Building `edge_index` from DOM
```python
# Parse HTML with BeautifulSoup or lxml
# Assign each element an integer ID
# For parent-child edges:
edge_index = torch.tensor([
    [parent_id_1, parent_id_2, ...],  # source
    [child_id_1,  child_id_2,  ...]   # target
], dtype=torch.long)
```

### 2. Handling Variable Text Length
- **Option 1**: Pre-compute BERT embeddings per node, truncate/pad to fixed `d_text`
- **Option 2**: Use a text embedding lookup table if vocabulary is small

### 3. Tag Encoding
```python
# All HTML tags ~100 unique
tag_vocab = {tag: idx for idx, tag in enumerate(all_html_tags)}
tag_embedding = nn.Embedding(num_tags, embedding_dim)
```

### 4. Libraries Needed
| Purpose | Library |
|---|---|
| HTML parsing | `beautifulsoup4`, `lxml` |
| Text embeddings | `transformers` (BERT) |
| Visual embeddings | `timm` or `transformers` (ViT) — only if using screenshots |
| Graph ML | `torch_geometric` |
| Rendering (visual) | `playwright` or `selenium` |

---

## Questions Before Implementation

1. **What is your prediction task?**
   - Node-level: classify each element (e.g., "this is a form input")
   - Graph-level: classify the whole page (e.g., "this page has accessibility violations")
   - Link-level: predict missing relationships

2. **Do you need rendered/visual features?**
   - Yes → Requires Playwright to get bounding boxes + screenshots (like GRASP)
   - No → Pure DOM-only, much faster to build

3. **Single page or multi-page graph?**
   - `bt.com.html` is one page. Do you want to model the full website hyperlink graph eventually?

4. **Accessibility focus?**
   - Should node features include ARIA attributes, axe-core violation labels, accessibility tree roles?

5. **Which PyG architecture appeals to you?**
   - Simple: `GCNConv` on homogeneous graph
   - Flexible: `HGTConv` on heterogeneous graph
   - GRASP-like: Custom message passing with separate modalities



1. **What is your prediction task?**
   - Graph-level: classify the whole page (e.g., "this page has accessibility violations")

2. **Do you need rendered/visual features?**
  BOTH

3. **Single page or multi-page graph?**
   - `bt.com.html` is one page. Do you want to model the full website hyperlink graph eventually?
   - YES

4. **Accessibility focus?**
   - Should node features include ARIA attributes, axe-core violation labels, accessibility tree roles?
   - YES

5. **Which PyG architecture appeals to you?**
   - Simple: `GCNConv` FOR NOW
