Graph Neural Networks Applied to Websites and Web Interfaces
Executive summary
This survey covers papers that apply Graph Neural Networks directly to web graphs, HTML/DOM trees, semi-structured webpages, web navigation environments, and web-style UI layouts. The strongest and most consistent line of work models page-local structure rather than the open web at Internet scale: nodes are usually HTML elements, text fields, or UI components, and edges encode DOM hierarchy, spatial adjacency, or explicit layout constraints.

Across tasks, the recurring lesson is that structure helps most when it is semantically meaningful. DOM parent-child edges alone are often useful for classification and quality scoring, but papers that add spatial or relational edges tend to do better on extraction and layout-sensitive tasks. ZeroShotCeres improves zero-shot webpage relation extraction by combining horizontal, vertical, and DOM edges; Graph4GUI improves GUI autocompletion by introducing heterogeneous graphs with explicit constraint nodes; AccessFixer uses relational UI graphs so repairs can propagate consistently across related components.

The best-supported correlation with success is the combination of rich node features + explicit relational structure + graph-level aggregation or pre-training. GROWN+UP shows that deep DOM-based GNNs with self-supervised pre-training transfer across two very different webpage tasks; the Baidu webpage-quality paper shows that adding a virtual node and category-aware optimization improves real search ranking metrics; DOM-Q-NET shows that a GNN over the DOM can make RL-based web navigation more sample-efficient, especially in multi-task settings.

The field is promising but fragmented. Benchmarks are split across webpage ranking, relation extraction, boilerplate removal, genre classification, search quality, navigation, and UI design. Several datasets are small, dated, proprietary, or weakly standardized; some papers omit exact architectural detail or code; and cross-paper comparisons are often imperfect because tasks and metrics differ sharply.

The publication timeline below summarizes the papers covered here. Dates and venues come from the linked paper or official project pages.

2005
Graph NeuralNetworks forRanking Web Pages
2019
DOM-Q-NET
2020
ZeroShotCeres
2021
Web Image ContextExtraction on theDOM Tree
2022
GROWN+UP
2023
Layout-awareWebpage QualityAssessment
PLM-GNN
2024
Graph4GUI
2025
AccessFixer
Selected GNN papers on websites, webpages, DOMs, and web UIs


Show code
Scope and inclusion
I included papers that apply a GNN to one of four objects: a web graph of linked pages, a webpage DOM or semi-structured webpage graph, a web navigation environment whose state/action space is the DOM, or a web-style UI graph that is structurally analogous to webpages and DOMs. I prioritized peer-reviewed venues and arXiv preprints with accessible official pages. I excluded broader “web knowledge graph” papers that are not actually about webpage or website structure.

Two graph schemas recur throughout the literature. The first is the DOM-style page graph, where nodes are DOM elements or text fields and edges encode hierarchy and sometimes spatial adjacency. The second is the constraint-style heterogeneous UI graph, where element nodes connect to explicit alignment, size, or grouping constraints. These two schemas explain most of the modeling differences across the surveyed papers.

child
child
spatial or sibling
virtual/global
DOM or text node
DOM or text node
DOM or text node
DOM or text node
Graph-level context


Show code
Element node
Element node
Element node
Constraint node: alignment
Constraint node: same-size
Constraint node: grouping


Show code
Paper catalogue
Graph Neural Networks for Ranking Web Pages — early web-graph ranking with GNNs. Venue: IEEE/WIC/ACM Web Intelligence 2005. Authors: Franco Scarselli, Sweah Liang Yong, Marco Gori, Markus Hagenbuchner, Ah Chung Tsoi, Marco Maggini.

Field	Details
Task	Customized page ranking on the hyperlink graph.
Dataset(s)	Unspecified in the accessible sources reviewed here.
Node types	Web pages; node attributes unspecified in the accessible sources.
Edge types	Directed hyperlink edges; edge attributes unspecified.
Relationship semantics	A directed edge exists because one page links to another.
Model architecture	Early graph neural network for ranking on graph-structured web data; detailed layer design is not visible in the accessible snippets.
Training objective / loss	Learn page-ranking functions from examples; exact loss unspecified in the accessible sources.
Key results	Reports promising preliminary results for adaptive/customized page-rank computation; exact metrics were not recoverable from the accessible snippets.
Code availability	No public code link surfaced in the accessed official sources.

Evidence:

DOM-Q-NET — RL for web navigation over the DOM. Venue: ICLR 2019. Authors: Sheng Jia, Jamie Kiros, Jimmy Ba. 

Field	Details
Task	Web navigation in MiniWoB with RL.
Dataset(s)	MiniWoB benchmark tasks.
Node types	DOM elements with attributes including tag, class, focus, tampered, and text information; also soft alignment between DOM text and goal tokens.
Edge types	DOM-tree structural relations used for message passing; edge attributes are not separately described in the accessible text.
Relationship semantics	Edges encode HTML tree structure so the agent can reason over local and contextual relations between webpage elements.
Model architecture	GNN backbone producing local, neighbor, and global representations; GRU-based message passing for the neighbor module; global module from max pooling or goal-attention; separate MLP Q-heads for DOM selection, word-token selection, and mode.
Training objective / loss	Off-policy Q-learning with Rainbow components; minimize squared TD error.
Key results	Reaches 100% success on most selected tasks, 86% on social-media without goal-attention, solves click-widget and social-media at 100% with goal-attention, and shows roughly 2× sample-efficiency gains in multi-task training.
Code availability	Official GitHub repository available.

Evidence:

ZeroShotCeres — zero-shot relation extraction from semi-structured webpages. Venue: ACL 2020. Authors: Colin Lockard, Prashant Shiralkar, Xin Luna Dong, Hannaneh Hajishirzi.

Field	Details
Task	Zero-shot OpenIE and ClosedIE on semi-structured webpages.
Dataset(s)	Extended SWDE: 21 English-language sites in Movie, NBA, and University verticals, 400–2,000 pages per site.
Node types	Text fields. Features include bounding-box coordinates, box height/width, font size, one-hot typeface, font weight, font style, color, alignment, plus textual features; for ClosedIE it uses BERT text representations, while OpenIE uses a text-field frequency signal for better out-of-vertical generalization.
Edge types	Horizontal, vertical, and DOM edges, plus self-loops in GAT.
Relationship semantics	Horizontal/vertical edges encode page layout adjacency; DOM edges encode XPath/DOM proximity for sibling or cousin text fields.
Model architecture	2-layer GAT over the page graph; downstream FFNN classifiers over contextualized node pairs or relation labels.
Training objective / loss	Cross-entropy for pre-training, binary OpenIE classification, and multi-class ClosedIE classification.
Key results	Abstract reports a 31% F1 gain over baseline on a new vertical. Detailed tables report OpenIE Level-I average F1 = 0.46 for ZSCERES-GNN versus 0.35 for the colon baseline, and ClosedIE F1 = 0.58 versus 0.46 for the no-context FFNN baseline.
Code availability	No official public code link was visible in the accessed sources.

Evidence:

Web Image Context Extraction with Graph Neural Networks and Sentence Embeddings on the DOM Tree — image-context extraction from HTML structure. Venue: ECML PKDD 2021 workshop / CCIS chapter. Authors: Chen Dang, Hicham Randrianarivo, Raphaël Fournier-S’Niehotta, Nicolas Audebert.

Field	Details
Task	Web image context extraction without browser rendering.
Dataset(s)	No gold WICE dataset; train/evaluate on a proxy task that predicts the text semantically closest to the image caption. Exact dataset composition is not fully specified in the accessible snippets.
Node types	DOM/text nodes with node type information plus text features; sentence embeddings are injected into the DOM graph.
Edge types	DOM-tree relations; exact directedness and extra edge attributes are unspecified in the accessible snippets.
Relationship semantics	DOM structure provides context for associating images with nearby or structurally relevant text.
Model architecture	GNN-based DOM model combined with NLP sentence embeddings; exact GNN variant and layer count were not fully recoverable from the accessible sources.
Training objective / loss	Proxy-task cosine-similarity regression loss between predicted and reference text embeddings.
Key results	The paper reports “promising results” for large-scale HTML-only WICE, but exact numeric best scores were not fully visible in the accessible snippets.
Code availability	No official public code link was visible; CatalyzeX indicates code was not directly linked.

Evidence:

GROWN+UP — general-purpose DOM parser with self-supervised pre-training. Venue: CIKM 2022. Authors: Benedict Yeoh, Huijuan Wang.

Field	Details
Task	General webpage representation, evaluated on boilerplate removal and genre classification.
Dataset(s)	Pre-training on 180K CommonCrawl 2008 webpages. Downstream: CleanEval, Dragnet, 7-Web, KI-04.
Node types	DOM tags with features: enclosed text + length, class, id, tag type, font weight, font style, font size, number of children, child index, Laplacian positional encoding.
Edge types	Directed node→parent, node→child, and node→self edges.
Relationship semantics	Parent-child edges preserve DOM structure; self-loops preserve source-node information during aggregation.
Model architecture	Deep residual GNN with GraphConvGated blocks, LSTM layers after graph convolution, and appended Transformer blocks for graph-level aggregation.
Training objective / loss	Self-supervised masked DOM-feature prediction plus same-website prediction, using cross-entropy, cosine loss, and binary cross-entropy; downstream tasks add supervised heads, including Li-Arcface for genre classification.
Key results	Boilerplate removal: 93.4 ± 0.2 Micro-F1 on CleanEval and 96.9 ± 0.3 on Dragnet. Genre classification: 96.8 ± 0.9 on 7-Web and 85.9 ± 4.0 on KI-04.
Code availability	Official GitHub repository released.

Evidence:

Layout-aware Webpage Quality Assessment — DOM-layout GNN for search quality. Venue: arXiv preprint; reported as deployed in Baidu Search. Authors: Anfeng Cheng and colleagues.

Field	Details
Task	Webpage quality assessment for search ranking.
Dataset(s)	Private search-engine corpus with 600,000 training webpages and 20,000 test webpages, plus online evaluation in a real ranking system.
Node types	Layout-graph nodes from DOM elements; example types include text, image, video. Features include height, width, x/y position, position type, number of words, font size, font style, line height, font weight, alignment, border, padding, margin, visibility, display style, outline style, outline width, tag name, and webpage category.
Edge types	DOM parent-child adjacency plus a global virtual node connected to all nodes.
Relationship semantics	DOM edges encode local content-layout interactions; the virtual node captures graph-level interactions between local content and overall page layout.
Model architecture	GAT backbone with stacked attention layers, improved with attentive readout via a virtual node and category-aware optimization.
Training objective / loss	Supervised webpage-quality prediction against human labels; exact loss is not explicitly stated in the accessible snippets reviewed here.
Key results	Best offline model (Virt-GAT) reports PNR 3.71 and AUC +24.08% relative improvement over the prior online method. Online evaluation reports +0.19% DCG on random queries, +0.42% DCG on tail queries, and +4.10% / +0.52% / +5.13% side-by-side gains for random / tail / same-quality settings.
Code availability	No official public code link was visible in the accessed sources.

Evidence:

PLM-GNN — joint PLM + DOM-graph encoder for webpage classification. Venue: arXiv preprint. Authors: Lang and co-authors; the full author list was not cleanly exposed in the accessible parsed text. 

Field	Details
Task	Webpage classification.
Dataset(s)	KI-04, SWDE, AHS 1.0, AHS 2.0.
Node types	DOM nodes represented by XPath embeddings; XPath unit embeddings combine tag-unit and subscript-unit embeddings.
Edge types	Relationships considered are parent-child and sibling, but the paper states it uses only directed parent→child and child→parent edges for GNN message passing.
Relationship semantics	DOM ancestry defines the graph used to encode webpage structure.
Model architecture	Hybrid model: PLM text encoder plus DOM tree GNN encoder and graph readout; the generic AGGREGATE/UPDATE formulation is given, but the exact GNN variant is left unspecified in the accessible text. Final classifier is a 2-layer MLP.
Training objective / loss	Standard cross-entropy for multi-class classification.
Key results	Reported performance: KI-04 Acc/F1 = 1.000/1.000, SWDE Acc/F1 = 0.902/0.897, AHS 1.0 Acc/F1 = 0.992/0.992, AHS 2.0 Acc/F1 = 1.000/0.999. The perfect KI-04 score should be interpreted cautiously because the paper itself notes the dataset is small.
Code availability	No official code link was visible in the accessed paper sources; a GitHub result found by search did not clearly correspond to this webpage paper and is therefore not relied on here.

Evidence:

Graph4GUI — heterogeneous graph representation for GUIs. Venue: CHI 2024. Authors: Yue Jiang, Changkong Zhou, Vikas Garg, Antti Oulasvirta.

Field	Details
Task	Primarily GUI autocompletion; also GUI topic classification and GUI retrieval.
Dataset(s)	Derived from ENRICO and VINS, refined to 5,653 GUIs, with five-fold cross-validation and large numbers of incomplete-GUI/target pairs.
Node types	Two node types: GUI element nodes and constraint nodes. Element-node properties include position, size, visual appearance, textual content, and element type. Constraint-node types include alignment, same-size, element grouping, multimodal grouping.
Edge types	Bipartite edges between element nodes and constraint nodes; GUI elements do not connect directly to each other in the main graph formulation.
Relationship semantics	Edges indicate that an element participates in a particular alignment, same-size, or grouping constraint.
Model architecture	Heterogeneous graph with SAGEConv as the GNN backbone; element features include ResNet152 visual embeddings and BERT text embeddings; output vectors are 256-dimensional.
Training objective / loss	For autocompletion, element prediction uses MSE over position/size plus boundary penalties, and constraint prediction uses binary cross-entropy.
Key results	Human-preference study on autocompletion: 70.33% preferred Graph4GUI, 13.54% preferred GRIDS, 16.13% no preference. GUI-topic classification reaches 91.53% accuracy, outperforming ResNet50, nearest neighbors, and random forest baselines.
Code availability	The official project page links the paper and BibTeX, but no public code repository was linked on that page.

Evidence:

AccessFixer — relational GNN for accessibility repair in GUIs. Venue: arXiv preprint; paper text indicates IEEE Transactions on Software Engineering formatting. Authors: Mengxi Zhang, Huaxiao Liu, Chunyang Chen, Guangyong Gao, Han Li, Jian Zhao.

Field	Details
Task	Accessibility repair for low-vision users: fix small size, narrow interval, and low color contrast issues in GUIs.
Dataset(s)	8,554 collected GUIs, filtered to 2,050 issue-free GUIs, then 1,925 validated GUIs used for R-GCN pre-training; additional evaluation on 30 real-world apps and 10 open-source apps.
Node types	Component nodes and container nodes. Component-node attributes include primary attributes bounds and color, and accessibility-related attributes size, color contrast, intervals. Container nodes have no attributes.
Edge types	Three types: component-component adjacency within a container, component-container membership, and container-container sequential layout. The graph is effectively treated symmetrically in the adjacency matrix description.
Relationship semantics	Edges express local spatial adjacency, containment, and top-to-bottom / left-to-right layout order.
Model architecture	R-GCN encoder with two convolutional layers, ReLU, and max-pooling, followed by a DistMult decoder for missing-edge prediction; the paper uses resulting node “spectral signals” to infer accessibility-preserving repairs.
Training objective / loss	Link-prediction-style edge loss based on a union loss formulation with DistMult scoring.
Key results	Solves an average of 81.2% of accessibility issues after repair; fixes 3.54% more size-related problematic components than the baseline; in 10 open-source apps, 8 pull requests were merged or under fixing.
Code availability	The paper states that datasets and code are open-sourced, but the accessed sources did not expose an official repository link directly.

Evidence:

Cross-paper synthesis
The dominant and most successful node design is the semantically enriched DOM/UI node. Papers rarely succeed with structure alone. Instead, they combine structural nodes with text, attributes, or layout descriptors: DOM-Q-NET uses tag/class/state/text features; ZeroShotCeres uses text-field geometry, style, and text; GROWN+UP uses text, class/id, typography, and position-like features; the Baidu model uses rich layout and content descriptors; Graph4GUI uses visual, textual, type, size, and position features. This pattern strongly suggests that GNNs help most when the graph edges supply relational inductive bias and the node features carry the content signal.

A second strong pattern is that page- or UI-level success depends on edge semantics being task-aligned. Parent-child DOM edges are sufficient for graph-level webpage classification, boilerplate removal, and quality scoring, because these tasks depend on hierarchical organization and aggregate layout/style. By contrast, relation extraction and UI layout tasks benefit from spatial or constraint edges that cannot be read off from the DOM alone. ZeroShotCeres shows performance drops when DOM or spatial edges are removed. Graph4GUI was designed explicitly because view hierarchies or image models do not naturally encode constraints like alignment and same-size. AccessFixer likewise requires relations that correspond to containment and adjacency rather than just raw hierarchy. 

A third pattern is the importance of global aggregation. GROWN+UP uses Transformer blocks and same-website pre-training for graph-level tasks; the Baidu paper adds a virtual node; DOM-Q-NET uses a global module and goal-conditioned attention; Graph4GUI combines graph embeddings with element-level and constraint predictions. These choices correlate with better graph-level classification or decision-making because webpage-level tasks need both local detail and holistic context. 

Failure modes are also recurrent. First, insufficient relation design hurts performance: DOM-Q-NET’s ablation shows that omitting the neighbor module makes DOM selection brittle when many DOM elements share similar attributes; ZeroShotCeres’ ablations show removal of DOM or spatial edges reduces F1; Graph4GUI participants noted weaker performance when the next element had no useful alignment/grouping cues. Second, deep GNN optimization and over-smoothing remain concerns: GROWN+UP explicitly adds residuals and LSTMs to mitigate this, while AccessFixer reports best performance at two R-GCN layers and degradation with more layers. 

Evaluation quality is uneven. GROWN+UP is one of the stronger papers methodologically because it benchmarks on multiple tasks and standardizes comparisons on boilerplate removal. ZeroShotCeres is strong because it evaluates true cross-site and cross-vertical generalization. The Baidu paper is operationally compelling because it includes online metrics, but its core dataset is private. PLM-GNN reports strong numbers, but one benchmark is tiny and an exact GNN variant is not specified in the accessible text. GUI papers often include perceptual or user-study evidence, which is valuable, but they are not directly comparable to webpage IR tasks. 

Comparison of node and edge patterns
Pattern family	Representative papers	Typical tasks	What the nodes look like	What the edges mean	Typical outcome
Hyperlink graph	Ranking Web Pages 	Ranking	Web pages as nodes	Directed hyperlinks	Historically important, but less detailed local semantics than later DOM models
Pure DOM hierarchy	DOM-Q-NET, GROWN+UP, Layout-aware QA, PLM-GNN	Navigation, classification, quality, boilerplate removal	HTML elements with text/attributes/layout features	Parent-child, reverse, self, or virtual-node links	Strong for page-level tasks when node features are rich
DOM + spatial adjacency	ZeroShotCeres, WICE 	Relation extraction, image-context extraction	Text fields or DOM/text nodes	Horizontal, vertical, DOM proximity	Better for layout-sensitive extraction and cross-template generalization
Heterogeneous element-constraint graph	Graph4GUI 	Autocompletion, classification, retrieval	Elements + alignment/size/grouping constraint nodes	Membership in a constraint	Strong for layout generation and designer preference
Relational GUI repair graph	AccessFixer 	Accessibility repair	Component nodes + container nodes	Adjacency, containment, sequential layout	Good for coordinated, dependency-aware fixes

The most reliable cross-paper conclusion is that node richness without appropriate edges underuses the graph, while good edges without meaningful node features leave the model blind to semantics. The best-performing papers nearly always combine both. 

Open challenges and limitations
The biggest open challenge is benchmark standardization. There is still no widely accepted benchmark suite that jointly covers webpage DOM understanding, layout-sensitive extraction, navigation, and page-quality prediction. The result is that papers often look impressive in isolation but remain hard to compare rigorously. 

A second challenge is cross-site generalization under modern web complexity. ZeroShotCeres is notable because it explicitly targets unseen sites and unseen verticals, but many other papers are trained on relatively old or narrow datasets. Today’s web pages are more dynamic, script-heavy, and visually mediated than those benchmarks capture. 

A third challenge is multimodality. Many papers still rely primarily on text plus DOM/layout attributes. GROWN+UP explicitly argues that adding rendered visual features, images, audio, or video should help, and Graph4GUI’s success suggests that explicit visual/spatial information matters when layout quality is central. A modern frontier is likely DOM + rendered view + interaction traces in a single heterogeneous graph. 

Some details remain incomplete in this report because the accessible sources did not expose them cleanly. In particular, exact layer counts or message functions are not always specified in the accessible snippets for the 2005 ranking paper and WICE, and PLM-GNN’s accessible text does not clearly identify a named GNN variant. Where that happened, I marked the detail as unspecified rather than inferring it.
