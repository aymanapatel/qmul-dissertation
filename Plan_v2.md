### Proposed title

**AccessibilityGraph-RAG: A Hybrid Structural, Visual and Retrieval-Augmented System for Detecting and Repairing Web Accessibility Issues**

### Project aim

To develop and evaluate a hybrid web-accessibility system that combines:

- deterministic accessibility testing;
- DOM and accessibility-tree graph representations;
- Graph Neural Networks for relational accessibility issues;
- rendered-pixel and visual analysis;
- browser-based interaction testing;
- retrieval-augmented generation for contextual remediation;
- automatic repair validation.

The system should not attempt to use one model for every WCAG issue. Instead, it should route each candidate issue to the most appropriate specialist analyser.

### Main research question

> Does evidence-aware routing between deterministic, graph-based, visual, interaction-based and semantic analysers improve the detection and repair of web-accessibility issues compared with conventional automated accessibility tools and standard vector RAG?

### Supporting research questions

1. Does graph information improve detection of relational accessibility issues?
2. Does rendered visual evidence improve detection of contrast, focus visibility and occlusion issues?
3. Does graph-aware retrieval produce more contextually appropriate repairs than flat vector retrieval?
4. Does automatic re-validation reduce invalid or regressive LLM-generated repairs?
5. Which graph edges, node features and specialist analysers contribute most to overall performance?

---

## 2. Core Design Principle

The system should be **hybrid**.

Different accessibility problems require different forms of evidence:

| Issue characteristic | Preferred detector |
|---|---|
| Exact HTML, ARIA or numerical rule | Deterministic rule engine |
| Relationship between elements | Graph rules or GNN |
| Rendered pixels, colours or visual geometry | Visual/pixel analyser |
| Keyboard, hover, focus or dynamic behaviour | Browser interaction analyser |
| Meaning, quality or contextual appropriateness | NLP/LLM or human review |
| Multiple characteristics | Run multiple specialist detectors and fuse evidence |

The GNN should not replace axe-core, browser testing or deterministic contrast calculations. It should be used where relational structure provides useful information.

---

## 3. High-Level Architecture

```text
Client, CLI or dashboard
          |
          v
POST /v1/analyse
          |
          v
+---------------------------------------+
| Accessibility Orchestrator            |
| - creates scan job                    |
| - launches browser                    |
| - controls specialist routing         |
| - merges evidence                     |
+---------------------------------------+
          |
          v
+---------------------------------------+
| Playwright Evidence Collector         |
| - DOM snapshot                        |
| - accessibility tree                  |
| - computed styles                     |
| - screenshots and element crops       |
| - bounding boxes and paint order      |
| - keyboard focus sequence             |
| - hover, focus and modal states        |
+---------------------------------------+
          |
          v
+---------------------------------------+
| Candidate Generation Layer            |
| - axe-core findings                   |
| - custom deterministic rules          |
| - suspicious graph patterns           |
| - visual candidates                   |
| - interaction-state candidates        |
+---------------------------------------+
          |
          v
+----------------------------------------------------+
| Evidence-Aware Router                              |
|                                                    |
| deterministic candidate -> Rule Engine             |
| relational candidate    -> Graph/GNN Analyser      |
| visual candidate        -> Visual/Pixel Analyser   |
| interaction candidate   -> State/Focus Analyser    |
| semantic candidate      -> LLM/Human Review        |
+----------------------------------------------------+
          |
          v
+---------------------------------------+
| Finding Fusion and Confidence Layer   |
| - normalises detector output          |
| - combines supporting evidence        |
| - assigns confidence                  |
| - flags manual review                 |
+---------------------------------------+
          |
          v
+---------------------------------------+
| Graph-RAG Repair Engine               |
| - retrieves WCAG/ACT guidance         |
| - retrieves ARIA patterns             |
| - retrieves similar graph cases       |
| - generates contextual repair         |
+---------------------------------------+
          |
          v
+---------------------------------------+
| Repair Validation Sandbox             |
| - applies temporary patch             |
| - reruns detector                     |
| - checks accessibility tree           |
| - checks keyboard behaviour           |
| - checks visual regression            |
| - accepts or rejects repair            |
+---------------------------------------+
          |
          v
Structured report, evidence, repair and validation result