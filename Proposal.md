# Accessibility

## Proposed Aim

To develop a graph-augmented retrieval system that leverages Graphical Neural Networks. The system parses the accessibility tree of web interfaces to detect WCAG violations and delivers targeted remediation recommendations by reasoning over a knowledge graph encoding accessibility criteria, ARIA semantics, and structural repair patterns.

## Rationale

### The Accessibility Problem is Hard

Despite decades of WCAG standards, over 96% of homepages still have detectable accessibility errors (WebAIM 2024). The challenge is not a lack of rules—it's that accessibility is inherently *relational*: a button's accessibility depends on its label, its context, its parent container, and its interaction with focus management. Existing tools like axe-core or Lighthouse detect surface-level violations but struggle to explain *why* something fails or *how* to fix it in context.

### Why use Graphical Neural Networks?

- **Modelling similarity:** Graphical Neural Networks model the hierarchal relationship similar to the accessibility tree.
- **Embedding structure connection to RAG:** As RAG works in embedding vector space, it makes it easier to build an end-to-end LLM+RAG pipeline.

### Why take this project?

Based on my experience with developing accessible sites and a hackathon project that included building a RAG system, I am able to combine these two topics to achieve my goal of building truly accessible sites.