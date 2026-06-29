arXiv:2511.03471v1 [cs.AI] 5 Nov 2025

# Towards Scalable Web Accessibility Audit with MLLMs as Copilots

Ming Gu¹,², Ziwei Wang¹,², Sicen Lai¹,³, Zirui Gao¹,², Sheng Zhou¹,³*, Jiajun Bu¹,²

¹Zhejiang Key Laboratory of Accessible Perception and Intelligent Systems, Zhejiang University

²College of Computer Science and Technology, Zhejiang University

³School of Software Technology, Zhejiang University

{gmwork, wangziwei98, laisicen, gaozirui.zju, zhousheng.zju, bjj}@zju.edu.cn

## Abstract

Ensuring web accessibility is crucial for advancing social welfare, justice, and equality in digital spaces, yet the vast majority of website user interfaces remain non-compliant, due in part to the resource-intensive and unscalable nature of current auditing practices. While WCAG-EM offers a structured methodology for site-wise conformance evaluation, it involves great human efforts and lacks practical support for execution at scale. In this work, we present an auditing framework, AAA, which operationalizes WCAG-EM through a human-AI partnership model. AAA is anchored by two key innovations: GRASP, a graph-based multimodal sampling method that ensures representative page coverage via learned embeddings of visual, textual, and relational cues; and MaC, a multimodal large language model-based copilot that supports auditors through cross-modal reasoning and intelligent assistance in high-effort tasks. Together, these components enable scalable, end-to-end web accessibility auditing, empowering human auditors with AI-enhanced assistance for real-world impact. We further contribute four novel datasets designed for benchmarking core stages of the audit pipeline. Extensive experiments demonstrate the effectiveness of our methods, providing insights that small-scale language models can serve as capable experts when fine-tuned.

**Code & Datasets** — https://github.com/eaglelab-zju/AAA

**Standard version** —

https://openreview.net/forum?id=kz2hcsWcGu

## 1 Introduction

*Web accessibility* is a foundational principle in the pursuit of an inclusive digital environment, ensuring that all users including those with disabilities can perceive, navigate, and interact with online content (Web Accessibility Initiative (WAI) 2024; Sharif et al. 2022). Despite the widespread adoption of standards such as the Web Content Accessibility Guidelines (WCAG) (Consortium 2024b), and substantial efforts devoted to web accessibility evaluation, the state of web accessibility remains alarmingly poor. A recent study reported that 94.8% of homepages across one million websites contained accessibility violations (WebAIM

2025). Emerging research suggests that this stagnation stems not from a lack of education or tooling, but from the intrinsic complexity of web accessibility as a resource management problem (Abramovich and Patitsas 2024; Elgaly et al. 2024). This means that *the time-consuming and labor-intensive nature of Web Accessibility Audits (WAA) increasingly misaligned with the growing scale and maintenance cost of modern websites* (SolarWinds Worldwide 2025).

To address WAA, the World Wide Web Consortium (W3C) introduced the Website Accessibility Conformance Evaluation Methodology (WCAG-EM) (Group 2014), a five-step protocol designed to standardize evaluation procedures. However, it lacks a corresponding technical framework that supports scalable execution in practice. In this context, **scalability** refers to two critical capabilities: (1) accelerating audit processes via automation, and (2) minimizing unavoidable manual effort through intelligent human-AI collaboration. Yet, most existing tools operate only at the page or element level, *covering only fragments of the WCAG-EM pipeline* (Huang et al. 2024). This narrow scope hinders scalability with bottlenecks in both time and labor.

To overcome the limitations, we propose a comprehensive framework anchored in three pillars: Automation, AI, and Auditor (AAA). *AAA operationalizes five procedures aligned with WCAG-EM's five steps*, including web crawling, automated checks, page sampling, manual evaluation, and reporting/remediation, with the goal of enabling scalability across the full audit lifecycle. Despite advances in automating tasks such as crawling and hard-coded checks, *two fundamental challenges remain*. **First**, existing page sampling methods fail to satisfy WCAG-EM's representativeness requirements. Recent clustering-based approaches rely primarily on textual similarity (Hambley et al. 2023), overlooking the rich multimodal semantics of web pages including visual layout, textual content and hyperlink relationships, which are essential for capturing diversity and representativeness. **Second**, intelligent assistance for manual auditing tasks remains underexplored. Given that no single tool can fully determine whether a website meets accessibility standards (W3C Web Accessibility Initiative (WAI) 2023), WAA inevitably requires human evaluation. However, current methods offer minimal assistance in high-effort tasks such as identifying accessibility-critical components, which often demand sophisticated multimodal reasoning, making

*Corresponding Author

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

---

them particularly burdensome for human auditors to collect.

To tackle these challenges, we first introduce Graph-based Representative Page Clustering for Sampling (GRASP), a novel multimodal approach that generates WCAG-EM-compliant representative page subsets. GRASP defines representativeness across three complementary dimensions: textual semantic, visual layout, and linkage relationships, and employs graph neural networks (GNNs) to learn a unified embedding space for representative clustering. A dedicated structure learning module further improves sampling quality by mutually enhancing representativeness and clustering. In parallel, we explore the emerging potential of multimodal large language models (MLLMs) in accessibility workflows (Ara and Sik-Lanyi 2025). We present MLLMs as Copilot Assistant, Auditor and Consultant (MaC), a holistic AI companion designed to support multiple stages of WAA. By enabling cross-modal reasoning, MaC assists in identifying audit-critical elements and pages, thereby accelerating both sampling and manual evaluation. Furthermore, it broadens audit coverage by facilitating the evaluation of underrepresented accessibility issues, particularly those affecting less commonly addressed disabilities like cognitive impairment. To support future research, we also release four new datasets tailored for distinct stages of the WAA pipeline. These datasets address the current lack of accessibility-specific benchmarks. Our contributions are summarized as:

- Scalable WAA Framework: We propose a full-lifecycle audit framework AAA aligned with WCAG-EM, advancing scalability across web accessibility audit lifecycle.
- Multimodal Sampling Method: We introduce GRASP, a novel graph-based multimodal page sampling technique satisfying WCAG-EM representativeness criteria.
- MLLM as Copilot Strategy: We introduce MAC, a versatile MLLM-powered strategy augmenting multiple labor-intensive procedures via multimodal reasoning.
- Benchmark Datasets: We release four new datasets tailored to different stages of the AAA pipeline, facilitating comprehensive evaluation and comparison.
- Empirical Insights: Through extensive experiments, we demonstrate the effectiveness of our methods and uncover the potential of small MLLMs as domain experts.

## 2 Related Work

Web Accessibility Audit (WAA). With growing demand for an inclusive web, web accessibility auditing has become essential. Traditional tools like WAVE (WebAIM, Utah State University 2025) and Axe (Deque Systems, Inc 2025) rely on hard-coded checks that detect syntactic issues (e.g., missing alt text, low contrast), but often missing contextual and semantic aspects (López-Gil and Pereira 2024; Ara and Sik-Lanyi 2025). Recent advances leverage large language models (LLMs) to provide intelligent evaluation and repair suggestions, aiming to reduce manual effort (Huang et al. 2024; Othman, Dhouib, and Nasser Al Jabor 2023). Nevertheless, accessibility gaps persist: a 2025 audit of one million websites revealed WCAG violations on 94.8% of home pages (WebAIM 2025). Studies suggest this

stems less from tool limitations and more from resource constraints (Abramovich and Patitsas 2024). Auditing remains time- and labor-intensive, especially as website scale increases nowadays (SolarWinds Worldwide 2025). Existing approaches predominantly focus on individual elements or pages, lacking a framework to address labor and resource challenges across the full site-wise audit lifecycle.

Page Sampling for WAA. Full-site evaluation is infeasible for large sites, making page sampling essential for producing representative results. WCAG-EM outlines two dimensions: (1) the individual level, which targets pages with critical accessibility relevance (e.g., essential functionality, accessibility statements, or home-linked common pages), and (2) the collective level, which ensures diversity and representativeness across the site. However, existing methods often address narrow aspects, such as URL patterns (Zhang et al. 2015b), Web Accessibility Quantitative Metric (Zhang et al. 2015a), or structure-based active learning (Yu et al. 2020), falling short of multi-level requirements. A recent method, Web Structure Derived Clustering (SDC) (Hambley et al. 2023), attempts to align with collective-level sampling by clustering. Yet, it excludes individual-level sampling, potentially omitting accessibility-critical pages, and relies solely on shallow statistical textual features, lacking semantic depth and multimodal integration.

LLM Applications in WAA. Several studies have explored the use of LLMs for web accessibility evaluation (Huang et al. 2024; Othman, Dhouib, and Nasser Al Jabor 2023; He, Huq, and Malek 2025), demonstrating impressive automation in addressing element- and page-level issues. However, existing approaches mainly focus on text semantic alignment (Zhong et al. 2025), such as text or code generation, and rarely involve more modularities and complex reasoning about accessibility knowledge which multimodal LLMs (MLLMs) are capable of. Moreover, most research on LLMs is confined to evaluation and remediation tasks, overlooking the broader applicability across the entire WAA lifecycle addressing numerous labor-intensive steps.

## 3 AAA: Scalable WAA Framework

### 3.1 Pipeline of the Proposed Framework AAA

We propose a scalable WAA framework for large or multiple sites centered on Automation, AI, and Auditor (AAA). Here, AI denotes artificial intelligence technologies such as computer vision and natural language processing, which can understand abstract concepts beyond rule-based programmatic automation. Moreover, since no single tool can independently determine whether a website meets accessibility standards (L. Holliday 2020), and given the reliability and application challenges associated with LLMs, such as hallucinations (Kaddour et al. 2023), knowledgeable evaluation by human auditors remains essential. Inspired by the five-step guidance of WCAG-EM (Group 2014), AAA is designed for scalable auditing of large websites or multi-sites, where exhaustive assessment of all content is impractical. We reinterpret WCAG-EM from a technical perspective and organize it into five structured procedures. An overview of AAA along with a comparison to WCAG-EM is in Figure 1.

---

![img-0.jpeg](images_md/img-0.jpeg)

Figure 1: Overview of AAA.

Website Crawling. For large-scale or multi-site evaluations, manually defining the evaluation scope, as required by WCAG-EM, is impractical. To address this, automated website crawling is introduced to systematically explore and extract site structures and content at scale.

Auto Check. Automated checks in AAA are performed using two types of checkers: ① Hard-coded checkers, which include tools based on static DOM parsing (e.g., Axe (Deque Systems, Inc 2025)) and dynamic UI testing frameworks (e.g., Selenium (Project 2025)). These tools provide deterministic accuracy but are limited in assessing semantic-level issues. ② AI-powered checkers, which leverage intelligent technologies to perform visual and textual semantic analysis, enabling the detection of accessibility violations that require deeper contextual understanding. While accuracy trade-offs are introduced due to the black-box nature, significant potential are offered for evaluating complex issues.

Page Sampling. WCAG-EM prescribes three steps to construct a representative sample: (i) Include a structured sample, (ii) Include a randomly selected sample, (iii) Include complete processes. Except for random sampling, these steps require sophisticated semantic understanding of web pages at multiple levels. To alleviate the labor burden, we introduce: ① A novel deep-learning method integrating visual, textual, and relational information to optimize the random selection of representative diverse pages. ② The use of MLLMs to recognize structured samples and complete processes, leveraging their strong multi-modal semantic understanding capabilities (Li et al. 2024).

Manual Check. While manual evaluation follows WCAG 2.2 (Consortium 2024b,a) success criteria, its scalability and coverage remains a challenge. Two key optimizations are introduced by MLLM-powered assistance: ① Pre-extraction of accessibility-critical items: Some critical ac-

cessibility issues occur infrequently across a website, potentially reducing audit coverage in a sampling-based pipeline. MLLMs assist in identifying these elements in advance to ensure a faster and comprehensive evaluation. ② Automation of evaluating underrepresented accessibility issues: Leveraging the powerful multi-modal reasoning capabilities of MLLMs (Kil et al. 2024), we can automate the evaluation of accessibility issues that typically receive less attention.

Report/Remediation. The remediation process is integrated into the reporting step for rich and actionable feedback with automated remediation potentials. First, reports should contain detailed violation descriptions and developer-friendly repair suggestions. Second, recent advances in AI-driven code generation provide promising avenues for automated accessibility fixes.

### 3.2 Challenges in Implementing AAA

Representative Page Sampling for Scalable Audits. First, existing approaches to page sampling in WAA are predominantly based on statistical analysis of DOM text (Hambley et al. 2023), which lack a deeper understanding of multimodal semantics like visual styles and layouts, textual topics, and functional diversity. These dimensions are essential for identifying representative pages, as emphasized in WCAG-EM Step 2.c "Identify the variety of web page types". Second, with the advancement of MLLMs in comprehending complex semantics across modalities, it is now feasible to automatically identify many key web page types previously reliant on manual inspection, like common web pages and essential functionality in WCAG-EM Step 2.a "Identify common web pages" and Step 2.b "Identify essential functionality of the website". MLLMs enables the construction of page samples that are not only statistically representative but also semantically aligned with accessibility

---

|  Sub-step | Type | Description | Methods  |
| --- | --- | --- | --- |
|  3.a | Common Web Pages and States | Linked Directly from the Main Entry Point (Home Page) | MaC  |
|   |  Relevant Web Page and States | Relevant for People with Disabilities and the Accessibility of the Website  |   |
|   |  Additional Web Pages and States | Essential Functionality  |   |
|   |   |  Web Technologies Relied Upon  |   |
|   |   |  Variety of Web Page Types | GRASP  |
|  3.b | Randomly Selected Sample | Number: 10% of the 3.a sample | —  |
|  3.c | Complete Processes | Pages belonging to a series presenting a complete process | MaC  |

Table 1: Three Sub-steps (3.a, 3.b, 3.c) of WCAG-EM Step 3 Achieved by Our Proposed Methods MaC and GRASP.

requirements, largely saving labor and resource.

Comprehensive MLLMs Integration in WAA. First, most existing works on LLMs-powered WAA only explore checks that have been well covered by existing automated tools and limited within textual information (Zhong et al. 2025; He, Huq, and Malek 2025), leaving out evaluation that relies on complex modalities or semantics, which is the unique advantage of MLLMs. Second, the existing work on the application of MLLMs in WAA is mostly limited to evaluation or redediation (Suh et al. 2025), and has not fully explored its potential in the entire lifecycle.

Datasets for WAA Benchmarking. Standardized evaluation datasets are still lacking, which include not only cases of accessibility issues that are not detectable by automated tools, but also an evaluation of the application of MLLMs in other steps of the WAA lifecycle.

## 4 GRASP: Graph-based Page Sampling

WCAG-EM Step 3 (S3, "Select a Representative Sample") has three sub-steps as shown in Table 1, in which the randomly selected sample in 3.b can be trivially realized as a random sample with certain tools (Consortium 2024b). Therefore, we focus on the other two sub-steps, which are achieved by a two-fold method. From the individual perspective, when the selection of each page is based on the individual characteristic of itself, MLLMs are used as an alternative to the labor-intensive human recognition of (1) common and relevant web pages and states, (2) two kinds of additional web pages and states and (3) all pages of complete processes. From the collective perspective, as tasks not solvable by scale is a well-known challenge faced by MLLMs (Kaddour et al. 2023), which calls for other UI understanding models to deal with this task where a whole picture of different pages across the web-

site is needed to be captured for sampling representative pages among them. Individual sampling will be introduced in next section, and for collectivitive sampling, we propose Graph-based Representative pAge clustering for sampling (GRASP). Figure 2 presents the overview of GRASP.

### 4.1 Triple Representativeness for Page Variety

S3 defines the variety of web page types as varying styles, layouts, structures, and functionality with varying support for accessibility (Group 2014). These variety representativeness calls for understanding of multimodal semantics, like visual and textual, which are not fully covered in existing researches of statistical analysis. GRASP addresses this issue by taking into account three types of page representativeness from the perspectives of text, layouts and linkages.

Textual Semantic Representativeness. Traditional approaches based on token frequency or lexical analysis fall short, as they tend to reflect only surface-level textual features, neglecting deeper semantic structures and functional intentions. To address this, we leverage BERT (Devlin 2018), a contextualized language model that captures the nuanced meaning of words and phrases within their broader linguistic and structural context. For instance, it distinguishes between the use of “submit” in a login form versus in a feedback module by attending to nearby content and structural patterns. This capability makes BERT particularly well-suited for extracting semantically representativeness.

Layout Visual Representativeness. With the advancement of web technologies, the textual structure of the DOM has become increasingly inadequate for reflecting the rendered visual layout of modern web pages—especially in dynamically generated single-page applications (SPAs). To overcome this challenge, we employ Vision Transformers (ViT) (Dosovitskiy 2020) to learn visual representations directly from page screenshots. This approach enables robust extraction of layout-level representativeness, capturing the spatial and visual organization of web content beyond what is available through static DOM analysis.

Linkage Relational Representativeness. Moreover, websites inherently contain rich hyperlink structures that are often overlooked in existing approaches. These linkages naturally form a graph structure that encodes functional relatedness and semantic proximity across multiple pages. For instance, pages belonging to the same functional module frequently exhibit clustering behavior within the hyperlink network, while pages with similar layouts tend to share common linking patterns or structural relationships. To model this, we adopt Graph Neural Networks (GNNs), which are well-suited for learning structured data (Hamilton, Ying, and Leskovec 2017). GNNs support the integration of node attributes with topological context, making them ideal for capturing and fusing this third modality of representativeness.

### 4.2 Representativeness-enhanced Page Sampling

GNN-based Graph Representativeness Clustering. Recent clustering-based page sampling approaches (Hambley et al. 2023) leverages shallow statistical representations derived from textual content, which fail to capture the deeper semantic nuances of text, and more critically, neglect both

---

![img-1.jpeg](images_md/img-1.jpeg)

Figure 2: Overview of GRASP.

the visual structure and inter-page relational context. To address this, we introduce a GNN-based graph clustering approach that explicitly integrates multiple modalities into the clustering process. Our method consists of two key stages. (1) Modality-specific Representation Learning, (2) Semantic Fusion via GNN Message Passing. The first stage is:

\[
\mathbf {H} _ {t} = \text { BERT } (\text { text } _ {\text { DOM }}), \mathbf {H} _ {v} = \text { ViT } (\text { image } _ {\text { screen }}), \tag {1}
\]

\[
\mathbf {X} = \mathbf {H} _ {t} | | \mathbf {H} _ {v}, \mathbf {H} _ {g} = \operatorname{GNN} (\mathbf {X}, \mathbf {A}), \mathbf {C} = \mathcal {C} (\mathbf {H} _ {g}), \tag {2}
\]

where || is concatenation, and A is the adjacent matrix of the hyperlinks. C is a clustering method like k-means, and C is the clustering assignments of web pages. This allows us to learn fused embeddings that encapsulate the combined representativeness across all modalities.

Representativeness-enhanced Graph Learning. While hyperlink structures offer a natural foundation for graph-based modeling of websites, they often suffer from noise and sparsity. First, pages with dissimilar semantics may still be interconnected due to structural conventions, such as universally linked footer pages, which do not necessarily indicate semantic relatedness. Second, pages that are semantically similar may not be directly connected, especially in hierarchical site architectures where linkage primarily reflects parent-child relationships. Sibling pages with shared intent may remain unlinked by multi-hops in the graph.

We propose a representativeness-enhanced graph learning approach that refines the raw hyperlink structure using representativeness affinity and disparity inspired by graph structure learning (Gu et al. 2023; Liu et al. 2022). Specifically, we leverage the representativeness clustering results derived from textual and visual representation to guide the reconstruction of the linkage graph. This refinement involves both the removal of noisy links and the recovery of semantically meaningful missing ones. Formally, we define

the hyperlink edge removal and recovery sets as:

\[
\mathcal {E} _ {\mathrm{rm}} = \mathcal {S} _ {\mathrm{sim}} (\mathbf {C}, \mathbf {H} _ {g}, \gamma), \mathcal {E} _ {\mathrm{rc}} = \mathcal {S} _ {\mathrm{dis}} (\mathbf {C}, \mathbf {H} _ {g}, \beta), \tag {3}
\]

\[
\mathcal {E} _ {\text { new }} = (\mathcal {E} _ {\mathbf {A}} \cap \mathcal {E} _ {\mathrm{rc}}) \setminus \mathcal {E} _ {\mathrm{rm}}, \tag {4}
\]

where  \( \gamma \)  and  \( \beta \)  control the thresholds for the least and most semantically similar node pairs, respectively, as determined by similarity functions  \( S_{sim} \)  and  \( S_{dis} \)  operating over cluster assignments C and node embeddings  \( H_{g} \) . The set  \( E_{rm} \)  identifies representativeness-disparity guided redundant or misleading links to be pruned from the original edge set  \( E_{A} \) , while  \( E_{rc} \)  identifies representativeness-affinity guided strong candidate connections. The refined set  \( E_{new} \)  is obtained by adding promising edges and removing low-quality ones.

Representative Centroids-based Sample Selection. We perform representative page sampling by selecting exemplar nodes from each cluster. For each cluster  \( c_{i} \)  of C, we first compute its centroid  \( \mu_{i} \)  and then select the representative node  \( v_{i} \)  closest to the centroid. The final sampled page set is obtained by aggregating all selected nodes across clusters:

\[
v _ {i} ^ {*} = \arg \min _ {v \in c _ {i}} \| \mathbf {H} _ {g} (v) - \mu_ {i} \| _ {2}, \mathcal {P} _ {\text { sample }} = \bigcup_ {i} v _ {i} ^ {*}. \tag {5}
\]

This strategy ensures that the sampled pages are deemed the most representative of the semantic, visual, and relational characteristics captured by their cluster.

## 5 Proposed MLLMs Strategies and Datasets
5.1 MaC: MLLMs as Various Copilots

Current applications of LLMs in accessibility face two limitations. (1) Limited Exploration of Evaluation. Most existing applications focus on a narrow range of rules, often overlapping with those addressed by traditional tools (e.g., missing alt text) (He, Huq, and Malek 2025), while complex tasks involving cognitive accessibility remain largely

---

|  Dataset | Task | Type | Input | Output | Method  |
| --- | --- | --- | --- | --- | --- |
|  TPS | Page Sampling | Multimodal sampling | Crawled pages' DOMs, screenshots, auto-check results, linkage graph, and sample size N | N representative pages | GRASP  |
|  APR | Accessibility-Relevant Page Recognition | Classification | Page screenshots | Prediction of 5 WCAG-EM-defined categories (common, relevant, essential, technology-dependent, or none) | MaC  |
|  CCT | Cognitive CAPTCHA Tests | Recognition, reasoning and classification | CAPTCHA screenshots | (1)Binary WCAG compliance, (2)violation reasons, (3)type (17 classes) | MaC  |
|  CPE | Complete Process Extraction | Multi-label classification | Page screenshots | Presence of 5 WCAG-EM-defined processes (search, filter, form, CAPTCHA, contact) | MaC  |

Table 2: Task definitions and their relationships with datasets and methods.

unaddressed. We argue that the true potential of MLLMs lies in enabling fairer and more comprehensive audits by filling evaluation gaps for underrepresented disabilities through expert-informed multimodal reasoning. (2) Restricted Scope of Applications. The application of MLLMs in web accessibility remains underexplored beyond evaluation and remediation, particularly in resource-intensive stages like page sampling and manual auditing. We argue that integrating MLLMs across the full audit pipeline can alleviate these bottlenecks by supporting human-in-the-loop workflows, enhancing scalability and enabling more holistic accessibility.

To address them, an integrated strategy is proposed to explore the competence of MLLMs as Copilots (MaC).

Assistant: Automating Labor-Intensive Tasks through Multimodal Reasoning. We explore two key use cases: ①Individuality-based Page Sampling: The WCAG-EM relies on manually identifying structured sample pages based on individual factors like functional role and structural position. By combining web crawling with MLLMs' multimodal reasoning, we automate this process, enabling informed sampling that captures accessibility-critical Individuality beyond what is achievable through collective data-driven methods like GRASP. ②Pre-audit Element Localization: MLLMs can be used to preprocess pages, automatically identifying and labeling candidate elements for manual review. This transforms manual auditing into a more efficient process of test without search, where human experts validate elements without needing to navigate the page exhaustively.

Auditor: Identifying Underrepresented Accessibility Barriers. Recent studies highlight an overemphasis on visual and auditory disabilities in accessibility guidelines and tools, often overlooking cognitive and situational disabilities that resist rule-based detection (Abramovich and Patitsas 2024). MLLMs, with their contextual and inferential capabilities, offer a promising alternative. We focus on WCAG 2.2 success criteria 3.3.8 and 3.3.9, which address accessible authentication and require reasoning about cognitive demands

in user verification. MLLMs can help identify such mechanisms and assess potential barriers by interpreting page semantics at a higher level of abstraction than existing tools.

Consultant: Providing Informed Remediation Suggestions. The consultant role envisions MLLMs as intelligent agents for recommending fixes to accessibility issues, a direction supported by prior work on generating image descriptions, improving HTML semantics, and correcting ARIA attributes. While promising, we focus on the assistant and auditor roles to address foundational scalability challenges, identifying the consultant role as a key avenue for future research in large-scale remediation.

In addition, we contend that a major barrier to evaluating the effectiveness of MLLMs as Copilots is the lack of comprehensive, task-specific datasets tailored to accessibility.

### 5.2 AWA: Datasets of AI for Web Accessibility

To overcome this, we propose four novel datasets designed to advance the application of AI for Web Accessibility (AWA) in various tasks as shown in Table 2.

Triple-representativeness Page Sampling (TPS). We have developed a dataset consisting of 495 publicly accessible websites, categorized into 117 distinct classes (please refer to Appendix). These websites were crawled using an automated web crawler, ensuring that no private or sensitive data was involved. On average, each website contains 196 webpages (ranging from a minimum of 104 to a maximum of 200), totaling 97,246 pages. The dataset includes the following data for each page: (1) the page's DOM, (2) a screenshot of the page, (3) auto-check results covering 131 rules, where each rule corresponds to the number of violations detected (using Axe-core (Deque Systems, Inc 2025)), and (4) an adjacency matrix representing the website's overall linkage graph. The inclusion of this novel adjacency matrix allows for more granular web accessibility sampling.

Accessibility-relevant Page Recognition (APR). To evaluate the ability of MLLMs Assistants in individuality-

---

based page sampling, we constructed a manually annotated dataset consisting of 968 pages from five websites of different classes (entertainment, job search, e-commerce, government & organizations, and social media), covering four category labels defined by WCAG-EM for a structured sample: (1) Common Web Pages and States, (2) Relevant Web Pages and States, (3) Pages of Essential Functionality, and (4) Pages of Web Technologies Relied Upon for Conformance. This dataset contains 951 human labels of four types with an equal distribution of positive and negative labels. Fifty cases are selected as a few-shot fine-tuning set.

CAPTCHA of Cognitive Tests (CCT). Completely Automated Public Turing test to tell Computers and Humans Apart (CAPTCHA) is a widely utilized mechanism for distinguishing human users from automated bots, typically as part of login verification processes. It presents various cognitive tasks designed to challenge and differentiate human recognition abilities from those of machines, making it an ideal test scenario for evaluating cognitive accessibility. In this study, we have collected a dataset of 1,985 CAPTCHA images from the internet, spanning 17 distinct categories of authentication requirements (see Appendix). Among these categories, three meet the criteria outlined in WCAG 2.2 Success Criteria 3.3.9 for cognitive disabilities. We also provide a 50% train split drawn from each class for fine-tuning.

Complete Process Extraction (CPE). To assess MLLMs in Pre-audit Element Localization, we annotated 1,199 pages from the APR dataset, marking pages that contain five key components relevant to complete processes or accessibility defined by WCAG-EM: (1) search bar, (2) select/filter panel, (3) input form, (4) CAPTCHA, and (5) contact information. 598 positive labels and 601 negative labels are annotated. Given the low occurrence frequency of some elements, we constructed 50 representative cases for few-shot fine-tuning, with an equal split between positive and negative examples.

## 6 Experiments

### 6.1 Baselines and Experimental Settings

Baselines and Experimental Settings: For the collective page sampling, we utilize the TPS dataset and compare GRASP with five statistical representations proposed by a recent study on Web Structure Derived Clustering for Accessibility Page Sampling (SDC)(Hambley et al. 2023) and its dimension reduction variants with t-SNE (Van der Maaten and Hinton 2008). We also assess two variants of GRASP: one leveraging GCN(Kipf and Welling 2016) tailored for homophilic graphs, where nodes with similar labels tend to be interconnected, and another utilizing IGNN (Gu et al. 2025) for heterophilic graphs, where nodes of differing labels are more likely to be adjacent. For the MaC, we adopt the APR, CCT, and CPE datasets, with various models including GPT-4o (200B), GPT-4o-mini (8B), Qwen2.5-VL-72B (Qwen2.5), Intern2-VL-8B (Chen et al. 2023), and MiniCPM-V 8B (CPM) (Yao et al. 2024). See more details in Appendix. Metrics: SDC evaluates page sampling using the mean internal cosine similarity of clusters, which reflects cluster cohesiveness but overlooks the quality of the sampled results. We propose two enhanced metrics: (i) The

|  Method | Layout Space |   | Textual Space  |   |
| --- | --- | --- | --- | --- |
|   |  S_{sampled} | D_{intra-inter} | S_{sampled} | D_{intra-inter}  |
|  SDC_content | 56.66 | 9.96 | 89.29 | 2.73  |
|  +TSNE | 55.14 | 6.46 | 88.32 | 1.66  |
|  SDC_struc_cont | 55.61 | 11.53 | 89.59 | 1.93  |
|  +TSNE | 55.89 | 8.91 | 88.89 | 1.60  |
|  SDC_structure | 56.11 | 11.07 | 89.77 | 1.39  |
|  +TSNE | 55.93 | 10.07 | 89.16 | 1.51  |
|  SDC_tags | 54.18 | 10.76 | 88.76 | 2.12  |
|  +TSNE | 55.80 | 9.02 | 89.05 | 1.51  |
|  SDC_tree | 54.17 | 10.55 | 88.79 | 2.09  |
|  +TSNE | 55.86 | 8.81 | 88.85 | 1.63  |
|  GRASP_GCN | 51.54 | 13.05 | 86.99 | 1.59  |
|  GRASP_IGNN | 44.31 | 14.94 | 80.45 | 7.40  |

Table 3: Mean Performance across 495 Websites of GRASP on TPS. The smallest S$_{sampled}$ and largest D$_{intra-inter}$ are highlighted in bold, while the second are underlined.

|   | Category | GPT-4o | 4o-mini | Qwen | CPM  |
| --- | --- | --- | --- | --- | --- |
|  Element | Form | 90.48 | 88.10 | 35.44 | 86.08  |
|   |  Contact | 43.23 | 16.13 | 38.67 | 60.93  |
|   |  Select/Filter | 98.15 | 90.74 | 100 | 85.71  |
|   |  Search | 98.80 | 64.00 | 89.39 | 79.18  |
|   |  CAPTCHA | 92.72 | 87.27 | 92.00 | 92.00  |
|  Page | Com. | 99.05 | 100 | 56.44 | 71.29  |
|   |  Ess. | 82.09 | 82.09 | 90.63 | 85.94  |
|   |  Rel. | 22.95 | 7.10 | 84.92 | 42.46  |
|   |  Tech. | 77.50 | 5.83 | 93.91 | 85.22  |

Table 4: Recall on APR and CPE Datasets. Com., Ess., Rel. and Tech. denote Common Web Pages, Essential Functionality, Relevant Web Pages, and Web Technologies, respectively.

mean inter-cluster cosine similarity S$_{sampled}$ of sampled nodes in the layout and textual embedding spaces derived from BERT and ViT. A lower value indicates greater diversity in sample nodes, suggesting more distinct representativeness. (ii) the difference D$_{intra-inter}$ between the mean intra-cluster cosine similarity of all nodes and the mean inter-cluster cosine similarity of sampled nodes. A larger difference indicates that not only the clusters are internally cohesive but also sample nodes are distinct from one another. (2) For MaC tasks, we use accuracy for recognition and extraction on APR and CPE, and use precision, recall and macro F1-score for classification on CCT.

### 6.2 Performance Analysis

GRASP Performance. The performance of GRASP is presented in Table 3. Several key observations can be made: First, GRASP variants consistently yield more representative node samples, evidenced by lower inter-cluster layout and textual semantic similarities S$_{sampled}$, as well as higher differences D$_{intra-inter}$. Second, GRASP demonstrates superior performance when utilizing the heterophilic IGNN, compared to the homophilic GCN. This suggests that the linkage relationships within websites are more likely to exhibit diverse connections across distinct semantic clusters.

---

|   | Category | GPT-4o | 4o-mini | Qwen | CPM  |
| --- | --- | --- | --- | --- | --- |
|  Element | Form | 68.64 | 86.98 | 32.70 | 46.54  |
|   |  Contact | 54.64 | 47.68 | 53.26 | 56.51  |
|   |  Select/Filter | 78.70 | 89.81 | 85.71 | 48.98  |
|   |  Search | 94.58 | 77.83 | 85.99 | 65.70  |
|   |  CAPTCHA | 79.09 | 93.64 | 95.00 | 87.00  |
|  Page | Com. | 54.50 | 62.09 | 38.92 | 51.23  |
|   |  Ess. | 63.43 | 79.10 | 52.76 | 51.97  |
|   |  Rel. | 54.73 | 49.28 | 77.94 | 44.41  |
|   |  Tech. | 83.75 | 43.75 | 47.60 | 43.48  |

Table 5: Precision on APR and CPE Datasets.

|   | Category | GPT-4o | 4o-mini | Qwen | CPM  |
| --- | --- | --- | --- | --- | --- |
|  Element | Form | 77.95 | 87.06 | 34.36 | 61.54  |
|   |  Contact | 50.38 | 24.04 | 46.03 | 59.15  |
|   |  Select/Filter | 87.60 | 89.91 | 87.50 | 62.69  |
|   |  Search | 98.01 | 77.29 | 88.31 | 73.21  |
|   |  CAPTCHA | 95.33 | 93.20 | 94.85 | 87.62  |
|  Page | Com. | 68.65 | 72.41 | 47.90 | 59.26  |
|   |  Ess. | 77.46 | 79.71 | 65.91 | 64.33  |
|   |  Rel. | 35.44 | 12.81 | 80.21 | 44.57  |
|   |  Tech. | 87.32 | 9.40 | 64.29 | 60.12  |

Table 6: F1 on APR and CPE Datasets.

Finally, the five representations of SDC show comparable performance to each other, with inclusion of t-SNE mostly improving representativeness. GRASP_IGNN demonstrates significantly better representativeness results across both textual and visual spaces, while SDC only performs relatively better in the textual space compared to GRASP_GCN.

MaC Assistant Performance. The performance of MaC in individuality-based page sampling and pre-audit element localization is presented in Table 4, 5 and 6. First, the MLLM assistant demonstrates high accuracy, exceeding 50% for most high-level multimodal semantic understanding and recognition tasks, with several task types reaching over 90%, indicating promising capabilities. Second, larger MLLMs mostly outperform smaller MLLMs with the first or second ranks, although both large and small MLLMs exhibit varying preferences and strengths. GPT-4o excels in extracting smaller elements such as contact forms and search boxes, whereas GPT-4o-mini performs better with larger components like forms and CAPTCHAs. This suggests that smaller MLLMs can also find effective use, highlighting the critical role of selecting an appropriate model or integrating multiple models to improve performance.

MaC Auditor Performance. The results of the MaC on CAPTCHA cognition are presented in Table 7 with several key observations as follows. (1) While the recognition of the existence of CAPTCHAs approaches 100%, the classification results still leave considerable room for optimization. This may be partly attributed to the semantic similarities between the types. However, even when CAPTCHAs are similar, the distinct cognitive tests and operational requirements can introduce different barriers. For example, there are several initial recognition tasks, each followed by different subsequent tasks such as matching, segmentation, or

|   | Model T. | Exist. | P. | R. | F1 | Vio.  |
| --- | --- | --- | --- | --- | --- | --- |
|  MiniCPM-sft 8B | ✓ | 85.64 | 15.00 | 10.88 | 11.72 | 38.20  |
|  Intern2-VL 8B | ✓ | 100 | 49.54 | 43.85 | 45.58 | 99.88  |
|  GPT-4o-mini 8B |  | 91.06 | 27.19 | 16.66 | 19.33 | 93.33  |
|  GPT-4o 200B |  | 96.30 | 34.24 | 27.34 | 29.16 | 97.47  |
|  Qwen2.5-VL 72B |  | 90.75 | 39.79 | 32.47 | 34.55 | 84.96  |

Table 7: Results on Dataset CCT. T. is short for training, while Exist. means recall of existence of CAPTCHAs. P, R, F1 are precision, recall and macro F1-score, respectively. Vio. is the accuracy of violation judgement of cognition test.

|  Category | Exist. | P. | R. | F1 | Vio.  |
| --- | --- | --- | --- | --- | --- |
|  1 | 100 | 100 | 100 | 100 | 100  |
|  2 | 100 | 50.00 | 25.00 | 33.33 | 100  |
|  3 | 100 | 50.00 | 43.75 | 46.67 | 100  |
|  4 | 100 | 100 | 100 | 100 | 100  |
|  5 | 100 | 20.00 | 19.27 | 19.63 | 99.09  |
|  6 | 100 | 100 | 100 | 100 | 100  |
|  7 | 100 | 50.00 | 46.67 | 48.28 | 100  |
|  9 | 100 | 33.33 | 32.96 | 33.15 | 100  |
|  10 | 100 | 25.00 | 21.05 | 22.86 | 100  |
|  11 | 100 | 0 | 0 | 0 | 100  |
|  12 | 100 | 14.29 | 1.90 | 3.36 | 100  |
|  13 | 100 | 100 | 100 | 100 | 100  |
|  14 | 100 | 50.00 | 37.50 | 42.86 | 100  |
|  15 | 100 | 50.00 | 44.64 | 47.17 | 100  |
|  16 | 100 | 25.00 | 24.70 | 24.85 | 99.00  |
|  17 | 100 | 25.00 | 4.17 | 7.14 | 100  |
|  Mean | 100 | 49.54 | 43.85 | 45.58 | 99.88  |

Table 8: Performance of Intern2-VL on CCT. Exist. denotes the recall for CAPTCHA detection. Vio. represents the accuracy of cognitive accessibility violation assessment.

recognition. These variations may introduce different cognitive challenges. (2) Nevertheless, MLLMs demonstrate high accuracy in determining whether CAPTCHAs might impede users with cognitive impairments. This suggests their reasoning capabilities regarding functionality and barriers remain strong, compensating for classification shortcomings.

More case studies are in the Appendix, e.g., limitations of small MLLMs in structured response and fine-tuned small MLLMs outperforming large untrained ones.

## 7 Conclusion

In response to the scalability challenges in web accessibility audits, we present a full-lifecycle WAA framework that operationalizes WCAG-EM through integration of Automation, AI, and Auditor (AAA). Our contributions address critical resource bottlenecks in both sampling and evaluation, including GRASP for representative multimodal page sampling, MAC strategies for MLLM as Copilots for scalable WAA, and a suite of benchmark datasets. They advance the state of scalable, WCAG-EM-aligned accessibility auditing and lay the groundwork for more scalable WAA practices.

---

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No.62372408).

## References

Abramovich, S.; and Patitsas, E. 2024. 'Slipping through the cracks': A Duoethnography of Web Accessibility. In *Proceedings of the 26th International ACM SIGACCESS Conference on Computers and Accessibility*, 1–6.

Ara, J.; and Sik-Lanyi, C. 2025. Automated evaluation of accessibility issues of webpage content: tool and evaluation. *Scientific Reports*, 15(1): 9516.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; Li, B.; Luo, P.; Lu, T.; Qiao, Y.; and Dai, J. 2023. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. *arXiv preprint arXiv:2312.14238*.

Consortium, W. W. W. 2024a. How to Meet WCAG (Quick Reference). Accessed: 2025-3-26.

Consortium, W. W. W. 2024b. Web Content Accessibility Guidelines (WCAG) 2.2. Accessed: 2025-3-26.

Deque Systems, Inc. 2025. AXE-CORE. Accessed: 2025-3-26.

Devlin, J. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

Dosovitskiy, A. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*.

Elglaly, Y. N.; Baker, C. M.; Ross, A. S.; and Shinohara, K. 2024. Beyond HCI: The need for accessibility across the CS curriculum. In *Proceedings of the 55th ACM Technical Symposium on Computer Science Education V. 1*, 324–330.

Group, W. W. 2014. Website Accessibility Conformance Evaluation Methodology (WCAG-EM) 1.0. Accessed: 2025-3-26.

Gu, M.; Yang, G.; Zhou, S.; Ma, N.; Chen, J.; Tan, Q.; Liu, M.; and Bu, J. 2023. Homophily-enhanced structure learning for graph clustering. In *Proceedings of the 32nd ACM International Conference on Information and Knowledge Management*, 577–586.

Gu, M.; Zheng, Z.; Zhou, S.; Liu, M.; Chen, J.; Tan, Q.; Li, L.; and Bu, J. 2025. Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness–Generalization Perspective. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*.

Hambley, A.; Yesilada, Y.; Vigo, M.; and Harper, S. 2023. Web structure derived clustering for optimised web accessibility evaluation. In *Proceedings of the ACM Web Conference 2023*, 1345–1354.

Hamilton, W.; Ying, Z.; and Leskovec, J. 2017. Inductive representation learning on large graphs. *Advances in neural information processing systems*, 30.

He, Z.; Huq, S. F.; and Malek, S. 2025. Enhancing Web Accessibility: Automated Detection of Issues with Generative AI. *Proceedings of the ACM on Software Engineering*, 2(FSE): 2264–2287.

Huang, C.; Ma, A.; Vyasamudri, S.; Puype, E.; Kamal, S.; Cheema, S.; and Lutz, M. 2024. Deep-Learning Approaches for Optimized Web Accessibility: Correcting Violations and Enhancing User Experience.

Kaddour, J.; Harris, J.; Mozes, M.; Bradley, H.; Raileanu, R.; and McHardy, R. 2023. Challenges and Applications of Large Language Models. *arXiv:2307.10169*.

Kil, J.; Mai, Z.; Lee, J.; Chowdhury, A.; Wang, Z.; Cheng, K.; Wang, L.; Liu, Y.; and Chao, W.-L. H. 2024. Mllm-combbench: A comparative reasoning benchmark for multimodal llms. *Advances in Neural Information Processing Systems*, 37: 28798–28827.

Kipf, T. N.; and Welling, M. 2016. Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*.

L. Holliday, E. 2020. The Compliance Mindset: Exploring Accessibility Adoption in Client-Based Settings. In *Proceedings of the 22nd International ACM SIGACCESS Conference on Computers and Accessibility*, 1–3.

Li, W.; Fan, H.; Wong, Y.; Yang, Y.; and Kankanhalli, M. 2024. Improving context understanding in multimodal large language models via multimodal composition learning. In *Forty-first International Conference on Machine Learning*.

Liu, N.; Wang, X.; Wu, L.; Chen, Y.; Guo, X.; and Shi, C. 2022. Compact Graph Structure Learning via Mutual Information Compression. In *Proceedings of the ACM Web Conference 2022*, 1601–1610.

López-Gil, J.-M.; and Pereira, J. 2024. Turning manual web accessibility success criteria into automatic: an LLM-based approach. *Universal Access in the Information Society*, 1–16.

Othman, A.; Dhouib, A.; and Nasser Al Jabor, A. 2023. Fostering websites accessibility: A case study on the use of the Large Language Models ChatGPT for automatic remediation. In *Proceedings of the 16th International Conference on PErvasive Technologies Related to Assistive Environments*, 707–713.

Project, T. S. 2025. Selenium. Accessed: 2025-3-26.

Sharif, A.; Pruekcharoen, P.; Ramesh, T.; Shang, R.; Williams, S.; and Hsieh, G. 2022. 'What's going on in Accessibility Research?' Frequencies and Trends of Disability Categories and Research Domains in Publications at ASSETS. In *Proceedings of the 24th International ACM SIGACCESS Conference on Computers and Accessibility*, 1–5.

SolarWinds Worldwide, L. 2025. Webpages Are Getting Larger Every Year, and Here's Why it Matters. Accessed: 2025-4-9.

Suh, H.; Tafreshipour, M.; Malek, S.; and Ahmed, I. 2025. Human or LLM? A Comparative Study on Accessible Code Generation Capability. *arXiv preprint arXiv:2503.15885*.

---

Van der Maaten, L.; and Hinton, G. 2008. Visualizing data using t-SNE. *Journal of machine learning research*, 9(11).

W3C Web Accessibility Initiative (WAI). 2023. Evaluating Web Accessibility Overview. Accessed: 2025-3-25.

Web Accessibility Initiative (WAI). 2024. Introduction to Web Accessibility. Accessed: 2024-11-26. First published: February 2005. Last updated: 7 March 2024.

WebAIM. 2025. WebAIM: The WebAIM Million - The 2025 report on the accessibility of the top 1,000,000 home pages. Accessed: 2025-4-9.

WebAIM, Utah State University. 2025. WAVE. Accessed: 2025-3-26.

Yao, Y.; Yu, T.; Zhang, A.; Wang, C.; Cui, J.; Zhu, H.; Cai, T.; Li, H.; Zhao, W.; He, Z.; et al. 2024. MiniCPM-V: A GPT-4V Level MLLM on Your Phone. *arXiv preprint arXiv:2408.01800*.

Yu, Z.; Bu, J.; Shen, C.; Wang, W.; Dai, L.; Zhou, Q.; and Zhao, C. 2020. A multi-site collaborative sampling for web accessibility evaluation. In *Computers Helping People with Special Needs: 17th International Conference, ICCHP 2020, Lecco, Italy, September 9–11, 2020, Proceedings, Part I* 17, 329–335. Springer.

Zhang, M.; Wang, C.; Bu, J.; Yu, Z.; Lu, Y.; Zhang, R.; and Chen, C. 2015a. An optimal sampling method for web accessibility quantitative metric. In *Proceedings of the 12th International Web for All Conference*, 1–4.

Zhang, M.-n.; Wang, C.; Bu, J.-j.; Yu, Z.; Zhou, Y.; and Chen, C. 2015b. A sampling method based on URL clustering for fast web accessibility evaluation. *Frontiers of Information Technology & Electronic Engineering*, 16(6): 449–456.

Zhong, M.; Chen, R.; Chen, X.; Fogarty, J.; and Wobbrock, J. O. 2025. ScreenAudit: Detecting Screen Reader Accessibility Errors in Mobile Apps Using Large Language Models. In *Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems*, 1–19.

---

# Appendix

## Datasets, Experimental Settings and Prompts

**Datasets Information** Categories of the TPS dataset are presented in Table 9, while 17 CAPTCHA categories of the CCT datasets are documented in Table 10. All datasets and its detailed information are available at our open repository.

**Experimental Settings** GRASP employs the open-source pre-trained models BERT and ViT for the initial textual and visual embedding construction. During training, the parameters of these two models are frozen, and only the parameters of the GNN and transformation matrices are updated. For clustering, we apply the k-means algorithm and graph learning pipeline in HoLe uniformly across all baselines and GRASP, setting the number of clusters and sampled nodes to 20 and the iteration number of representativeness clustering and graph learning to 5 for consistency. For the MLLMs, we perform fine-tuning on open-source small models with 8B parameters, while for GPT-4o-mini and larger models (Qwen2.5-VL-72B, GPT-4o), we conduct inference directly without training.

**Detailed Prompts** Detailed prompts for each datasets are presented in Prompt 1, Prompt 2 and Prompt 3.

## Empirical Lessons from Case Studies

**Limitations of Small MLLMs in Structured Response.** Small-scale MLLMs (e.g., 8B) often struggle to generate well-structured or instruction-compliant answers without fine-tuning. For example, when prompted to provide binary outputs (e.g., a simple 'yes' or 'no') to facilitate downstream parsing, large models generally comply, whereas small models frequently embed the answer within lengthy and verbose explanations, making answer analysis difficult. In more complex scenarios, such as analyzing CAPTCHA images from the CCT dataset—where the model must determine the presence of a CAPTCHA, identify its type, and assess its WCAG compliance, we require responses to be formatted as a JSON object with three fields (see detailed prompts in Appendix). Without fine-tuning, small models tend to ignore this instruction, returning unrelated or overly verbose explanations, or even refusing to answer. *However, after a brief fine-tuning phase, these models learn to adhere to the required structure.* While minor formatting errors may persist, the responses become reliably parseable. *This suggests that for tasks demanding specific output structures, small MLLMs can be effectively adapted via light fine-tuning, mitigating one of their key usability challenges.*

**Fine-Tuned Small Models Can Outperform Larger Untrained Models.** In tasks requiring complex multimodal reasoning, such as multi-stage interpretation and response generation over CAPTCHA inputs, *small fine-tuned models demonstrated surprising capabilities.* Specifically, after being trained on 50% of the CCT dataset, Intern2-VL with 8B parameters significantly outperformed other larger baselines, including GPT-4o (200B), achieving nearly 100% in the reasoning of CAPTCHA existence and WCAG conformance within this specialized task. *This highlights the latent potential of smaller models to serve as domain-specific*

*experts in accessibility, provided they are trained on high-quality, targeted datasets.* The implications are twofold: first, it emphasizes the value of domain-adapted datasets in unlocking performance gains; and second, it points to a resource-efficient pathway for deploying MLLMs in accessibility applications. Compared to their larger counterparts, small models are cheaper to deploy and easier to integrate into public service infrastructures, making them promising candidates for scalable, inclusive, and democratized accessibility solutions.

## Additional Results

Various MLLM results on different CAPTCHA types are presented in Table 11, 12, 13, 14 and 15 with corresponding type name of category ids in Table 10.

## Broader social impact

**Scalable, standardized audits for web accessibility:** Our AAA framework operationalizes the international standard WCAG-EM through a human-AI collaboration paradigm, enabling scalable accessibility audits that reduce reliance on labor-intensive manual processes.

**Advance AI-driven accessibility research:** The MaC and GRASP components provide novel methodologies, accelerating AI-driven research in web accessibility, supporting more inclusive digital experiences for users with disabilities, particularly in underserved regions or domains.

**Open datasets for community impact:** Our open, real-world datasets provide a foundation for transparent benchmarking, fostering community-wide progress in WAA.

**Informing next-generation standards:** As WCAG-EM is undergoing a potential update, our work offers timely and technically grounded insights that can shape future accessibility standards, amplifying societal benefits at scale.

---

Table 9: Website Categories of TPS Dataset

|  (1) Consumer Electronics & Digital Devices | (60) Hospitals & Clinics  |
| --- | --- |
|  (2) Government Functions | (61) Radio & Television  |
|  (3) Advertising & Marketing | (62) Primary & Secondary Education  |
|  (4) Government Portals | (63) IT News & Information  |
|  (5) Finance & Economics | (64) General Discussion Forums  |
|  (6) Pharmaceuticals & Pharmacy | (65) Online Lending Platforms  |
|  (7) Hotels & Hospitality | (66) Video & Film Platforms  |
|  (8) Wedding & Photography Studios | (67) Overseas Study & Exchange  |
|  (9) Business Services | (68) Medical Devices & Equipment  |
|  (10) Hardware & Electrical Engineering | (69) Computer Hardware  |
|  (11) E-Commerce Services | (70) Astrology & Horoscopes  |
|  (12) Military & National Defense | (71) Ticketing & Reservations  |
|  (13) Daily Chemicals & Household Products | (72) Fiction & Literature Platforms  |
|  (14) Astronomy & History | (73) Lottery & Sports Betting  |
|  (15) Online Education | (74) Commercial & Department Stores  |
|  (16) E-Commerce Platforms | (75) Lifestyle Encyclopedias  |
|  (17) Foreign Language Resources | (76) Classified Information  |
|  (18) Personalized Messaging & Avatars | (77) Construction Materials  |
|  (19) Food & Culinary Services | (78) Animation & Comics Platforms  |
|  (20) Search Engines | (79) Bicycles & Motor Vehicles  |
|  (21) Apparel & Accessories | (113) Data Analytics  |
|  (22) Franchise & Investment | (80) Regional Portals  |
|  (23) Humor & Jokes | (81) Maternal & Infant Platforms  |
|  (24) Power & Water Utilities | (82) Consumer Electronics & Digital Devices  |
|  (25) Encyclopedias & Dictionaries | (83) Education & Examination  |
|  (26) Automotive Platforms | (84) Higher Education Institutions  |
|  (27) Banking & Insurance | (85) Chemical & Energy Industries  |
|  (28) Advertising Networks | (86) Public Organizations  |
|  (29) Libraries & Exhibitions | (87) Home Furnishing & Building Materials  |
|  (30) Logistics & Transportation | (88) Music Platforms  |
|  (31) Software Downloads | (89) Agriculture, Forestry,  |
|  (32) Broadcasting & Telecommunications | Animal Husbandry & Fisheries  |
|  (33) Collectibles & Hobbies | (90) Social Sciences & Humanities  |
|  (34) Household Items & Accessories | (91) Automotive Manufacturers  |
|  (35) Food & Beverage | (92) Celebrity & Fan Communities  |
|  (36) Driving Schools & Training | (93) Blog Platforms  |
|  (37) Real Estate Platforms | (94) Email & Communication Services  |
|  (38) Cybersecurity | (95) Machinery & Industrial Equipment  |
|  (39) Gaming Platforms | (96) Job Search & Recruitment  |
|  (40) Web Directories | (97) Law & Regulations  |
|  (41) Training & Certification Institutions | (98) Public Institutions  |
|  (42) Software Applications & Utilities | (99) Multinational Corporations  |
|  (43) Health & Wellness | (100) Domain & Hosting Services  |
|  (44) Group Buying Platforms | (101) Sports & Athletics  |
|  (45) Chat & Social Networking | (102) Electronic Components  |
|  (46) Web Portals | (103) Shopping & Product Reviews  |
|  (47) Plumbing & Security Systems | (104) Social Networking Platforms  |
|  (48) Travel Agencies | (105) Traffic & Maps  |
|  (49) Domestic Services | (106) Packaging & Printing  |
|  (50) Cashback & Price Comparison | (107) Entertainment & Fashion  |
|  (51) General Lookup Tools | (108) News & Press Publications  |
|  (52) Design Resources | (109) Petrochemical & Energy  |
|  (53) Webmaster Tools & Resources | (110) Images & Photography  |
|  (54) Auto Parts & Accessories | (111) Educational Information  |
|  (55) Entrepreneurship & Investment | (112) Travel Platforms  |
|  (56) Beauty & Cosmetic Surgery | (114) Securities & Stock Platforms  |
|  (57) Textiles & Leather Products | (115) Pets & Toys  |
|  (58) Mobile Devices & Gadgets | (116) Ball Sports  |
|  (59) Technology & Programming | (117) Electronic Payment Systems  |

---

### Prompt 3: CCT Dataset

1 Please analyze the provided image information accurately and answer the following questions. Return the results in **JSON format**:
2
3 - **Question a**: Does the image contain a CAPTCHA? (Output 1 for yes; 0 for no)
4 - **Question b**: If a CAPTCHA is present, which of the following 17 types does it belong to? (If not present, output 0)
5 1. Biometric verification
6 2. Click on a specific area
7 3. Slide a specific component
8 4. Drag a button to rotate an image to the correct angle
9 5. Drag an element to complete a puzzle
10 6. Swap elements to complete a puzzle
11 7. Object recognition (only visible and touchable objects)
12 8. Personal content recognition (user-provided content such as images, text, etc.)
13 9. Concept recognition (recognizing abstract concepts, text, graphics, etc.)
14 10. Matching elements such as graphics, text, or patterns provided by the website
15 11. Verification involving domain-specific conceptual knowledge
16 12. Composite verification involving multiple cognitive requirements
17 13. Draw along a specified path
18 14. Object segmentation
19 15. Mathematical operations
20 16. Cross-device verification, such as scanning a QR code or sending/receiving text via SMS, email, etc.
21 17. Other types of CAPTCHA not covered above, or confusing CAPTCHA types that cannot be clearly classified
22 - **Question c**: Does the CAPTCHA fail to meet WCAG 2.2 AA standards and the cognitive function test requirements? (Output 1 if it fails, 0 if it meets the standard)
23 - **Question d**: If the answer to question c is 1, briefly explain why it does not comply; otherwise, return an empty string.
24
25 Ensure that the output is in standard JSON format, for example:
26 'json'
27 {
28 "a": 1,
29 "b": 10,
30 "c": 1,
31 "d": "The CAPTCHA requires users to recognize specific objects or abstract concepts, which may pose challenges for users with cognitive disabilities and does not comply with WCAG 2.2.",
32 }
33 ...

### Prompt 1: APR and CPE Dataset

1 1. Please analyze the provided webpage screenshot and determine whether the page **contains** the specified element.
2 - **Element Name**: Search Box
3 - **Element Description**: Check whether the page includes a search box element, such as an input field for searching or a search button.
4 - **Output**: "Contains" or "Does Not Contain"
5 2. Please analyze the provided webpage screenshot and determine whether the page **contains** the specified element.
6 - **Element Name**: Filters
7 - **Element Description**: Check whether the page includes filtering elements, such as filter buttons, filter criteria, etc.
8 - **Output**: "Contains" or "Does Not Contain"
9 3. Please analyze the provided webpage screenshot and determine whether the page **contains** the specified element.
10 - **Element Name**: Contact Information
11 - **Element Description**: Check whether the page includes ways to contact the site owner or staff, such as QR codes, phone numbers, email addresses, or physical addresses.
12 - **Output**: "Contains" or "Does Not Contain"
13 4. Please analyze the provided webpage screenshot and determine whether the page **contains** the specified element.
14 - **Element Name**: Form
15 - **Element Description**: Check whether the page includes any form elements, such as login, registration, or submission forms.
16 - **Output**: "Contains" or "Does Not Contain"
17 5. Please analyze the provided webpage screenshot and determine whether the page **contains** the specified element.
18 - **Element Name**: CAPTCHA
19 - **Element Description**: Check whether the page includes any CAPTCHA elements, such as slider puzzles, image-based CAPTCHAs, or SMS verification codes.
20 - **Output**: "Contains" or "Does Not Contain"
21 6. Please analyze the provided webpage screenshot and determine whether the page belongs to the specified page type.
22 - **Page Type**: Common Web Pages

---

# Prompt 2: APR and CPE Dataset

1 - **Type Description**: Identify whether the page is a common web page or a typical web application state. These pages are usually directly accessible from the homepage or navigated to via headers, navigation bars, or footers. Examples include the homepage, landing pages of submodules, login pages, and category pages.

2 - **Output**: "Yes" or "No"

3 7. Please analyze the provided webpage screenshot and determine whether the page belongs to the specified page type.

4 - **Page Type**: Essential Functionality

5 - **Type Description**: Identify whether the page supports the core functionality of the website. While some functions are obvious, others may require more exploration. The goal is to recognize key actions users can perform on the site. Examples include product purchasing pages in an online store, registration pages, video playback pages, or survey submission pages.

6 - **Output**: "Yes" or "No"

7 8. Please analyze the provided webpage screenshot and determine whether the page belongs to the specified page type.

8 - **Page Type**: Relevant Web Pages

9 - **Type Description**: Identify whether the page is particularly relevant to accessibility or people with disabilities. This includes pages about accessibility features, help and support, language or font settings, privacy policies, or contact details. These may or may not already be identified as part of the common pages.

10 - **Output**: "Yes" or "No"

11 9. Please analyze the provided webpage screenshot and determine whether the page belongs to the specified page type.

12 - **Page Type**: Web Technologies

13 - **Type Description**: Identify the key web technologies used on the page that affect accessibility. This includes fundamental technologies like HTML and CSS, assistive technologies like JavaScript and WAI-ARIA, or specialized technologies like SMIL, SVG, and PDF. Examples include pages using slider CAPTCHA, interactive maps, offline tools like Google Docs, or live data maps.

14 - **Output**: "Yes" or "No"

|  id | Type | P.  |
| --- | --- | --- |
|  1 | Biometric verification | ✓  |
|  2 | Click on a specific area |   |
|  3 | Slide a specific component |   |
|  4 | Drag a button to rotate images to the correct angle |   |
|  5 | Drag an element to complete a puzzle |   |
|  6 | Swap elements to complete a puzzle |   |
|  7 | Object recognition (only visible and touchable objects) |   |
|  8 | Personal content recognition of user-provided content such as images and text | ✓  |
|  9 | Concept recognition of abstract concepts, text, graphics |   |
|  10 | Matching elements such as graphics, text or patterns, provided by the website |   |
|  11 | Verification involving domain-specific knowledge |   |
|  12 | Composite verification involving multiple cognitive requirements |   |
|  13 | Draw along a specified path |   |
|  14 | Object segmentation |   |
|  15 | Mathematical operations |   |
|  16 | Cross-device verification, including scanning a QR code, sending or receiving text via SMS and email that allows copy and paste. | ✓  |
|  17 | Other types of CAPTCHA not covered above, or confusing CAPTCHA types that cannot be clearly classified |   |

Table 10: CAPTCHA Categories in the CCT Dataset. P. indicates compliance with WCAG 2.2 Success Criterion 3.3.9 (Accessible Authentication - Enhanced, Level AAA). Notably, 7 Object Recognition category also satisfies Criterion 3.3.8 (Accessible Authentication - Minimum, Level AA).

|  Category | Exist. | P. | R. | F1 | Vio.  |
| --- | --- | --- | --- | --- | --- |
|  1 | 100 | 100 | 100 | 100 | 93.33  |
|  2 | 100 | 50.00 | 33.33 | 40.00 | 100  |
|  3 | 100 | 25.00 | 15.62 | 19.23 | 100  |
|  4 | 100 | 50.00 | 46.15 | 48.00 | 100  |
|  5 | 100 | 25.00 | 24.09 | 24.53 | 100  |
|  6 | 100 | 100 | 100 | 100 | 100  |
|  7 | 100 | 33.33 | 6.90 | 11.43 | 100  |
|  9 | 100 | 20.00 | 16.95 | 18.35 | 99.81  |
|  10 | 100 | 33.33 | 2.63 | 4.88 | 100  |
|  11 | 100 | 0 | 0 | 0 | 100  |
|  12 | 100 | 0 | 0 | 0 | 100  |
|  13 | 100 | 50.00 | 40.00 | 44.44 | 100  |
|  14 | 100 | 0 | 0 | 0 | 100  |
|  15 | 100 | 50.00 | 45.61 | 47.71 | 87.72  |
|  16 | 59.00 | 11.11 | 6.20 | 7.96 | 96.80  |
|  17 | 81.82 | 0 | 0 | 0 | 81.82  |
|  Mean | 96.30 | 34.24 | 27.34 | 29.16 | 97.47  |

Table 11: GPT4o on CCT. T. is training, while Exist. means recall of existence of CAPTCHAs. P., R., F1 are precision, recall and macro F1-score, respectively. Vio. denotes accuracy of violation judgment of cognition accessibility.

---

|  Category | Exist. | P. | R. | F1 | Vio.  |
| --- | --- | --- | --- | --- | --- |
|  1 | 66.67 | 50.00 | 33.33 | 40.00 | 86.67  |
|  2 | 100 | 50.00 | 16.67 | 25.00 | 100  |
|  3 | 100 | 25.00 | 20.31 | 22.41 | 93.75  |
|  4 | 100 | 33.33 | 15.38 | 21.05 | 92.31  |
|  5 | 99.54 | 20.00 | 17.63 | 18.74 | 88.58  |
|  6 | 100 | 100 | 100 | 100 | 100  |
|  7 | 100 | 33.33 | 12.64 | 18.33 | 93.10  |
|  9 | 98.70 | 9.09 | 0.03 | 0.07 | 92.19  |
|  10 | 100 | 16.67 | 10.96 | 13.23 | 97.37  |
|  11 | 100 | 0 | 0 | 0 | 100  |
|  12 | 100 | 0 | 0 | 0 | 100  |
|  13 | 100 | 50.00 | 10.00 | 16.67 | 100  |
|  14 | 100 | 0 | 0 | 0 | 100  |
|  15 | 100 | 33.33 | 27.49 | 30.13 | 82.46  |
|  16 | 19.30 | 14.29 | 2.14 | 3.73 | 94.10  |
|  17 | 72.73 | 0 | 0 | 0 | 72.73  |
|  **Mean** | **91.06** | **27.19** | **16.66** | **19.33** | **93.33**  |

Table 12: GPT4o-mini on CCT.

|  Category | Exist. | P. | R. | F1 | Vio.  |
| --- | --- | --- | --- | --- | --- |
|  1 | 0 | 0 | 0 | 0 | 25.00  |
|  2 | 50.00 | 0 | 0 | 0 | 100  |
|  3 | 100 | 33.33 | 25.00 | 28.57 | 0  |
|  4 | 100 | 33.33 | 23.81 | 27.78 | 0  |
|  5 | 99.09 | 20.00 | 18.36 | 19.15 | 2.73  |
|  6 | 100 | 100 | 100 | 100 | 0  |
|  7 | 86.67 | 20.00 | 1.33 | 2.50 | 60.00  |
|  9 | 99.26 | 0 | 0 | 0 | 90.00  |
|  10 | 100 | 0 | 0 | 0 | 57.89  |
|  11 | 100 | 0 | 0 | 0 | 71.43  |
|  12 | 100 | 0 | 0 | 0 | 53.33  |
|  13 | 100 | 0 | 0 | 0 | 0  |
|  14 | 100 | 0 | 0 | 0 | 0  |
|  15 | 96.43 | 0 | 0 | 0 | 35.71  |
|  16 | 55.40 | 0 | 0 | 0 | 31.80  |
|  17 | 83.33 | 33.33 | 5.56 | 9.52 | 83.33  |
|  **Mean** | **85.64** | **15.00** | **10.88** | **11.72** | **38.20**  |

Table 14: MiniCPM on CCT.

|  Category | Exist. | P. | R. | F1 | Vio.  |
| --- | --- | --- | --- | --- | --- |
|  1 | 37.50 | 50.00 | 25.00 | 33.33 | 100  |
|  2 | 50.00 | 0 | 0 | 0 | 0  |
|  3 | 100 | 50.00 | 43.75 | 46.67 | 0  |
|  4 | 100 | 100 | 100 | 100 | 100  |
|  5 | 100 | 25.00 | 22.05 | 23.43 | 98.18  |
|  6 | 100 | 100 | 100 | 100 | 100  |
|  7 | 100 | 50.00 | 30.00 | 37.50 | 100  |
|  9 | 100 | 0 | 0 | 0 | 85.56  |
|  10 | 100 | 33.33 | 17.54 | 22.99 | 100  |
|  11 | 100 | 0 | 0 | 0 | 100  |
|  12 | 100 | 20.00 | 1.33 | 2.50 | 100  |
|  13 | 100 | 100 | 100 | 100 | 100  |
|  14 | 100 | 0 | 0 | 0 | 100  |
|  15 | 100 | 50.00 | 48.21 | 49.09 | 92.86  |
|  16 | 81.20 | 33.33 | 27.53 | 30.16 | 99.40  |
|  17 | 83.33 | 25.00 | 4.17 | 7.14 | 83.33  |
|  **Mean** | **90.75** | **39.79** | **32.47** | **34.55** | **84.96**  |

Table 13: Qwen2.5 on CCT.

|  Category | Exist. | P. | R. | F1 | Vio.  |
| --- | --- | --- | --- | --- | --- |
|  1 | 100 | 100 | 100 | 100 | 100  |
|  2 | 100 | 50.00 | 25.00 | 33.33 | 100  |
|  3 | 100 | 50.00 | 43.75 | 46.67 | 100  |
|  4 | 100 | 100 | 100 | 100 | 100  |
|  5 | 100 | 20.00 | 19.27 | 19.63 | 99.09  |
|  6 | 100 | 100 | 100 | 100 | 100  |
|  7 | 100 | 50.00 | 46.67 | 48.28 | 100  |
|  9 | 100 | 33.33 | 32.96 | 33.15 | 100  |
|  10 | 100 | 25.00 | 21.05 | 22.86 | 100  |
|  11 | 100 | 0 | 0 | 0 | 100  |
|  12 | 100 | 14.29 | 1.90 | 3.36 | 100  |
|  13 | 100 | 100 | 100 | 100 | 100  |
|  14 | 100 | 50.00 | 37.50 | 42.86 | 100  |
|  15 | 100 | 50.00 | 44.64 | 47.17 | 100  |
|  16 | 100 | 25.00 | 24.70 | 24.85 | 99.00  |
|  17 | 100 | 25.00 | 4.17 | 7.14 | 100  |
|  **Mean** | **100** | **49.54** | **43.85** | **45.58** | **99.88**  |

Table 15: Internvl2-sft on CCT.

---

