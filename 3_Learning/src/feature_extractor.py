"""
feature_extractor.py

Extracts node features from DOM elements:
- Text embeddings via sentence-transformers
- Visual bounding boxes via Playwright
- Axe-core violation labels
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer

from graph_sources import (
    GRAPH_SOURCE_DOM,
    GRAPH_SOURCE_RENDERED_VISUAL,
    apply_visual_edges,
    build_graph,
)
from html_graph_builder import DOMNode
from wcag_rules import (
    NUM_RULES,
    RULE_INDEX,
    rule_ids_for_graph_source,
    rule_mask_for_graph_source,
)


class FeatureExtractor:
    """Extracts multimodal features for DOM nodes."""
    
    def __init__(self, text_model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.text_model = SentenceTransformer(text_model_name, device=device)
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        print(f"Loaded text model: {text_model_name} (dim={self.text_dim})")
    
    def extract_text_embeddings(self, node_map: Dict[int, DOMNode]) -> torch.Tensor:
        """Extract sentence embeddings for visible text of each node."""
        texts = []
        valid_ids = []
        
        for node_id, node in node_map.items():
            text = node.text_content.strip()
            if text:
                texts.append(text[:512])  # Truncate long text
                valid_ids.append(node_id)
            else:
                texts.append("")
                valid_ids.append(node_id)
        
        # Batch encode
        embeddings = self.text_model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
            batch_size=32,
        )
        
        return embeddings.cpu()
    
    def extract_visual_features(
        self,
        html_path: Path,
        node_map: Dict[int, DOMNode],
        viewport: Dict[str, int] = {"width": 1280, "height": 720},
    ) -> Dict[int, Dict]:
        """
        Use Playwright to render the page and extract bounding boxes for each element.
        Maps DOM nodes to rendered elements via CSS selectors or tag+attribute matching.
        """
        print(f"Rendering {html_path} with Playwright...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport=viewport)
            
            # Load enough DOM for layout extraction without waiting on external
            # assets referenced by saved pages.
            page.goto(
                f"file://{html_path.resolve()}",
                wait_until="domcontentloaded",
                timeout=10_000,
            )
            page.wait_for_load_state("domcontentloaded")
            
            # Get viewport size
            viewport_size = page.viewport_size
            vw = viewport_size["width"] if viewport_size else 1280
            vh = viewport_size["height"] if viewport_size else 720
            
            # Extract bounding boxes for all elements
            # We use a JavaScript approach to get element info with their computed styles
            elements_info = page.evaluate("""
                () => {
                    const results = [];
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_ELEMENT,
                        null,
                        false
                    );
                    
                    let node;
                    while (node = walker.nextNode()) {
                        const rect = node.getBoundingClientRect();
                        const style = window.getComputedStyle(node);
                        results.push({
                            tag: node.tagName.toLowerCase(),
                            id: node.id,
                            class: node.className,
                            cssPath: getCssPath(node),
                            xpath: getXPath(node),
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                            visible: rect.width > 0 && rect.height > 0 && 
                                     style.display !== 'none' && 
                                     style.visibility !== 'hidden',
                            text: node.innerText?.substring(0, 200) || ''
                        });
                    }
                    return results;
                    
                    function getXPath(element) {
                        if (element.id) return '//*[@id="' + element.id + '"]/';
                        const parts = [];
                        while (element && element.nodeType === Node.ELEMENT_NODE) {
                            let index = 1;
                            let sibling = element.previousSibling;
                            while (sibling) {
                                if (sibling.nodeType === Node.ELEMENT_NODE && 
                                    sibling.tagName === element.tagName) index++;
                                sibling = sibling.previousSibling;
                            }
                            const tagName = element.tagName.toLowerCase();
                            const part = index > 1 ? tagName + '[' + index + ']' : tagName;
                            parts.unshift(part);
                            element = element.parentNode;
                        }
                        return parts.length ? '/' + parts.join('/') : '';
                    }

                    function getCssPath(element) {
                        const parts = [];
                        while (element && element.nodeType === Node.ELEMENT_NODE) {
                            const tag = element.tagName.toLowerCase();
                            if (element.id) {
                                parts.unshift(tag + '#' + CSS.escape(element.id));
                                break;
                            }
                            let index = 1;
                            let sibling = element.previousElementSibling;
                            while (sibling) {
                                if (sibling.tagName === element.tagName) index++;
                                sibling = sibling.previousElementSibling;
                            }
                            parts.unshift(tag + ':nth-of-type(' + index + ')');
                            element = element.parentElement;
                        }
                        return parts.join(' > ');
                    }
                }
            """)
            
            browser.close()
        
        # Map rendered elements back to our DOM nodes
        # Simple heuristic: match by tag name, id, class, and text similarity
        visual_features = {}
        
        for node_id, node in node_map.items():
            if node.is_text or node.tag in {"__TEXT__", "__UNK__"}:
                continue
            
            best_match = None
            best_score = -1
            
            node_id_attr = node.attrs.get("id", "")
            node_class = " ".join(node.attrs.get("class", [])) if isinstance(node.attrs.get("class", []), list) else str(node.attrs.get("class", ""))
            node_text = node.text_content.strip()[:100]
            node_dom_path = getattr(node, "dom_path", "")
            
            for elem in elements_info:
                score = 0

                if node_dom_path and elem.get("cssPath") == node_dom_path:
                    score += 100
                
                # Tag match
                if elem["tag"] == node.tag:
                    score += 10
                
                # ID match
                if node_id_attr and elem["id"] == node_id_attr:
                    score += 50
                
                # Class overlap
                elem_classes = set(elem["class"].split()) if isinstance(elem["class"], str) else set()
                node_classes = set(node_class.split())
                if node_classes and elem_classes:
                    overlap = len(node_classes & elem_classes) / max(len(node_classes), len(elem_classes))
                    score += overlap * 20
                
                # Text similarity (first 50 chars)
                if node_text and elem["text"]:
                    elem_text = elem["text"][:50]
                    if node_text[:50] == elem_text:
                        score += 15
                    elif node_text[:30] in elem_text or elem_text[:30] in node_text:
                        score += 5
                
                if score > best_score:
                    best_score = score
                    best_match = elem
            
            if best_match and best_score > 5:
                visual_features[node_id] = {
                    "x": best_match["x"] / vw,
                    "y": best_match["y"] / vh,
                    "width": best_match["width"] / vw,
                    "height": best_match["height"] / vh,
                    "visible": best_match["visible"],
                }
            else:
                visual_features[node_id] = {
                    "x": -1, "y": -1, "width": -1, "height": -1, "visible": False
                }
        
        # Update node_map with visual features
        for node_id, feats in visual_features.items():
            if node_id in node_map:
                node_map[node_id].bbox = {
                    "x": feats["x"] * vw if feats["x"] >= 0 else -1,
                    "y": feats["y"] * vh if feats["y"] >= 0 else -1,
                    "width": feats["width"] * vw if feats["width"] >= 0 else -1,
                    "height": feats["height"] * vh if feats["height"] >= 0 else -1,
                }
                node_map[node_id].is_visible = feats["visible"]
        
        return visual_features
    
    def load_axe_labels(
        self,
        axe_report_path: Path,
        node_map: Dict[int, DOMNode],
        graph_source: str = GRAPH_SOURCE_DOM,
    ) -> tuple:
        """
        Load axe-core report and map violations to DOM nodes.
        Uses CSS selector matching for accurate mapping.
        
        Returns:
            - node_labels_binary: [N] long tensor, 1 if element has any violation
            - node_labels_multi: [N, NUM_RULES] float tensor, multi-hot encoded rules
        """
        with open(axe_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        allowed_rule_ids = set(rule_ids_for_graph_source(graph_source))
        
        # Collect all violated node targets with their rule info
        violated_nodes = []  # List of (target_str, rule_id, impact)
        
        def most_specific_target(target) -> str:
            if isinstance(target, str):
                return target
            if isinstance(target, list):
                for item in reversed(target):
                    selector = most_specific_target(item)
                    if selector:
                        return selector
                return ""
            return str(target) if target else ""

        for violation in report.get("violations", []):
            rule_id = violation["id"]
            if rule_id not in allowed_rule_ids:
                continue
            impact = violation.get("impact", "minor")
            for node in violation.get("nodes", []):
                target = node.get("target", [])
                if target:
                    # Target is usually a list of selectors; nested lists can appear
                    # for iframe/shadow traversal, so peel down to the final selector.
                    target_str = most_specific_target(target)
                    if target_str:
                        violated_nodes.append((target_str, rule_id, impact))
        
        node_labels_binary = torch.zeros(len(node_map), dtype=torch.long)
        node_labels_multi = torch.zeros(len(node_map), NUM_RULES, dtype=torch.float)
        
        # AYMAN_NOTE: Root cause of 0 recall
        # Build lookups for exact CSS matching first, then heuristic fallback.
        root_element = node_map[0].element if node_map else None
        element_to_node = {
            id(node.element): node_id
            for node_id, node in node_map.items()
            if not node.is_text
        }

        def nearest_accessible_node_id(element) -> Optional[int]:
            """Project a DOM element to the closest node retained in the graph."""
            direct = element_to_node.get(id(element))
            if direct is not None:
                return direct

            for parent in getattr(element, "parents", []):
                parent_id = element_to_node.get(id(parent))
                if parent_id is not None:
                    return parent_id

            for child in getattr(element, "descendants", []):
                child_id = element_to_node.get(id(child))
                if child_id is not None:
                    return child_id

            return None

        def apply_label(node_id: int, rule_id: str, rule_idx: int, impact: str) -> None:
            if graph_source == GRAPH_SOURCE_RENDERED_VISUAL:
                node = node_map[node_id]
                if not getattr(node, "is_visible", False):
                    return
            node_labels_binary[node_id] = 1
            node_labels_multi[node_id, rule_idx] = 1.0
            node_map[node_id].axe_violations.append(rule_id)
            node_map[node_id].axe_impact = impact

        # Build a lookup from id/class/tag to node_ids for faster fallback matching
        id_to_nodes = {}
        class_to_nodes = {}
        tag_to_nodes = {}
        
        for node_id, node in node_map.items():
            if node.is_text:
                continue
            
            tag_to_nodes.setdefault(node.tag, []).append(node_id)
            
            if node.attrs.get("id"):
                id_to_nodes.setdefault(node.attrs["id"], []).append(node_id)
            
            node_classes = node.attrs.get("class", [])
            if isinstance(node_classes, str):
                node_classes = node_classes.split()
            for cls in node_classes:
                class_to_nodes.setdefault(cls, []).append(node_id)
        
        # Match each violated target to DOM nodes
        matched_count = 0
        for target_str, rule_id, impact in violated_nodes:
            target_str = target_str.strip()
            rule_idx = RULE_INDEX.get(rule_id)
            if rule_idx is None:
                continue  # Unknown rule
            
            matched = False

            # Strategy 0: exact CSS selector match using the parsed DOM.
            # axe targets often contain child combinators, nth-child, and escaped Tailwind classes.
            if root_element is not None:
                try:
                    selected_elements = root_element.select(target_str)
                except Exception:
                    selected_elements = []

                exact_node_ids = []
                for elem in selected_elements:
                    node_id = nearest_accessible_node_id(elem)
                    if node_id is not None:
                        exact_node_ids.append(node_id)

                if exact_node_ids:
                    for nid in sorted(set(exact_node_ids)):
                        apply_label(nid, rule_id, rule_idx, impact)
                    matched = True
                    matched_count += 1
                    continue
            
            # Strategy 1: Direct ID match (#id)
            if target_str.startswith("#"):
                elem_id = target_str[1:].split(".")[0].split(":")[0]
                if elem_id in id_to_nodes:
                    for nid in id_to_nodes[elem_id]:
                        apply_label(nid, rule_id, rule_idx, impact)
                        matched = True
                    matched_count += 1
                    continue
            
            # Strategy 2: Match by class (.class)
            if target_str.startswith("."):
                class_name = target_str[1:].split(" ")[0].split(".")[0]
                if class_name in class_to_nodes:
                    for nid in class_to_nodes[class_name][:3]:  # Limit matches
                        apply_label(nid, rule_id, rule_idx, impact)
                        matched = True
                    matched_count += 1
                    continue
            
            # Strategy 3: Parse compound selectors (tag#id.class1.class2)
            # Extract tag, id, and classes from the selector
            import re
            
            # Get base tag
            tag_match = re.match(r'^([a-zA-Z0-9_-]+)', target_str)
            base_tag = tag_match.group(1).lower() if tag_match else None
            
            # Extract ID
            id_match = re.search(r'#([a-zA-Z0-9_-]+)', target_str)
            sel_id = id_match.group(1) if id_match else None
            
            # Extract classes
            classes = re.findall(r'\.([a-zA-Z0-9_-]+)', target_str)
            
            # Find candidate nodes
            candidates = set()
            if base_tag and base_tag in tag_to_nodes:
                candidates.update(tag_to_nodes[base_tag])
            elif sel_id and sel_id in id_to_nodes:
                candidates.update(id_to_nodes[sel_id])
            elif classes:
                for cls in classes:
                    if cls in class_to_nodes:
                        candidates.update(class_to_nodes[cls])
            
            # Score candidates
            best_score = 0
            best_nodes = []
            
            for nid in candidates:
                node = node_map[nid]
                score = 0
                
                if base_tag and node.tag == base_tag:
                    score += 10
                if sel_id and node.attrs.get("id") == sel_id:
                    score += 50
                
                node_classes = node.attrs.get("class", [])
                if isinstance(node_classes, str):
                    node_classes = node_classes.split()
                for cls in classes:
                    if cls in node_classes:
                        score += 15
                
                if score > best_score:
                    best_score = score
                    best_nodes = [nid]
                elif score == best_score and score > 0:
                    best_nodes.append(nid)
            
            if best_nodes and best_score >= 10:
                for nid in best_nodes[:3]:
                    apply_label(nid, rule_id, rule_idx, impact)
                matched = True
                matched_count += 1
            
            # Strategy 4: For remaining unmatched targets, try text/attribute matching
            if not matched:
                for nid, node in node_map.items():
                    if node.is_text:
                        continue
                    
                    # Check if target mentions the node's tag
                    if base_tag and node.tag == base_tag:
                        # Check if any attribute matches
                        node_id_val = node.attrs.get("id", "")
                        if node_id_val and node_id_val in target_str:
                            apply_label(nid, rule_id, rule_idx, impact)
                            matched_count += 1
                            break
        
        print(f"  Matched {matched_count} / {len(violated_nodes)} violation targets to DOM nodes")
        print(f"  Rules found: {(node_labels_multi.sum(dim=0) > 0).sum().item()} / {NUM_RULES}")
        return node_labels_binary, node_labels_multi
    
    def process_page(
        self,
        html_path: Path,
        axe_report_path: Optional[Path] = None,
        extract_visual: bool = True,
        graph_source: str = GRAPH_SOURCE_DOM,
    ) -> "ProcessedPage":
        """
        Full pipeline: parse HTML, extract text, visual, and axe features.
        """
        graph_result = build_graph(html_path, graph_source=graph_source)
        data = graph_result.data
        node_map = graph_result.node_map
        print(f"Built {graph_source} graph: {len(node_map)} nodes ({graph_result.description})")
        
        # Extract text embeddings
        print("Extracting text embeddings...")
        text_embeddings = self.extract_text_embeddings(node_map)
        data.text_embeddings = text_embeddings
        
        # Formula: x_i = [a_i || t_i], where a_i is the current attribute,
        # accessibility, and visual feature vector in data.x, and t_i is the
        # MiniLM text embedding for the same node.
        # Concatenate attribute features + text embeddings for model input
        data.x = torch.cat([data.x, text_embeddings], dim=-1)
        
        # Extract visual features
        if graph_source == GRAPH_SOURCE_RENDERED_VISUAL:
            extract_visual = True

        if extract_visual:
            print("Extracting visual features via Playwright...")
            self.extract_visual_features(html_path, node_map)
            
            # Update attribute features with visual info (recompute and replace)
            for node_id, node in node_map.items():
                new_attrs = node.get_attribute_features()
                data.x[node_id, :new_attrs.shape[0]] = new_attrs
            
            data = apply_visual_edges(data, node_map, graph_source=graph_source)
        
        # Load axe labels
        if axe_report_path and axe_report_path.exists():
            print(f"Loading axe labels from {axe_report_path}...")
            node_labels_binary, node_labels_multi = self.load_axe_labels(
                axe_report_path,
                node_map,
                graph_source=graph_source,
            )
            data.node_y = node_labels_binary
            data.node_y_multi = node_labels_multi
            data.y = torch.tensor([1 if node_labels_binary.sum() > 0 else 0], dtype=torch.long)
            data.has_ground_truth = True
            print(f"Found {node_labels_binary.sum().item()} violating elements out of {len(node_map)} nodes")
            print(f"  Multi-label: {(node_labels_multi.sum(dim=0) > 0).sum().item()} unique rules violated")
        else:
            print("No axe report provided — using dummy labels for inference only")
            data.node_y = torch.zeros(len(node_map), dtype=torch.long)
            data.node_y_multi = torch.zeros(len(node_map), NUM_RULES, dtype=torch.float)
            data.y = torch.tensor([0], dtype=torch.long)
            data.has_ground_truth = False

        data.available_rule_mask = rule_mask_for_graph_source(graph_source).unsqueeze(0)
        data.rendered_visible_mask = torch.tensor(
            [bool(getattr(node, "is_visible", False)) for node in node_map.values()],
            dtype=torch.bool,
        )
        
        return ProcessedPage(data=data, node_map=node_map, html_path=html_path)


class ProcessedPage:
    """Container for a processed page."""
    
    def __init__(self, data, node_map: Dict[int, DOMNode], html_path: Path):
        self.data = data
        self.node_map = node_map
        self.html_path = html_path
    
    def save(self, output_path: Path):
        """Save processed graph to disk."""
        torch.save({
            "data": self.data,
            "html_path": str(self.html_path),
            "num_nodes": len(self.node_map),
            "graph_source": getattr(self.data, "graph_source", GRAPH_SOURCE_DOM),
        }, output_path)
        print(f"Saved processed graph to {output_path}")
    
    @classmethod
    def load(cls, path: Path):
        """Load processed graph from disk."""
        checkpoint = torch.load(path, weights_only=False)
        data = checkpoint["data"]
        if not hasattr(data, "graph_source"):
            data.graph_source = checkpoint.get("graph_source", GRAPH_SOURCE_DOM)
        if not hasattr(data, "available_rule_mask"):
            data.available_rule_mask = rule_mask_for_graph_source(data.graph_source).unsqueeze(0)
        return cls(
            data=data,
            node_map={},  # Reconstruct if needed
            html_path=Path(checkpoint["html_path"]),
        )
