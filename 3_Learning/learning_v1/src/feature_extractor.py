"""
feature_extractor.py

Extracts node features from DOM elements:
- Text embeddings via sentence-transformers
- Visual bounding boxes via Playwright
- Axe-core violation labels
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

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


RENDERED_VISUAL_SNAPSHOT_VERSION = 1
RENDERED_VISUAL_FEATURE_VERSION = 2


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

    @staticmethod
    def _captured_visual_path(html_path: Path) -> Path:
        return html_path.with_suffix(".visual.json")

    def _load_captured_visual_features(
        self,
        html_path: Path,
        node_map: Dict[int, DOMNode],
        fallback_viewport: Dict[str, int],
    ) -> Optional[Dict[int, Dict]]:
        """Load styles captured in the live browser, when available.

        Saved HTML does not contain fetched CSS or runtime layout state. The
        crawler therefore writes a sidecar beside new snapshots. Prefer that
        ground-truth rendering context over trying to reconstruct it from an
        incomplete offline document.
        """
        snapshot_path = self._captured_visual_path(html_path)
        if not snapshot_path.exists():
            return None

        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  Ignoring unreadable visual snapshot {snapshot_path}: {exc}")
            return None

        if snapshot.get("version") != RENDERED_VISUAL_SNAPSHOT_VERSION:
            print(
                f"  Ignoring unsupported visual snapshot version at {snapshot_path}: "
                f"{snapshot.get('version')!r}"
            )
            return None

        viewport = snapshot.get("viewport", {})
        viewport_width = float(viewport.get("width") or fallback_viewport["width"])
        viewport_height = float(viewport.get("height") or fallback_viewport["height"])
        by_snapshot_id = {
            str(item.get("snapshot_node_id")): item
            for item in snapshot.get("nodes", [])
            if item.get("snapshot_node_id") is not None
        }

        visual_features: Dict[int, Dict] = {}
        matched = 0
        for node_id, node in node_map.items():
            if node.is_text or node.tag in {"__TEXT__", "__UNK__"}:
                continue
            snapshot_node_id = node.attrs.get("data-gnn-node-id")
            item = by_snapshot_id.get(str(snapshot_node_id))
            if item is None:
                visual_features[node_id] = {
                    "x": -1,
                    "y": -1,
                    "width": -1,
                    "height": -1,
                    "visible": False,
                    "visual_match_found": False,
                    "visual_source": "captured-live",
                }
                continue

            matched += 1
            visual_features[node_id] = {
                "x": float(item.get("x", -1)) / viewport_width,
                "y": float(item.get("y", -1)) / viewport_height,
                "width": float(item.get("width", -1)) / viewport_width,
                "height": float(item.get("height", -1)) / viewport_height,
                "visible": bool(item.get("visible", False)),
                "foreground_rgb": item.get("foreground_rgb", [-1, -1, -1]),
                "background_rgb": item.get("background_rgb", [-1, -1, -1]),
                "contrast_ratio": float(item.get("contrast_ratio", -1)),
                "font_size": float(item.get("font_size", -1)),
                "font_weight": float(item.get("font_weight", -1)),
                "required_contrast_ratio": float(item.get("required_contrast_ratio", -1)),
                "contrast_deficit": float(item.get("contrast_deficit", -1)),
                "has_direct_text": bool(item.get("has_direct_text", False)),
                "opacity": float(item.get("opacity", -1)),
                "text_decoration_underline": bool(item.get("text_decoration_underline", False)),
                "link_color_delta": float(item.get("link_color_delta", -1)),
                "scrollable": bool(item.get("scrollable", False)),
                "focusable": bool(item.get("focusable", False)),
                "clipped": bool(item.get("clipped", False)),
                "in_viewport": bool(item.get("in_viewport", False)),
                "visual_match_found": True,
                "visual_source": "captured-live",
            }

        if matched == 0 and by_snapshot_id:
            print(
                f"  Visual snapshot IDs did not match nodes in {html_path}; "
                "falling back to offline rendering"
            )
            return None

        print(f"  Loaded live visual snapshot: {snapshot_path} ({matched} matched nodes)")
        return visual_features

    @staticmethod
    def _apply_visual_features(
        node_map: Dict[int, DOMNode],
        visual_features: Dict[int, Dict],
        viewport: Dict[str, int],
    ) -> None:
        for node_id, feats in visual_features.items():
            if node_id not in node_map:
                continue
            node_map[node_id].bbox = {
                "x": feats["x"] * viewport["width"] if feats["x"] >= 0 else -1,
                "y": feats["y"] * viewport["height"] if feats["y"] >= 0 else -1,
                "width": feats["width"] * viewport["width"] if feats["width"] >= 0 else -1,
                "height": feats["height"] * viewport["height"] if feats["height"] >= 0 else -1,
            }
            node_map[node_id].is_visible = feats["visible"]
            node_map[node_id].visual.update(
                {key: value for key, value in feats.items() if key not in {"x", "y", "width", "height", "visible"}}
            )
    
    def extract_visual_features(
        self,
        html_path: Path,
        node_map: Dict[int, DOMNode],
        viewport: Dict[str, int] = {"width": 1280, "height": 720},
    ) -> Dict[int, Dict]:
        """
        Use Playwright to render the page and extract bounding boxes for each element.
        Maps DOM nodes to rendered elements through temporary stable node IDs.
        """
        print(f"Loading rendered visual features for {html_path}...")

        captured_features = self._load_captured_visual_features(html_path, node_map, viewport)
        if captured_features is not None:
            captured_viewport = json.loads(
                self._captured_visual_path(html_path).read_text(encoding="utf-8")
            ).get("viewport", {})
            apply_viewport = {
                "width": int(captured_viewport.get("width") or viewport["width"]),
                "height": int(captured_viewport.get("height") or viewport["height"]),
            }
            self._apply_visual_features(node_map, captured_features, apply_viewport)
            return captured_features

        instrumented_attrs = []
        for node_id, node in node_map.items():
            if node.is_text or node.tag in {"__TEXT__", "__UNK__"}:
                continue
            old_value = node.element.attrs.get("data-gnn-node-id")
            had_old_value = "data-gnn-node-id" in node.element.attrs
            node.element.attrs["data-gnn-node-id"] = str(node_id)
            instrumented_attrs.append((node, had_old_value, old_value))

        root_element = node_map[0].element if node_map else None
        rendered_html = str(root_element) if root_element is not None else html_path.read_text(encoding="utf-8")
        temp_path = None
        try:
            # Keep the instrumented snapshot beside the source document so
            # relative stylesheets, fonts, and images retain the same base URL.
            # JavaScript is disabled below, so saved-page scripts cannot mutate
            # the stable node mapping while computed CSS still loads normally.
            with tempfile.NamedTemporaryFile(
                "w",
                prefix=".gnn-visual-",
                suffix=".html",
                encoding="utf-8",
                dir=html_path.parent,
                delete=False,
            ) as handle:
                handle.write(rendered_html)
                temp_path = Path(handle.name)

            for node, had_old_value, old_value in instrumented_attrs:
                if had_old_value:
                    node.element.attrs["data-gnn-node-id"] = old_value
                else:
                    node.element.attrs.pop("data-gnn-node-id", None)

            elements_info = []
            with sync_playwright() as p:
                browser = p.chromium.launch()
                context = browser.new_context(
                    viewport=viewport,
                    java_script_enabled=False,
                )
                page = context.new_page()

                # Load enough DOM for layout extraction without waiting on external
                # assets referenced by saved pages.
                page.goto(
                    f"file://{temp_path.resolve()}",
                    wait_until="domcontentloaded",
                    timeout=10_000,
                )
                page.wait_for_load_state("domcontentloaded")

                elements_info = page.evaluate("""
                    () => {
                        const maxColorDelta = Math.sqrt(3 * 255 * 255);

                        function parseColor(value) {
                            const match = String(value || '').match(/rgba?\\(([^)]+)\\)/);
                            if (!match) return [0, 0, 0, 0];
                            const parts = match[1].split(',').map((part) => part.trim());
                            const r = Number.parseFloat(parts[0]) || 0;
                            const g = Number.parseFloat(parts[1]) || 0;
                            const b = Number.parseFloat(parts[2]) || 0;
                            const a = parts.length >= 4 ? Number.parseFloat(parts[3]) : 1;
                            return [r, g, b, Number.isFinite(a) ? a : 1];
                        }

                        function composite(fg, bg) {
                            const a = fg[3] + bg[3] * (1 - fg[3]);
                            if (a <= 0) return [0, 0, 0, 0];
                            return [
                                (fg[0] * fg[3] + bg[0] * bg[3] * (1 - fg[3])) / a,
                                (fg[1] * fg[3] + bg[1] * bg[3] * (1 - fg[3])) / a,
                                (fg[2] * fg[3] + bg[2] * bg[3] * (1 - fg[3])) / a,
                                a,
                            ];
                        }

                        function resolvedBackground(element) {
                            const chain = [];
                            let current = element;
                            while (current && current.nodeType === Node.ELEMENT_NODE) {
                                chain.push(parseColor(getComputedStyle(current).backgroundColor));
                                current = current.parentElement;
                            }
                            let color = [255, 255, 255, 1];
                            for (let i = chain.length - 1; i >= 0; i -= 1) {
                                color = composite(chain[i], color);
                            }
                            return color;
                        }

                        function luminance(rgb) {
                            const vals = rgb.slice(0, 3).map((channel) => {
                                const v = channel / 255;
                                return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
                            });
                            return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2];
                        }

                        function contrastRatio(fg, bg) {
                            const l1 = luminance(fg);
                            const l2 = luminance(bg);
                            const lighter = Math.max(l1, l2);
                            const darker = Math.min(l1, l2);
                            return (lighter + 0.05) / (darker + 0.05);
                        }

                        function colorDistance(a, b) {
                            return Math.sqrt(
                                Math.pow(a[0] - b[0], 2) +
                                Math.pow(a[1] - b[1], 2) +
                                Math.pow(a[2] - b[2], 2)
                            );
                        }

                        function isFocusable(node, style) {
                            if (node.matches('a[href], button, input, select, textarea, summary, iframe, object, embed')) {
                                return true;
                            }
                            const tabindex = node.getAttribute('tabindex');
                            return tabindex !== null && Number.parseInt(tabindex, 10) >= 0 && style.visibility !== 'hidden';
                        }

                        function isScrollable(node, style) {
                            const overflow = `${style.overflow} ${style.overflowX} ${style.overflowY}`;
                            return /(auto|scroll)/.test(overflow) && (
                                node.scrollHeight > node.clientHeight ||
                                node.scrollWidth > node.clientWidth
                            );
                        }

                        function isClipped(rect) {
                            return rect.left < 0 || rect.top < 0 || rect.right > window.innerWidth || rect.bottom > window.innerHeight;
                        }

                        return Array.from(document.querySelectorAll('[data-gnn-node-id]')).map((node) => {
                            const rect = node.getBoundingClientRect();
                            const style = window.getComputedStyle(node);
                            const bg = resolvedBackground(node);
                            const fgRaw = parseColor(style.color);
                            const fg = fgRaw[3] < 1 ? composite(fgRaw, bg) : fgRaw;
                            const parentStyle = node.parentElement ? window.getComputedStyle(node.parentElement) : style;
                            const parentFgRaw = parseColor(parentStyle.color);
                            const parentFg = parentFgRaw[3] < 1 ? composite(parentFgRaw, bg) : parentFgRaw;
                            const opacity = Number.parseFloat(style.opacity);
                            const hasBox = rect.width > 0 && rect.height > 0;
                            const inViewport = rect.bottom > 0 && rect.right > 0 && rect.left < window.innerWidth && rect.top < window.innerHeight;
                            const visible = hasBox && style.display !== 'none' && style.visibility !== 'hidden' && opacity > 0;
                            const fontWeight = Number.parseFloat(style.fontWeight);
                            const fontSize = Number.parseFloat(style.fontSize);
                            const isLargeText = fontSize >= 24 || (fontSize >= (14 * 96 / 72) && fontWeight >= 700);
                            const requiredContrastRatio = isLargeText ? 3 : 4.5;
                            const measuredContrastRatio = contrastRatio(fg, bg);
                            const contrastDeficit = Math.max(requiredContrastRatio - measuredContrastRatio, 0) / requiredContrastRatio;
                            const hasDirectText = Array.from(node.childNodes).some(
                                (child) => child.nodeType === Node.TEXT_NODE && child.textContent.trim().length > 0
                            );
                            const isLink = node.matches('a, [role="link"]');
                            const linkColorDelta = isLink ? colorDistance(fg, parentFg) : -1;

                            return {
                                nodeId: Number.parseInt(node.getAttribute('data-gnn-node-id'), 10),
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height,
                                visible,
                                foreground_rgb: fg.slice(0, 3).map((v) => Math.round(v)),
                                background_rgb: bg.slice(0, 3).map((v) => Math.round(v)),
                                contrast_ratio: measuredContrastRatio,
                                font_size: Number.isFinite(fontSize) ? fontSize : -1,
                                font_weight: Number.isFinite(fontWeight) ? fontWeight : -1,
                                required_contrast_ratio: requiredContrastRatio,
                                contrast_deficit: contrastDeficit,
                                has_direct_text: hasDirectText,
                                opacity: Number.isFinite(opacity) ? opacity : -1,
                                text_decoration_underline: style.textDecorationLine.includes('underline'),
                                link_color_delta: Number.isFinite(linkColorDelta) ? linkColorDelta : -1,
                                link_color_delta_normalized: Number.isFinite(linkColorDelta) && linkColorDelta >= 0 ? linkColorDelta / maxColorDelta : -1,
                                scrollable: isScrollable(node, style),
                                focusable: isFocusable(node, style),
                                clipped: isClipped(rect),
                                in_viewport: inViewport,
                                text: node.innerText?.substring(0, 200) || '',
                            };
                        });
                    }
                """)

                context.close()
                browser.close()
        finally:
            for node, had_old_value, old_value in instrumented_attrs:
                if had_old_value:
                    node.element.attrs["data-gnn-node-id"] = old_value
                else:
                    node.element.attrs.pop("data-gnn-node-id", None)
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        visual_features = {}

        by_node_id = {
            int(item["nodeId"]): item
            for item in elements_info
            if isinstance(item.get("nodeId"), int)
        }

        for node_id, node in node_map.items():
            if node.is_text or node.tag in {"__TEXT__", "__UNK__"}:
                continue

            best_match = by_node_id.get(node_id)
            if best_match:
                visual_features[node_id] = {
                    "x": best_match["x"] / viewport["width"],
                    "y": best_match["y"] / viewport["height"],
                    "width": best_match["width"] / viewport["width"],
                    "height": best_match["height"] / viewport["height"],
                    "visible": best_match["visible"],
                    "foreground_rgb": best_match["foreground_rgb"],
                    "background_rgb": best_match["background_rgb"],
                    "contrast_ratio": best_match["contrast_ratio"],
                    "font_size": best_match["font_size"],
                    "font_weight": best_match["font_weight"],
                    "required_contrast_ratio": best_match["required_contrast_ratio"],
                    "contrast_deficit": best_match["contrast_deficit"],
                    "has_direct_text": best_match["has_direct_text"],
                    "opacity": best_match["opacity"],
                    "text_decoration_underline": best_match["text_decoration_underline"],
                    "link_color_delta": best_match["link_color_delta"],
                    "scrollable": best_match["scrollable"],
                    "focusable": best_match["focusable"],
                    "clipped": best_match["clipped"],
                    "in_viewport": best_match["in_viewport"],
                    "visual_match_found": True,
                    "visual_source": "offline-reconstruction",
                }
            else:
                visual_features[node_id] = {
                    "x": -1,
                    "y": -1,
                    "width": -1,
                    "height": -1,
                    "visible": False,
                    "visual_match_found": False,
                    "visual_source": "offline-reconstruction",
                }

        self._apply_visual_features(node_map, visual_features, viewport)

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
                    node.visual_label_qa.append("rendered_label_on_nonvisible_or_unmatched_node")
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

            include_visual_details = graph_source == GRAPH_SOURCE_RENDERED_VISUAL
            attr_features = torch.stack(
                [
                    node.get_attribute_features(include_visual_details=include_visual_details)
                    for node in node_map.values()
                ]
            )
            data.x = torch.cat([attr_features, text_embeddings], dim=-1)
            
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
        data.rendered_visual_label_qa_mask = torch.tensor(
            [bool(getattr(node, "visual_label_qa", [])) for node in node_map.values()],
            dtype=torch.bool,
        )
        data.visual_match_found_mask = torch.tensor(
            [
                bool(getattr(node, "visual", {}).get("visual_match_found", False))
                for node in node_map.values()
            ],
            dtype=torch.bool,
        )
        data.rendered_visual_feature_version = (
            RENDERED_VISUAL_FEATURE_VERSION
            if graph_source == GRAPH_SOURCE_RENDERED_VISUAL
            else 0
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
