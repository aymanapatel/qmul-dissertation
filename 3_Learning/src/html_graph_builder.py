"""
html_graph_builder.py

Parses HTML files into PyTorch Geometric Data objects.
Nodes = HTML elements, Edges = parent-child + sibling + spatial relationships.
"""

from pathlib import Path
from math import sqrt
from typing import Dict, List, Optional, Tuple

import torch
from bs4 import BeautifulSoup, NavigableString
from torch_geometric.data import Data


# Keep these elements in the parsed document so the rendered-visual extractor
# can serialize the page with its original CSS intact, but do not turn them
# into graph nodes. In particular, decomposing <style> here made every later
# visual extraction render an unstyled page.
NON_GRAPH_TAGS = {"script", "style", "noscript"}


# Tag vocabulary: most common HTML tags + special tokens
TAG_VOCAB = {
    "__PAD__": 0,
    "__UNK__": 1,
    "a": 2, "abbr": 3, "address": 4, "area": 5, "article": 6, "aside": 7,
    "audio": 8, "b": 9, "base": 10, "bdi": 11, "bdo": 12, "blockquote": 13,
    "body": 14, "br": 15, "button": 16, "canvas": 17, "caption": 18, "cite": 19,
    "code": 20, "col": 21, "colgroup": 22, "data": 23, "datalist": 24, "dd": 25,
    "del": 26, "details": 27, "dfn": 28, "dialog": 29, "div": 30, "dl": 31,
    "dt": 32, "em": 33, "embed": 34, "fieldset": 35, "figcaption": 36, "figure": 37,
    "footer": 38, "form": 39, "h1": 40, "h2": 41, "h3": 42, "h4": 43, "h5": 44,
    "h6": 45, "head": 46, "header": 47, "hgroup": 48, "hr": 49, "html": 50,
    "i": 51, "iframe": 52, "img": 53, "input": 54, "ins": 55, "kbd": 56,
    "label": 57, "legend": 58, "li": 59, "link": 60, "main": 61, "map": 62,
    "mark": 63, "math": 64, "menu": 65, "meta": 66, "meter": 67, "nav": 68,
    "noscript": 69, "object": 70, "ol": 71, "optgroup": 72, "option": 73,
    "output": 74, "p": 75, "picture": 76, "pre": 77, "progress": 78, "q": 79,
    "rp": 80, "rt": 81, "ruby": 82, "s": 83, "samp": 84, "script": 85,
    "search": 86, "section": 87, "select": 88, "slot": 89, "small": 90,
    "source": 91, "span": 92, "strong": 93, "style": 94, "sub": 95, "summary": 96,
    "sup": 97, "svg": 98, "table": 99, "tbody": 100, "td": 101, "template": 102,
    "textarea": 103, "tfoot": 104, "th": 105, "thead": 106, "time": 107,
    "title": 108, "tr": 109, "track": 110, "u": 111, "ul": 112, "var": 113,
    "video": 114, "wbr": 115,
}

TAG_TO_INDEX = {tag: idx for tag, idx in TAG_VOCAB.items()}
NUM_TAG_EMBEDDINGS = len(TAG_VOCAB)

# Accessibility-related attributes to track
ARIA_ATTRIBUTES = [
    "aria-atomic", "aria-autocomplete", "aria-busy", "aria-checked",
    "aria-colcount", "aria-colindex", "aria-colspan", "aria-controls",
    "aria-current", "aria-describedby", "aria-details", "aria-disabled",
    "aria-dropeffect", "aria-errormessage", "aria-expanded", "aria-flowto",
    "aria-grabbed", "aria-haspopup", "aria-hidden", "aria-invalid",
    "aria-keyshortcuts", "aria-label", "aria-labelledby", "aria-level",
    "aria-live", "aria-modal", "aria-multiline", "aria-multiselectable",
    "aria-orientation", "aria-owns", "aria-placeholder", "aria-posinset",
    "aria-pressed", "aria-readonly", "aria-relevant", "aria-required",
    "aria-roledescription", "aria-rowcount", "aria-rowindex", "aria-rowspan",
    "aria-selected", "aria-setsize", "aria-sort", "aria-valuemax",
    "aria-valuemin", "aria-valuenow", "aria-valuetext", "role"
]

SEMANTIC_ROLES = [
    "article", "button", "checkbox", "complementary", "contentinfo",
    "dialog", "figure", "form", "grid", "gridcell", "heading", "img",
    "link", "list", "listbox", "listitem", "main", "navigation",
    "option", "progressbar", "radio", "region", "row", "rowgroup",
    "search", "searchbox", "separator", "slider", "spinbutton",
    "switch", "tab", "table", "tablist", "tabpanel", "textbox",
    "timer", "toolbar", "tooltip", "tree", "treeitem"
]


def get_tag_index(tag_name: str) -> int:
    return TAG_TO_INDEX.get(tag_name.lower(), TAG_TO_INDEX["__UNK__"])


class DOMNode:
    """Represents a node in the DOM tree with metadata."""
    
    def __init__(self, element, node_id: int, parent_id: Optional[int] = None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.element = element
        self.children: List[int] = []
        
        # Tag info
        if isinstance(element, NavigableString):
            self.tag = "__TEXT__"
            self.is_text = True
        else:
            self.tag = element.name.lower() if element.name else "__UNK__"
            self.is_text = False
        
        # Attributes
        if not self.is_text and hasattr(element, 'attrs'):
            self.attrs = element.attrs
            self.text_content = element.get_text(separator=" ", strip=True)
        else:
            self.attrs = {}
            self.text_content = str(element).strip() if self.is_text else ""
        
        # Visual features (populated later via Playwright)
        self.bbox = {"x": -1, "y": -1, "width": -1, "height": -1}
        self.is_visible = False
        self.visual = {
            "foreground_rgb": [-1, -1, -1],
            "background_rgb": [-1, -1, -1],
            "contrast_ratio": -1.0,
            "font_size": -1.0,
            "font_weight": -1.0,
            "required_contrast_ratio": -1.0,
            "contrast_deficit": -1.0,
            "has_direct_text": False,
            "opacity": -1.0,
            "text_decoration_underline": False,
            "link_color_delta": -1.0,
            "scrollable": False,
            "focusable": False,
            "clipped": False,
            "in_viewport": False,
            "visual_match_found": False,
        }
        self.visual_label_qa: List[str] = []
        self.dom_path = get_dom_path(element) if not self.is_text else ""
        
        # Axe violation labels (populated later)
        self.axe_violations: List[str] = []
        self.axe_impact: Optional[str] = None
    
    def get_attribute_features(self, include_visual_details: bool = False) -> torch.Tensor:
        """Extract binary attribute features."""
        features = []
        
        # Basic presence flags
        features.append(1.0 if "id" in self.attrs else 0.0)
        features.append(1.0 if "class" in self.attrs else 0.0)
        features.append(1.0 if "href" in self.attrs else 0.0)
        features.append(1.0 if "src" in self.attrs else 0.0)
        features.append(1.0 if "alt" in self.attrs else 0.0)
        features.append(1.0 if "title" in self.attrs else 0.0)
        features.append(1.0 if "type" in self.attrs else 0.0)
        features.append(1.0 if "name" in self.attrs else 0.0)
        features.append(1.0 if "value" in self.attrs else 0.0)
        features.append(1.0 if "placeholder" in self.attrs else 0.0)
        features.append(1.0 if "for" in self.attrs else 0.0)
        features.append(1.0 if "rel" in self.attrs else 0.0)
        features.append(1.0 if "target" in self.attrs else 0.0)
        
        # ARIA attributes
        for attr in ARIA_ATTRIBUTES:
            features.append(1.0 if attr in self.attrs else 0.0)
        
        # Semantic role (one-hot-ish)
        role = self.attrs.get("role", "").lower()
        for r in SEMANTIC_ROLES:
            features.append(1.0 if role == r else 0.0)
        
        # Text properties
        text_len = len(self.text_content)
        features.append(min(text_len / 1000.0, 1.0))  # Normalized text length
        features.append(1.0 if text_len > 0 else 0.0)  # Has text
        
        # Tag-specific features
        features.append(1.0 if self.tag in {"img", "picture", "svg"} else 0.0)  # Is image
        features.append(1.0 if self.tag in {"a", "button", "input", "select", "textarea"} else 0.0)  # Is interactive
        features.append(1.0 if self.tag in {"h1", "h2", "h3", "h4", "h5", "h6"} else 0.0)  # Is heading
        features.append(1.0 if self.tag in {"nav", "header", "footer", "main", "aside", "section", "article"} else 0.0)  # Is landmark
        features.append(1.0 if self.tag in {"form", "input", "label", "fieldset", "legend"} else 0.0)  # Is form-related
        
        # Visual features (if available)
        features.append(self.bbox["x"] / 1920.0 if self.bbox["x"] >= 0 else -1.0)
        features.append(self.bbox["y"] / 1080.0 if self.bbox["y"] >= 0 else -1.0)
        features.append(self.bbox["width"] / 1920.0 if self.bbox["width"] >= 0 else -1.0)
        features.append(self.bbox["height"] / 1080.0 if self.bbox["height"] >= 0 else -1.0)
        features.append(1.0 if self.is_visible else 0.0)

        if include_visual_details:
            max_color_delta = sqrt(3 * (255 ** 2))
            fg = self.visual.get("foreground_rgb", [-1, -1, -1])
            bg = self.visual.get("background_rgb", [-1, -1, -1])
            for channel in fg:
                features.append(float(channel) / 255.0 if channel >= 0 else -1.0)
            for channel in bg:
                features.append(float(channel) / 255.0 if channel >= 0 else -1.0)
            contrast_ratio = float(self.visual.get("contrast_ratio", -1.0))
            font_size = float(self.visual.get("font_size", -1.0))
            font_weight = float(self.visual.get("font_weight", -1.0))
            required_contrast_ratio = float(self.visual.get("required_contrast_ratio", -1.0))
            contrast_deficit = float(self.visual.get("contrast_deficit", -1.0))
            opacity = float(self.visual.get("opacity", -1.0))
            link_color_delta = float(self.visual.get("link_color_delta", -1.0))
            features.append(min(contrast_ratio / 21.0, 1.0) if contrast_ratio >= 0 else -1.0)
            features.append(min(font_size / 72.0, 1.0) if font_size >= 0 else -1.0)
            features.append(min(font_weight / 900.0, 1.0) if font_weight >= 0 else -1.0)
            features.append(max(0.0, min(opacity, 1.0)) if opacity >= 0 else -1.0)
            features.append(1.0 if self.visual.get("text_decoration_underline") else 0.0)
            features.append(min(link_color_delta / max_color_delta, 1.0) if link_color_delta >= 0 else -1.0)
            features.append(1.0 if self.visual.get("scrollable") else 0.0)
            features.append(1.0 if self.visual.get("focusable") else 0.0)
            features.append(1.0 if self.visual.get("clipped") else 0.0)
            features.append(1.0 if self.visual.get("in_viewport") else 0.0)
            features.append(1.0 if self.visual.get("visual_match_found") else 0.0)
            # Append new feature versions so old checkpoints can safely trim
            # the tail without shifting the meaning of their original inputs.
            features.append(
                min(required_contrast_ratio / 4.5, 1.0)
                if required_contrast_ratio >= 0
                else -1.0
            )
            features.append(
                max(0.0, min(contrast_deficit, 1.0))
                if contrast_deficit >= 0
                else -1.0
            )
            features.append(1.0 if self.visual.get("has_direct_text") else 0.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    @property
    def num_attribute_features(self) -> int:
        return len(self.get_attribute_features())


def get_dom_path(element) -> str:
    """Return a stable-ish CSS path for a BeautifulSoup element."""
    if isinstance(element, NavigableString) or not getattr(element, "name", None):
        return ""

    parts = []
    current = element
    while current is not None and getattr(current, "name", None):
        tag = current.name.lower()
        elem_id = current.attrs.get("id") if hasattr(current, "attrs") else None
        if elem_id:
            parts.append(f"{tag}#{elem_id}")
            break

        index = 1
        sibling = current.previous_sibling
        while sibling is not None:
            if getattr(sibling, "name", None) == current.name:
                index += 1
            sibling = sibling.previous_sibling
        parts.append(f"{tag}:nth-of-type({index})")
        current = current.parent

    return " > ".join(reversed(parts))


def parse_html_to_graph(
    html_path: Path,
    include_text_nodes: bool = False,
    max_nodes: Optional[int] = None,
) -> Tuple[Data, Dict[int, DOMNode]]:
    """
    Parse an HTML file into a PyTorch Geometric Data object.
    
    Args:
        html_path: Path to HTML file
        include_text_nodes: Whether to include NavigableString nodes
        max_nodes: Maximum number of nodes to include (for debugging)
    
    Returns:
        data: PyG Data object
        node_map: Mapping from node_id to DOMNode
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    
    node_map: Dict[int, DOMNode] = {}
    edge_index: List[List[int]] = [[], []]
    node_id_counter = 0

    # Formula: a webpage is represented as G = (V, E), where each parsed DOM
    # element becomes a node v_i in V and each stored relationship below becomes
    # a directed edge (v_parent, v_child) or sibling edge in E.
    
    def traverse(element, parent_id: Optional[int] = None):
        nonlocal node_id_counter
        
        if max_nodes is not None and node_id_counter >= max_nodes:
            return
        
        if isinstance(element, NavigableString):
            if include_text_nodes and str(element).strip():
                node = DOMNode(element, node_id_counter, parent_id)
                node_map[node_id_counter] = node
                if parent_id is not None:
                    edge_index[0].append(parent_id)
                    edge_index[1].append(node_id_counter)
                    node_map[parent_id].children.append(node_id_counter)
                node_id_counter += 1
            return
        
        if element.name is None:
            return

        if element.name.lower() in NON_GRAPH_TAGS:
            return
        
        # Create node
        node = DOMNode(element, node_id_counter, parent_id)
        node_map[node_id_counter] = node
        current_id = node_id_counter
        node_id_counter += 1
        
        # Parent-child edge
        if parent_id is not None:
            edge_index[0].append(parent_id)
            edge_index[1].append(current_id)
            node_map[parent_id].children.append(current_id)
        
        # Recurse on children
        for child in element.children:
            traverse(child, current_id)
    
    traverse(soup.html if soup.html else soup)
    
    # Formula: sibling edges add reading-order neighbourhoods to E, so the GNN
    # can pass messages between adjacent DOM elements as well as ancestors.
    # Add sibling edges (reading order)
    for node_id, node in node_map.items():
        if node.parent_id is not None:
            parent = node_map[node.parent_id]
            siblings = parent.children
            if len(siblings) > 1:
                idx = siblings.index(node_id)
                if idx > 0:
                    # Previous sibling
                    edge_index[0].append(siblings[idx - 1])
                    edge_index[1].append(node_id)
                if idx < len(siblings) - 1:
                    # Next sibling
                    edge_index[0].append(node_id)
                    edge_index[1].append(siblings[idx + 1])
    
    # Build PyG Data
    tag_indices = torch.tensor([get_tag_index(node.tag) for node in node_map.values()], dtype=torch.long)
    attr_features = torch.stack([node.get_attribute_features() for node in node_map.values()])
    
    # Node features = attribute features only (text embeddings added separately)
    # This avoids placeholder dimensions and makes feature dimensions explicit
    x = attr_features
    
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    
    # Node-level labels: 1 if element has axe violations
    node_labels = torch.zeros(len(node_map), dtype=torch.long)
    
    # Graph-level label: 1 if any node has violations
    graph_label = torch.tensor([0], dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index_tensor,
        tag_indices=tag_indices,
        node_y=node_labels,
        y=graph_label,
        num_nodes=len(node_map),
        text_embeddings=None,  # Will be populated by feature extractor
    )
    
    return data, node_map


def add_spatial_edges(
    data: Data,
    node_map: Dict[int, DOMNode],
    threshold: float = 0.5,
    max_neighbors: int = 8,
    max_spatial_edges: int = 80_000,
) -> Data:
    """
    Add spatial proximity edges based on rendered bounding boxes.
    Two visible nodes are connected if their bounding boxes overlap or are nearby.

    The old pairwise all-overlap version could create tens of millions of edges
    when large container boxes overlapped most descendants. GAT memory scales
    with edge count, so keep only a bounded local visual neighborhood.
    """
    if not node_map:
        return data
    
    spatial_src = []
    spatial_dst = []

    nodes = []
    for node_id, node in node_map.items():
        bbox = node.bbox
        if (
            not getattr(node, "is_visible", False)
            or bbox["width"] <= 0
            or bbox["height"] <= 0
        ):
            continue
        # Very large page containers overlap nearly everything and are already
        # connected structurally through DOM edges.
        if bbox["width"] * bbox["height"] > 1_000_000:
            continue
        center_x = bbox["x"] + bbox["width"] / 2
        center_y = bbox["y"] + bbox["height"] / 2
        nodes.append((center_y, center_x, node_id, node))

    if len(nodes) < 2:
        return data

    nodes.sort()
    neighbor_counts = {node_id: 0 for _, _, node_id, _ in nodes}
    lookahead = max(16, max_neighbors * 4)

    for i, (_, _, id1, node1) in enumerate(nodes):
        if len(spatial_src) >= max_spatial_edges:
            break
        if neighbor_counts[id1] >= max_neighbors:
            continue

        x1, y1 = node1.bbox["x"], node1.bbox["y"]
        w1, h1 = node1.bbox["width"], node1.bbox["height"]
        center1_x, center1_y = x1 + w1 / 2, y1 + h1 / 2

        for _, _, id2, node2 in nodes[i + 1:i + 1 + lookahead]:
            if len(spatial_src) >= max_spatial_edges:
                break
            if neighbor_counts[id1] >= max_neighbors:
                break
            if neighbor_counts[id2] >= max_neighbors:
                continue

            x2, y2 = node2.bbox["x"], node2.bbox["y"]
            w2, h2 = node2.bbox["width"], node2.bbox["height"]

            # Simple IoU or distance check
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

            if x_overlap > 0 and y_overlap > 0:
                spatial_src.extend([id1, id2])
                spatial_dst.extend([id2, id1])
                neighbor_counts[id1] += 1
                neighbor_counts[id2] += 1
            else:
                center2_x, center2_y = x2 + w2 / 2, y2 + h2 / 2
                dist = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2) ** 0.5
                if dist < threshold * max(w1, h1, w2, h2, 50):
                    spatial_src.extend([id1, id2])
                    spatial_dst.extend([id2, id1])
                    neighbor_counts[id1] += 1
                    neighbor_counts[id2] += 1
    
    if spatial_src:
        spatial_edges = torch.tensor([spatial_src, spatial_dst], dtype=torch.long)
        data.edge_index = torch.cat([data.edge_index, spatial_edges], dim=1)
    
    return data
