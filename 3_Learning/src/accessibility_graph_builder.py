"""
accessibility_graph_builder.py

Builds an accessibility-oriented PyTorch Geometric graph from static HTML.

This is a pragmatic a11y-tree graph source: it filters the parsed DOM down to
nodes that are exposed or semantically relevant to accessibility, infers roles
and accessible names, and connects each node to its nearest accessible ancestor.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from bs4 import BeautifulSoup, NavigableString
from torch_geometric.data import Data

from html_graph_builder import DOMNode, get_tag_index


LANDMARK_TAG_TO_ROLE = {
    "article": "article",
    "aside": "complementary",
    "footer": "contentinfo",
    "form": "form",
    "header": "banner",
    "main": "main",
    "nav": "navigation",
    "section": "region",
}

TAG_TO_ROLE = {
    "a": "link",
    "button": "button",
    "img": "img",
    "li": "listitem",
    "ol": "list",
    "ul": "list",
    "table": "table",
    "tbody": "rowgroup",
    "thead": "rowgroup",
    "tfoot": "rowgroup",
    "tr": "row",
    "td": "cell",
    "th": "columnheader",
    "select": "combobox",
    "textarea": "textbox",
    "summary": "button",
    "dialog": "dialog",
    **LANDMARK_TAG_TO_ROLE,
}

INPUT_TYPE_TO_ROLE = {
    "button": "button",
    "checkbox": "checkbox",
    "email": "textbox",
    "number": "spinbutton",
    "password": "textbox",
    "radio": "radio",
    "range": "slider",
    "reset": "button",
    "search": "searchbox",
    "submit": "button",
    "tel": "textbox",
    "text": "textbox",
    "url": "textbox",
}

ALWAYS_INCLUDE_ROLES = {
    "article",
    "banner",
    "button",
    "cell",
    "checkbox",
    "columnheader",
    "combobox",
    "complementary",
    "contentinfo",
    "dialog",
    "form",
    "grid",
    "gridcell",
    "heading",
    "img",
    "link",
    "list",
    "listbox",
    "listitem",
    "main",
    "navigation",
    "option",
    "radio",
    "region",
    "row",
    "rowgroup",
    "search",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "table",
    "textbox",
}

PRESENTATIONAL_ROLES = {"none", "presentation"}
IGNORED_TAGS = {"script", "style", "noscript", "template", "meta", "link"}


class AccessibilityNode(DOMNode):
    """DOMNode with inferred accessibility semantics attached."""

    def __init__(
        self,
        element,
        node_id: int,
        parent_id: Optional[int],
        role: str,
        name: str,
    ):
        super().__init__(element, node_id, parent_id)
        self.accessible_role = role
        self.accessible_name = name
        self.text_content = name or self.text_content

        # Reuse the existing attribute feature vector. Inject inferred role/name
        # without dropping original ids/classes needed for axe selector matching.
        self.attrs = dict(self.attrs)
        if role:
            self.attrs["role"] = role
        if name and not self.attrs.get("aria-label"):
            self.attrs["aria-label"] = name


def _attr(element, name: str, default: str = "") -> str:
    value = element.attrs.get(name, default) if hasattr(element, "attrs") else default
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _is_hidden(element) -> bool:
    if not hasattr(element, "attrs"):
        return False
    if element.attrs.get("hidden") is not None:
        return True
    if _attr(element, "aria-hidden").lower() == "true":
        return True
    style = _attr(element, "style").replace(" ", "").lower()
    return "display:none" in style or "visibility:hidden" in style


def _infer_role(element) -> str:
    explicit_role = _attr(element, "role").strip().lower()
    if explicit_role:
        return explicit_role.split()[0]

    tag = element.name.lower() if element.name else ""
    if tag == "img" and element.attrs.get("alt") == "":
        return "presentation"
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return "heading"
    if tag == "input":
        input_type = _attr(element, "type", "text").lower()
        return INPUT_TYPE_TO_ROLE.get(input_type, "textbox")
    if tag == "a" and not element.attrs.get("href"):
        return "generic"
    return TAG_TO_ROLE.get(tag, "generic")


def _collect_id_text(soup: BeautifulSoup) -> Dict[str, str]:
    id_to_text = {}
    for element in soup.find_all(True):
        element_id = element.attrs.get("id")
        if element_id:
            id_to_text[str(element_id)] = element.get_text(separator=" ", strip=True)
    return id_to_text


def _accessible_name(element, id_to_text: Dict[str, str]) -> str:
    labelledby = _attr(element, "aria-labelledby").strip()
    if labelledby:
        parts = [id_to_text.get(ref, "") for ref in labelledby.split()]
        name = " ".join(part for part in parts if part).strip()
        if name:
            return name

    for attr_name in ("aria-label", "alt", "title", "placeholder", "value"):
        value = _attr(element, attr_name).strip()
        if value:
            return value

    return element.get_text(separator=" ", strip=True) if hasattr(element, "get_text") else ""


def _is_focusable(element) -> bool:
    if not hasattr(element, "attrs"):
        return False
    if element.attrs.get("tabindex") is not None and _attr(element, "tabindex") != "-1":
        return True
    tag = element.name.lower() if element.name else ""
    if tag in {"button", "input", "select", "textarea", "summary"}:
        return True
    return tag == "a" and bool(element.attrs.get("href"))


def _include_in_a11y_tree(element, role: str, name: str) -> bool:
    tag = element.name.lower() if element.name else ""
    if tag in {"html", "body"}:
        return True
    if tag in IGNORED_TAGS or role in PRESENTATIONAL_ROLES:
        return False
    if _is_focusable(element):
        return True
    if role in ALWAYS_INCLUDE_ROLES:
        return True
    if name and (element.attrs.get("aria-label") or element.attrs.get("aria-labelledby")):
        return True
    return False


def parse_html_to_a11y_graph(html_path: Path) -> Tuple[Data, Dict[int, DOMNode]]:
    """Parse HTML into an accessibility-tree-style PyG Data object."""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup.find_all(IGNORED_TAGS):
        tag.decompose()

    id_to_text = _collect_id_text(soup)
    node_map: Dict[int, DOMNode] = {}
    edge_index: List[List[int]] = [[], []]
    node_id_counter = 0

    def traverse(element, accessible_parent_id: Optional[int] = None):
        nonlocal node_id_counter

        if isinstance(element, NavigableString) or element.name is None:
            return
        if _is_hidden(element):
            return

        role = _infer_role(element)
        name = _accessible_name(element, id_to_text)
        include = _include_in_a11y_tree(element, role, name)

        current_accessible_parent = accessible_parent_id
        if include:
            node = AccessibilityNode(
                element=element,
                node_id=node_id_counter,
                parent_id=accessible_parent_id,
                role=role,
                name=name,
            )
            node_map[node_id_counter] = node
            current_id = node_id_counter
            node_id_counter += 1

            if accessible_parent_id is not None:
                edge_index[0].append(accessible_parent_id)
                edge_index[1].append(current_id)
                node_map[accessible_parent_id].children.append(current_id)

            current_accessible_parent = current_id

        for child in element.children:
            traverse(child, current_accessible_parent)

    traverse(soup.html if soup.html else soup)

    if not node_map:
        root = soup.html if soup.html else soup
        node_map[0] = AccessibilityNode(root, 0, None, "document", "document")

    # Add sibling edges between adjacent accessible children.
    for node_id, node in list(node_map.items()):
        siblings = node.children
        for left, right in zip(siblings, siblings[1:]):
            edge_index[0].extend([left, right])
            edge_index[1].extend([right, left])

    tag_indices = torch.tensor([get_tag_index(node.tag) for node in node_map.values()], dtype=torch.long)
    attr_features = torch.stack([node.get_attribute_features() for node in node_map.values()])
    edge_index_tensor = (
        torch.tensor(edge_index, dtype=torch.long)
        if edge_index[0]
        else torch.empty((2, 0), dtype=torch.long)
    )

    data = Data(
        x=attr_features,
        edge_index=edge_index_tensor,
        tag_indices=tag_indices,
        node_y=torch.zeros(len(node_map), dtype=torch.long),
        y=torch.tensor([0], dtype=torch.long),
        num_nodes=len(node_map),
        text_embeddings=None,
    )

    return data, node_map
