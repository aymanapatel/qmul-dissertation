"""
graph_sources.py

Source-specific graph construction.

Current source:
- dom: builds a graph from the parsed HTML DOM.

Reserved source:
- a11y-tree: future browser accessibility-tree graph extraction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from utils import validate_graph_source

from torch_geometric.data import Data

from html_graph_builder import DOMNode, add_spatial_edges, parse_html_to_graph


GRAPH_SOURCE_DOM = "dom"
GRAPH_SOURCE_A11Y_TREE = "a11y-tree"

@dataclass(frozen=True)
class GraphBuildResult:
    data: Data
    node_map: Dict[int, DOMNode]
    description: str




def apply_visual_edges(
    data: Data,
    node_map: Dict[int, DOMNode],
    graph_source: str,
) -> Data:
    """Apply graph-source-specific visual/spatial edges after visual features exist."""
    validate_graph_source(graph_source)

    if graph_source == GRAPH_SOURCE_DOM:
        return add_spatial_edges(data, node_map)

    raise NotImplementedError(
        "Visual edges for the a11y-tree graph source are not implemented yet."
    )

def build_graph(html_path: Path, graph_source: str = GRAPH_SOURCE_DOM) -> GraphBuildResult:
    validate_graph_source(graph_source)

    if graph_source == GRAPH_SOURCE_DOM:
        return build_dom_graph(html_path)
    else:
        return build_a11y_tree_graph(html_path)

# Methods for graph creation structure
def build_dom_graph(html_path: Path) -> GraphBuildResult:
    """Build the current static DOM graph from BeautifulSoup-parsed HTML."""
    data, node_map = parse_html_to_graph(html_path)
    data.graph_source = GRAPH_SOURCE_DOM
    return GraphBuildResult(
        data=data,
        node_map=node_map,
        description="DOM graph from parsed HTML elements",
    )


def build_a11y_tree_graph(html_path: Path) -> GraphBuildResult:
    """
    Reserved integration point for future browser accessibility-tree extraction.

    Expected future behavior:
    - Use a browser accessibility snapshot rather than BeautifulSoup DOM nodes.
    - Create nodes for accessibility-tree entries such as roles/names/states.
    - Preserve enough mapping back to DOM/axe targets for labels and remediation.
    """
    raise NotImplementedError(
        "The a11y-tree graph source is reserved for future accessibility-tree "
        "extraction. Use --graph-source dom for the current DOM-based pipeline."
    )

