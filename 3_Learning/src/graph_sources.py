"""
graph_sources.py

Source-specific graph construction.

Current sources:
- dom: builds a graph from the parsed HTML DOM.
- a11y-tree: builds a static accessibility-tree-style graph from parsed HTML.
- rendered-visual: builds a DOM graph intended to be enriched with Playwright
  rendered visibility, geometry, and style-derived features.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from utils import validate_graph_source

from torch_geometric.data import Data

from accessibility_graph_builder import parse_html_to_a11y_graph
from html_graph_builder import DOMNode, add_spatial_edges, parse_html_to_graph


GRAPH_SOURCE_DOM = "dom"
GRAPH_SOURCE_A11Y_TREE = "a11y-tree"
GRAPH_SOURCE_RENDERED_VISUAL = "rendered-visual"

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

    if graph_source in {GRAPH_SOURCE_DOM, GRAPH_SOURCE_RENDERED_VISUAL}:
        return add_spatial_edges(data, node_map)

    if graph_source == GRAPH_SOURCE_A11Y_TREE:
        return add_spatial_edges(data, node_map)

    return data

def build_graph(html_path: Path, graph_source: str = GRAPH_SOURCE_DOM) -> GraphBuildResult:
    validate_graph_source(graph_source)

    if graph_source == GRAPH_SOURCE_DOM:
        return build_dom_graph(html_path)
    if graph_source == GRAPH_SOURCE_A11Y_TREE:
        return build_a11y_tree_graph(html_path)
    return build_rendered_visual_graph(html_path)

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
    """Build a static accessibility-tree-style graph from parsed HTML."""
    data, node_map = parse_html_to_a11y_graph(html_path)
    data.graph_source = GRAPH_SOURCE_A11Y_TREE
    return GraphBuildResult(
        data=data,
        node_map=node_map,
        description="Accessibility-tree-style graph from semantic HTML roles/names",
    )


def build_rendered_visual_graph(html_path: Path) -> GraphBuildResult:
    """Build the DOM graph that will receive rendered visual features."""
    data, node_map = parse_html_to_graph(html_path)
    data.graph_source = GRAPH_SOURCE_RENDERED_VISUAL
    return GraphBuildResult(
        data=data,
        node_map=node_map,
        description="DOM graph enriched with rendered visual features",
    )
