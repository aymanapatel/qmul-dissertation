
GRAPH_SOURCE_DOM = "dom"
GRAPH_SOURCE_A11Y_TREE = "a11y-tree"

SUPPORTED_GRAPH_SOURCES = (GRAPH_SOURCE_DOM, GRAPH_SOURCE_A11Y_TREE)



def validate_graph_source(graph_source: str) -> None:
    if graph_source not in SUPPORTED_GRAPH_SOURCES:
        raise ValueError(
            f"Unsupported graph source: {graph_source}. "
            f"Expected one of: {', '.join(SUPPORTED_GRAPH_SOURCES)}"
        )
