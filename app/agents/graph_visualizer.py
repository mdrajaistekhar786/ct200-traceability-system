"""
Interactive Knowledge Graph Visualization.

Converts the version-aware NetworkX CT-200 knowledge graph
into an interactive HTML visualization using PyVis.

The graph supports:
    - Zoom / pan
    - Node selection
    - Relationship labels
    - Node property inspection
    - Version-aware graph visualization
"""

import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network


# ============================================================================
# Visualization Configuration
# ============================================================================

DEFAULT_OUTPUT_PATH = (
    "data/ct200_knowledge_graph.html"
)


# ============================================================================
# Node Styling
# ============================================================================

NODE_COLORS = {
    "document": "#4F46E5",
    "requirement": "#2563EB",
    "constraint": "#DC2626",
    "error_code": "#EA580C",
    "table": "#16A34A",
    "test_case": "#9333EA",
}


NODE_SHAPES = {
    "document": "box",
    "requirement": "ellipse",
    "constraint": "diamond",
    "error_code": "hexagon",
    "table": "database",
    "test_case": "box",
}


# ============================================================================
# Main Visualization
# ============================================================================

def visualize_graph(
    graph: nx.MultiDiGraph,
    output_path: str = DEFAULT_OUTPUT_PATH,
):
    """
    Generate an interactive HTML visualization from a NetworkX graph.

    Parameters
    ----------
    graph:
        Version-aware CT-200 NetworkX graph.

    output_path:
        Location where the HTML visualization will be written.

    Returns
    -------
    str
        Path to the generated HTML file.
    """

    network = Network(
        height="850px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111827",
    )

    # ------------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------------

    network.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.01,
              "springLength": 150,
              "springConstant": 0.08,
              "damping": 0.4
            },
            "stabilization": {
              "enabled": true,
              "iterations": 200
            }
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.7
              }
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            },
            "font": {
              "size": 10,
              "align": "middle"
            }
          },
          "nodes": {
            "font": {
              "size": 14,
              "face": "Arial"
            },
            "borderWidth": 2
          }
        }
        """
    )

    # ------------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------------

    for node_id, data in graph.nodes(
        data=True
    ):

        node_type = data.get(
            "type",
            "unknown",
        )

        label = _build_node_label(
            data
        )

        title = _build_node_details(
            node_id,
            data,
        )

        network.add_node(
            node_id,
            label=label,
            title=title,
            color=NODE_COLORS.get(
                node_type,
                "#6B7280",
            ),
            shape=NODE_SHAPES.get(
                node_type,
                "ellipse",
            ),
            size=_node_size(
                node_type
            ),
        )

    # ------------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------------

    for source, target, data in graph.edges(
        data=True
    ):

        relationship = data.get(
            "relationship",
            "",
        )

        edge_title = _build_edge_details(
            data
        )

        network.add_edge(
            source,
            target,
            label=relationship,
            title=edge_title,
        )

    # ------------------------------------------------------------------------
    # Write HTML
    # ------------------------------------------------------------------------

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    network.write_html(
        str(path),
        open_browser=False,
    )

    return str(path)


# ============================================================================
# Node Helpers
# ============================================================================

def _build_node_label(
    data: dict,
) -> str:
    """
    Create a compact visual label for a graph node.
    """

    node_type = data.get(
        "type",
        "unknown",
    )

    if node_type == "document":

        return (
            f"{data.get('name', 'Document')}\n"
            f"{data.get('version', '')}"
        )

    if node_type == "requirement":

        return data.get(
            "node_id",
            "Requirement",
        )

    if node_type == "constraint":

        return data.get(
            "label",
            "Constraint",
        )

    if node_type == "error_code":

        return data.get(
            "code",
            "Error",
        )

    if node_type == "table":

        return data.get(
            "label",
            "Table",
        )

    if node_type == "test_case":

        return data.get(
            "test_case_id",
            "Test Case",
        )

    return data.get(
        "label",
        str(node_type),
    )


def _build_node_details(
    node_id: str,
    data: dict,
) -> str:
    """
    Build the hover/details panel for a node.

    Complex values are serialized as JSON so that tables,
    text arrays, and other structured properties remain visible.
    """

    lines = [
        f"<b>Node ID:</b> {node_id}",
        f"<b>Type:</b> {data.get('type', '')}",
    ]

    for key, value in data.items():

        if key == "type":
            continue

        if isinstance(
            value,
            (list, dict, tuple),
        ):

            value = json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
            )

        if value is None:
            value = ""

        value = str(value).replace(
            "\n",
            "<br>",
        )

        lines.append(
            f"<b>{key}:</b> {value}"
        )

    return "<br>".join(
        lines
    )


def _node_size(
    node_type: str,
) -> int:
    """
    Give important node types different visual sizes.
    """

    sizes = {
        "document": 30,
        "requirement": 24,
        "test_case": 22,
        "constraint": 18,
        "error_code": 18,
        "table": 18,
    }

    return sizes.get(
        node_type,
        18,
    )


# ============================================================================
# Edge Helpers
# ============================================================================

def _build_edge_details(
    data: dict,
) -> str:
    """
    Build hover information for an edge.
    """

    relationship = data.get(
        "relationship",
        "",
    )

    details = [
        f"<b>Relationship:</b> {relationship}"
    ]

    for key, value in data.items():

        if key == "relationship":
            continue

        if isinstance(
            value,
            (list, dict, tuple),
        ):

            value = json.dumps(
                value,
                ensure_ascii=False,
            )

        if value is None:
            value = ""

        details.append(
            f"<b>{key}:</b> {value}"
        )

    return "<br>".join(
        details
    )