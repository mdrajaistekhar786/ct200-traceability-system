"""
CT-200 Requirement Knowledge Graph.

Builds a version-aware property graph from the structured requirements
stored in MongoDB.

Graph nodes:
    Document
    Requirement
    ErrorCode
    Constraint
    Table
    TestCase

Graph relationships:
    CONTAINS
    HAS_CHILD
    HAS_ERROR_CODE
    HAS_CONSTRAINT
    HAS_TABLE
    VERIFIED_BY
    VERSION_OF

MongoDB remains the source of truth.

Version-aware design:
    - Each document version has its own graph node.
    - Each requirement is scoped to its document version.
    - Previous document versions remain in the graph.
    - Corresponding requirements across versions are connected using
      VERSION_OF.
    - Requirements are never mixed between document versions.
"""

import json
import re
from pathlib import Path

import networkx as nx

from app.database import (
    documents_collection,
    requirements_collection,
    test_cases_collection,
)


# ============================================================================
# Graph Construction
# ============================================================================

def create_knowledge_graph(
    document_id: str,
) -> nx.MultiDiGraph:
    """
    Build the CT-200 knowledge graph for the selected document version
    and all previous versions of the same document.
    """

    graph = nx.MultiDiGraph()

    selected_document = documents_collection.find_one(
        {
            "_id": _get_object_id(document_id)
        }
    )

    if not selected_document:
        return graph

    selected_version = _version_number(
        selected_document.get(
            "version",
            "v1",
        )
    )

    selected_base_name = _base_filename(
        selected_document.get(
            "name",
            "",
        )
    )

    raw_documents = list(
        documents_collection.find({})
    )

    documents = []

    for document in raw_documents:

        document_base_name = _base_filename(
            document.get(
                "name",
                "",
            )
        )

        version = _version_number(
            document.get(
                "version",
                "v1",
            )
        )

        if (
            document_base_name == selected_base_name
            and version <= selected_version
        ):
            documents.append(document)

    documents.sort(
        key=lambda document: _version_number(
            document.get(
                "version",
                "v1",
            )
        )
    )

    previous_document = None

    for document in documents:

        current_document_id = str(
            document["_id"]
        )

        current_version = document.get(
            "version",
            "v1",
        )

        document_graph_id = (
            f"document:{current_document_id}"
        )

        graph.add_node(
            document_graph_id,
            type="document",
            label=document.get(
                "name",
                "CT-200 Document",
            ),
            name=document.get(
                "name",
                "",
            ),
            version=current_version,
            document_id=current_document_id,
            version_number=_version_number(
                current_version
            ),
            version_status=document.get(
                "version_status"
            ),
        )

        if previous_document:

            previous_document_id = str(
                previous_document["_id"]
            )

            graph.add_edge(
                document_graph_id,
                f"document:{previous_document_id}",
                relationship="VERSION_OF",
                from_version=current_version,
                to_version=previous_document.get(
                    "version",
                    "",
                ),
            )

        requirements = list(
            requirements_collection.find(
                {
                    "document_id": current_document_id
                }
            )
        )

        for requirement in requirements:

            _add_requirement_node(
                graph=graph,
                requirement=requirement,
                document_id=current_document_id,
            )

        if previous_document:

            _add_requirement_version_edges(
                graph=graph,
                current_document_id=current_document_id,
                previous_document_id=str(
                    previous_document["_id"]
                ),
            )

        previous_document = document

    return graph


# ============================================================================
# Requirement Nodes
# ============================================================================

def _add_requirement_node(
    graph: nx.MultiDiGraph,
    requirement: dict,
    document_id: str,
):
    """
    Add one version-scoped requirement and all its relationships.
    """

    node_id = requirement[
        "node_id"
    ]

    graph_node_id = _requirement_graph_id(
        document_id,
        node_id,
    )

    graph.add_node(
        graph_node_id,
        type="requirement",
        label=node_id,
        node_id=node_id,
        document_id=document_id,
        title=requirement.get(
            "title",
            "",
        ),
        text=requirement.get(
            "text",
            "",
        ),
        number=requirement.get(
            "number",
            "",
        ),
        level=requirement.get(
            "level"
        ),
        page=requirement.get(
            "page"
        ),
        source_y=requirement.get(
            "source_y"
        ),
        content_hash=requirement.get(
            "content_hash",
            "",
        ),
        is_stale=requirement.get(
            "is_stale",
            False,
        ),
    )

    graph.add_edge(
        f"document:{document_id}",
        graph_node_id,
        relationship="CONTAINS",
    )

    parent_id = requirement.get(
        "parent_node_id"
    )

    if parent_id:

        parent_graph_id = _requirement_graph_id(
            document_id,
            parent_id,
        )

        graph.add_edge(
            parent_graph_id,
            graph_node_id,
            relationship="HAS_CHILD",
        )

    _add_error_code_nodes(
        graph=graph,
        requirement_graph_id=graph_node_id,
        requirement=requirement,
        document_id=document_id,
    )

    _add_constraint_nodes(
        graph=graph,
        requirement_graph_id=graph_node_id,
        requirement=requirement,
        document_id=document_id,
    )

    _add_table_nodes(
        graph=graph,
        requirement_graph_id=graph_node_id,
        requirement=requirement,
        document_id=document_id,
    )

    _add_test_case_nodes(
        graph=graph,
        requirement_graph_id=graph_node_id,
        document_id=document_id,
        requirement_node_id=node_id,
    )


# ============================================================================
# Requirement Version Relationships
# ============================================================================

def _add_requirement_version_edges(
    graph: nx.MultiDiGraph,
    current_document_id: str,
    previous_document_id: str,
):
    """
    Connect corresponding requirements between two document versions.
    """

    current_requirements = list(
        requirements_collection.find(
            {
                "document_id": current_document_id
            }
        )
    )

    previous_requirements = list(
        requirements_collection.find(
            {
                "document_id": previous_document_id
            }
        )
    )

    previous_by_node_id = {
        requirement["node_id"]: requirement
        for requirement in previous_requirements
    }

    for current in current_requirements:

        node_id = current[
            "node_id"
        ]

        previous = previous_by_node_id.get(
            node_id
        )

        current_graph_id = _requirement_graph_id(
            current_document_id,
            node_id,
        )

        if previous:

            previous_graph_id = _requirement_graph_id(
                previous_document_id,
                node_id,
            )

            current_hash = current.get(
                "content_hash",
                "",
            )

            previous_hash = previous.get(
                "content_hash",
                "",
            )

            if current_hash == previous_hash:
                change_status = "unchanged"
            else:
                change_status = "changed"

            graph.add_edge(
                current_graph_id,
                previous_graph_id,
                relationship="VERSION_OF",
                change_status=change_status,
                current_content_hash=current_hash,
                previous_content_hash=previous_hash,
            )

        else:

            graph.nodes[
                current_graph_id
            ]["change_status"] = "added"


# ============================================================================
# Error Codes
# ============================================================================

def _add_error_code_nodes(
    graph: nx.MultiDiGraph,
    requirement_graph_id: str,
    requirement: dict,
    document_id: str,
):
    """
    Extract error codes such as E2 and E3.
    """

    text = requirement.get(
        "text",
        "",
    ) or ""

    title = requirement.get(
        "title",
        "",
    ) or ""

    combined_text = (
        f"{title} {text}"
    )

    error_codes = sorted(
        set(
            re.findall(
                r"\bE\d+\b",
                combined_text,
                flags=re.IGNORECASE,
            )
        )
    )

    for code in error_codes:

        code = code.upper()

        graph_node_id = (
            f"error_code:{document_id}:{code}"
        )

        graph.add_node(
            graph_node_id,
            type="error_code",
            label=code,
            code=code,
            document_id=document_id,
        )

        graph.add_edge(
            requirement_graph_id,
            graph_node_id,
            relationship="HAS_ERROR_CODE",
        )


# ============================================================================
# Constraints
# ============================================================================

def _add_constraint_nodes(
    graph: nx.MultiDiGraph,
    requirement_graph_id: str,
    requirement: dict,
    document_id: str,
):
    """
    Extract explicit numeric constraints.
    """

    text = requirement.get(
        "text",
        "",
    ) or ""

    title = requirement.get(
        "title",
        "",
    ) or ""

    combined_text = (
        f"{title} {text}"
    )

    patterns = [
        r"\b\d+(?:\.\d+)?\s*mmHg\b",
        r"\b\d+(?:\.\d+)?\s*seconds?\b",
        r"\b\d+(?:\.\d+)?\s*minutes?\b",
        r"\b\d+(?:\.\d+)?\s*%\b",
    ]

    constraints = set()

    for pattern in patterns:

        matches = re.findall(
            pattern,
            combined_text,
            flags=re.IGNORECASE,
        )

        for match in matches:

            constraints.add(
                re.sub(
                    r"\s+",
                    " ",
                    match.strip(),
                )
            )

    for constraint in sorted(
        constraints
    ):

        constraint_id = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            constraint.lower(),
        ).strip("_")

        graph_node_id = (
            f"constraint:{document_id}:{constraint_id}"
        )

        graph.add_node(
            graph_node_id,
            type="constraint",
            label=constraint,
            value=constraint,
            document_id=document_id,
        )

        graph.add_edge(
            requirement_graph_id,
            graph_node_id,
            relationship="HAS_CONSTRAINT",
        )


# ============================================================================
# Tables
# ============================================================================

def _add_table_nodes(
    graph: nx.MultiDiGraph,
    requirement_graph_id: str,
    requirement: dict,
    document_id: str,
):
    """
    Represent extracted requirement tables as version-scoped
    graph nodes.
    """

    tables = requirement.get(
        "tables",
        [],
    ) or []

    for index, table in enumerate(
        tables,
        start=1,
    ):

        requirement_node_id = requirement[
            "node_id"
        ]

        graph_node_id = (
            f"table:{document_id}:"
            f"{requirement_node_id}:{index}"
        )

        graph.add_node(
            graph_node_id,
            type="table",
            label=(
                f"Table "
                f"{requirement_node_id}-"
                f"{index}"
            ),
            table_index=index,
            document_id=document_id,
            content=table,
        )

        graph.add_edge(
            requirement_graph_id,
            graph_node_id,
            relationship="HAS_TABLE",
        )


# ============================================================================
# Test Cases
# ============================================================================

def _add_test_case_nodes(
    graph: nx.MultiDiGraph,
    requirement_graph_id: str,
    document_id: str,
    requirement_node_id: str,
):
    """
    Connect generated test cases to their version-specific
    source requirement.
    """

    test_cases = list(
        test_cases_collection.find(
            {
                "document_id": document_id,
                "requirement_node_id": (
                    requirement_node_id
                ),
            }
        )
    )

    for test_case in test_cases:

        test_case_id = test_case.get(
            "test_case_id"
        )

        if not test_case_id:
            continue

        graph_node_id = (
            f"test_case:{document_id}:"
            f"{test_case_id}"
        )

        graph.add_node(
            graph_node_id,
            type="test_case",
            label=test_case_id,
            test_case_id=test_case_id,
            document_id=document_id,
            source_version=test_case.get(
                "source_version",
                "",
            ),
            source_content_hash=test_case.get(
                "source_content_hash",
                "",
            ),
            title=test_case.get(
                "title",
                "",
            ),
            priority=test_case.get(
                "priority",
                "",
            ),
        )

        graph.add_edge(
            requirement_graph_id,
            graph_node_id,
            relationship="VERIFIED_BY",
        )


# ============================================================================
# Graph Search
# ============================================================================

def search_graph(
    graph: nx.MultiDiGraph,
    query: str,
    document_id: str,
    max_hops: int = 1,
) -> list:
    """
    Search the knowledge graph for requirements related to a query.

    Search is restricted to the selected document version.

    Returns the matching requirement and nearby graph entities such as:
        - requirements
        - constraints
        - tables
        - error codes
        - test cases
        - parent/child requirements
    """

    if not query or not query.strip():
        return []

    query_terms = set(_tokenize_graph_query(query)
        
    )

    if not query_terms:
        return []

    matching_requirements = []

    for graph_node_id, data in graph.nodes(
        data=True
    ):

        if data.get("type") != "requirement":
            continue

        if data.get("document_id") != document_id:
            continue

        searchable_text = " ".join(
            [
                str(data.get("node_id", "")),
                str(data.get("title", "")),
                str(data.get("text", "")),
            ]
        ).lower()

        searchable_terms = set(
            _tokenize_graph_query(
                searchable_text
            )
        )

        matched_terms = (
            query_terms
            & searchable_terms
        )

        if not matched_terms:
            continue

        score = (
            len(matched_terms)
            / len(query_terms)
        )

        if str(
            data.get(
                "title",
                "",
            )
        ).lower().find(
            query.strip().lower()
        ) >= 0:

            score += 1.0

        matching_requirements.append(
            {
                "graph_node_id": graph_node_id,
                "score": round(
                    score,
                    4,
                ),
                "matched_terms": sorted(
                    matched_terms
                ),
            }
        )

    matching_requirements.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    results = []

    for match in matching_requirements:

        requirement_node = graph.nodes[
            match["graph_node_id"]
        ]

        related_nodes = _collect_graph_neighbors(
            graph=graph,
            start_node=match[
                "graph_node_id"
            ],
            max_hops=max_hops,
        )

        results.append(
            {
                "node_id": requirement_node.get(
                    "node_id"
                ),
                "title": requirement_node.get(
                    "title",
                    "",
                ),
                "text": requirement_node.get(
                    "text",
                    "",
                ),
                "document_id": document_id,
                "page": requirement_node.get(
                    "page"
                ),
                "score": match["score"],
                "matched_terms": match[
                    "matched_terms"
                ],
                "related_nodes": related_nodes,
            }
        )

    return results


def _collect_graph_neighbors(
    graph: nx.MultiDiGraph,
    start_node: str,
    max_hops: int,
) -> list:
    """
    Collect nearby graph entities around a matched requirement.
    """

    if start_node not in graph:
        return []

    max_hops = max(
        1,
        int(max_hops),
    )

    visited = {
        start_node
    }

    frontier = {
        start_node
    }

    related_nodes = []

    for _ in range(max_hops):

        next_frontier = set()

        for current_node in frontier:

            neighbors = set(
                graph.successors(
                    current_node
                )
            )

            neighbors.update(
                graph.predecessors(
                    current_node
                )
            )

            for neighbor in neighbors:

                if neighbor in visited:
                    continue

                visited.add(
                    neighbor
                )

                next_frontier.add(
                    neighbor
                )

                data = graph.nodes[
                    neighbor
                ]

                related_nodes.append(
                    {
                        "graph_node_id": neighbor,
                        "type": data.get(
                            "type",
                            "unknown",
                        ),
                        "label": data.get(
                            "label",
                            "",
                        ),
                        "node_id": data.get(
                            "node_id"
                        ),
                        "title": data.get(
                            "title",
                            "",
                        ),
                        "text": data.get(
                            "text",
                            "",
                        ),
                        "document_id": data.get(
                            "document_id"
                        ),
                        "page": data.get(
                            "page"
                        ),
                        "value": data.get(
                            "value"
                        ),
                        "test_case_id": data.get(
                            "test_case_id"
                        ),
                    }
                )

        frontier = next_frontier

        if not frontier:
            break

    return related_nodes


def _tokenize_graph_query(
    text: str,
) -> list:
    """
    Normalize text for lightweight graph keyword search.
    """

    return re.findall(
        r"[a-zA-Z0-9]+",
        str(text).lower(),
    )


# ============================================================================
# Save Graph
# ============================================================================

def save_graph(
    graph: nx.MultiDiGraph,
    output_path: str = "data/ct200_knowledge_graph.graphml",
):
    """
    Save the graph in GraphML format.
    """

    graph_copy = graph.copy()

    for node_id, attributes in graph_copy.nodes(
        data=True
    ):

        for key, value in list(
            attributes.items()
        ):

            if isinstance(
                value,
                (list, dict, tuple),
            ):

                attributes[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                )

            elif value is None:

                attributes[key] = ""

    for (
        source,
        target,
        edge_key,
        attributes,
    ) in graph_copy.edges(
        data=True,
        keys=True,
    ):

        for key, value in list(
            attributes.items()
        ):

            if isinstance(
                value,
                (list, dict, tuple),
            ):

                attributes[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                )

            elif value is None:

                attributes[key] = ""

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx.write_graphml(
        graph_copy,
        path,
    )

    return str(path)


# ============================================================================
# Graph Summary
# ============================================================================

def graph_summary(
    graph: nx.MultiDiGraph,
):
    """
    Return basic graph statistics.
    """

    node_types = {}

    for _, data in graph.nodes(
        data=True
    ):

        node_type = data.get(
            "type",
            "unknown",
        )

        node_types[node_type] = (
            node_types.get(
                node_type,
                0,
            )
            + 1
        )

    relationships = {}

    for _, _, data in graph.edges(
        data=True
    ):

        relationship = data.get(
            "relationship",
            "unknown",
        )

        relationships[relationship] = (
            relationships.get(
                relationship,
                0,
            )
            + 1
        )

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": node_types,
        "relationships": relationships,
    }


# ============================================================================
# Utility Functions
# ============================================================================

def _get_object_id(
    document_id: str,
):
    """
    Convert a string into MongoDB ObjectId when possible.
    """

    from bson import ObjectId

    try:

        return ObjectId(
            document_id
        )

    except Exception:

        return document_id


def _version_number(
    version: str,
) -> int:
    """
    Convert version strings such as v1, v2, v10 into integers.
    """

    match = re.search(
        r"(\d+)",
        str(version),
    )

    if match:

        return int(
            match.group(1)
        )

    return 1


def _base_filename(
    filename: str,
) -> str:
    """
    Remove the version suffix from a filename.
    """

    stem = Path(
        filename
    ).stem

    return re.sub(
        r"_v\d+$",
        "",
        stem,
        flags=re.IGNORECASE,
    )


def _requirement_graph_id(
    document_id: str,
    requirement_node_id: str,
) -> str:
    """
    Generate a version-scoped graph ID.
    """

    return (
        f"requirement:{document_id}:"
        f"{requirement_node_id}"
    )