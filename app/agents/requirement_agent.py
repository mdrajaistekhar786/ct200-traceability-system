"""
Requirement Agent.

Wraps the PDF parser (structural extraction of numbered headings, text,
tables, and source information) and persists the resulting requirement
tree in MongoDB.
"""

from app.parser import parse_manual
from app.models import RequirementNode
from app.database import requirements_collection


def extract_requirements(pdf_path: str) -> list:
    """Extract and structure testable requirements from the PDF."""
    return parse_manual(pdf_path)


def save_requirement_tree(
    db,
    nodes: list,
    document_id: str,
    parent_id: str | None = None,
):
    """
    Persist the complete structured requirement tree in MongoDB.

    Each requirement is stored as an independent document while
    parent-child relationships are preserved using parent_node_id.

    The structured information extracted by the parser is preserved,
    including:
        - requirement text
        - tables
        - source page
        - source Y-position
        - content hash
    """

    for node in nodes:

        source = node.get(
            "source",
            {}
        )

        requirement = RequirementNode(
            node_id=node["id"],
            document_id=document_id,
            parent_node_id=parent_id,

            number=node.get(
                "number"
            ),

            title=node.get(
                "title"
            ),

            level=node["number"].count(".") + 1,

            text="\n".join(
                node.get(
                    "text",
                    []
                )
            ),

            tables=node.get(
                "tables",
                []
            ),

            page=source.get(
                "page"
            ),

            source_y=source.get(
                "y"
            ),

            content_hash=node.get(
                "content_hash"
            ),
        )

        requirement_data = (
            requirement.model_dump(
                exclude_none=True
            )
        )

        result = requirements_collection.insert_one(
            requirement_data
        )

        requirement.id = str(
            result.inserted_id
        )

        if node.get("children"):

            save_requirement_tree(
                db=db,
                nodes=node["children"],
                document_id=document_id,
                parent_id=requirement.node_id,
            )