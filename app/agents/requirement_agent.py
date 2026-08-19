"""
Requirement Agent.

Wraps the PDF parser (structural extraction of numbered headings, text, and
tables) and persists the resulting tree as RequirementNode rows.
"""

from app.parser import parse_manual
from app.models import RequirementNode


def extract_requirements(pdf_path: str) -> list:
    """Extract and structure testable requirements from the PDF."""
    return parse_manual(pdf_path)


def save_requirement_tree(db, nodes: list, document_id: int, parent_id: str | None = None):
    for node in nodes:
        db_node = RequirementNode(
            node_id=node["id"],
            document_id=document_id,
            parent_node_id=parent_id,
            number=node["number"],
            title=node["title"],
            level=node["number"].count(".") + 1,
            text="\n".join(node.get("text", [])),
            content_hash=node.get("content_hash"),
            page=node["source"]["page"],
        )

        db.add(db_node)
        db.commit()
        db.refresh(db_node)

        if node.get("children"):
            save_requirement_tree(
                db=db,
                nodes=node["children"],
                document_id=document_id,
                parent_id=db_node.node_id,
            )
