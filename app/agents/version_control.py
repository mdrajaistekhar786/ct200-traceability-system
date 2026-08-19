"""
Version Detection node.

Decides whether an upload is the first document with this filename or a new
version of one already in the database, then (after the Requirement Agent
has saved the new tree) diffs it against the previous version's content
hashes to mark stale requirements.
"""

from app.models import Document, RequirementNode


def resolve_document_version(db, filename: str):
    """
    Returns (version_string, previous_document_or_none).
    """
    latest = (
        db.query(Document)
        .filter(Document.name == filename)
        .order_by(Document.id.desc())
        .first()
    )

    if latest:
        version_no = int(latest.version.replace("v", "")) + 1
        return f"v{version_no}", latest

    return "v1", None


def mark_stale_requirements(db, document: Document, previous_doc: Document | None) -> str:
    """
    Compares current requirement nodes (already saved for `document`) against
    the previous version's nodes by content_hash, and flags changed ones as
    stale.

    Returns one of: "first", "unchanged", "changed".
    """
    if previous_doc is None:
        document.version_status = "first"
        db.commit()
        return "first"

    previous_nodes = {n.node_id: n for n in previous_doc.nodes}

    current_nodes = (
        db.query(RequirementNode)
        .filter(RequirementNode.document_id == document.id)
        .all()
    )

    changed = False

    for node in current_nodes:
        old = previous_nodes.get(node.node_id)

        if old and old.content_hash != node.content_hash:
            node.is_stale = True
            changed = True
        else:
            node.is_stale = False

    status = "changed" if changed else "unchanged"
    document.version_status = status

    db.commit()

    return status
