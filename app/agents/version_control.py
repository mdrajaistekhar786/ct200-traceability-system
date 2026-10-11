"""
Version Detection and Document Identity.

Document identity is based on the SHA-256 hash of the actual PDF
content, not on the filename.

Behavior:

1. Exact same PDF uploaded again:
       -> detected as duplicate
       -> existing document_id is reused
       -> no new version is created

2. New PDF content:
       -> assigned the next actual version
       -> previous_document_id points to the immediately
          preceding real version

Example:

    upload A.pdf
        -> v1

    upload renamed_A.pdf
        -> duplicate of v1
        -> still v1

    upload B.pdf
        -> new content
        -> v2
        -> previous_document_id = v1 document_id
"""


import hashlib
from datetime import datetime
from pathlib import Path

from app.models import Document, RequirementNode
from app.database import (
    documents_collection,
    requirements_collection,
)


# ===========================================================================
# Document Hash
# ===========================================================================

def compute_file_hash(
    pdf_path: str,
) -> str:
    """
    Compute SHA-256 hash of the actual PDF bytes.

    The filename has no effect on the hash.

    Therefore:

        A.pdf
        renamed_document.pdf

    will produce the same hash if their actual contents are identical.
    """

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ===========================================================================
# MongoDB -> Pydantic Conversion
# ===========================================================================

def _document_from_mongo(
    raw_document: dict,
) -> Document:
    """
    Convert a MongoDB document into the application-level
    Document model.

    Older MongoDB documents may not yet contain content_hash or
    previous_document_id, so safe defaults are provided.
    """

    created_at = raw_document.get(
        "created_at"
    )

    if created_at is None:
        created_at = datetime.utcnow()

    return Document.model_validate(
        {
            "id": str(
                raw_document["_id"]
            ),

            "name": raw_document.get(
                "name",
                "",
            ),

            "version": raw_document.get(
                "version",
                "v1",
            ),

            "content_hash": raw_document.get(
                "content_hash",
                "",
            ),

            "previous_document_id": raw_document.get(
                "previous_document_id"
            ),

            "created_at": created_at,

            "is_relevant": raw_document.get(
                "is_relevant",
                True,
            ),

            "relevance_reason": raw_document.get(
                "relevance_reason"
            ),

            "version_status": raw_document.get(
                "version_status"
            ),
        }
    )


# ===========================================================================
# Find Exact Duplicate
# ===========================================================================

def find_duplicate_document(
    content_hash: str,
):
    """
    Find an existing document with exactly the same PDF content.

    Returns:
        Document
        None

    Filename is completely ignored.
    """

    raw_document = documents_collection.find_one(
        {
            "content_hash": content_hash
        }
    )

    if raw_document is None:
        return None

    return _document_from_mongo(
        raw_document
    )


# ===========================================================================
# Resolve Document Version
# ===========================================================================

def resolve_document_version(
    db,
    filename: str,
    content_hash: str,
):
    """
    Resolve the identity and version of an uploaded PDF.

    Returns:

        {
            "content_hash": ...,
            "is_duplicate": ...,
            "duplicate_document": ...,
            "version": ...,
            "previous_document": ...
        }

    Important:
        The filename is retained only as metadata.

        It is NOT used to determine whether the document is a
        duplicate or which version it represents.
    """

    # -----------------------------------------------------------------------
    # Step 1: Exact duplicate detection
    # -----------------------------------------------------------------------

    duplicate_document = find_duplicate_document(
        content_hash
    )

    if duplicate_document is not None:

        return {
            "content_hash": content_hash,
            "is_duplicate": True,
            "duplicate_document": duplicate_document,
            "version": duplicate_document.version,
            "previous_document": None,
        }

    # -----------------------------------------------------------------------
    # Step 2: Find the latest actual document version
    # -----------------------------------------------------------------------

    documents = documents_collection.find({})

    existing_documents = []

    for raw_document in documents:

        try:
            document = _document_from_mongo(
                raw_document
            )

            version_number = int(
                document.version.lower().replace(
                    "v",
                    "",
                )
            )

            existing_documents.append(
                (
                    version_number,
                    document,
                )
            )

        except (
            ValueError,
            TypeError,
        ):
            continue

    # -----------------------------------------------------------------------
    # Step 3: First unique document
    # -----------------------------------------------------------------------

    if not existing_documents:

        return {
            "content_hash": content_hash,
            "is_duplicate": False,
            "duplicate_document": None,
            "version": "v1",
            "previous_document": None,
        }

    # -----------------------------------------------------------------------
    # Step 4: New version
    # -----------------------------------------------------------------------

    existing_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    latest_version_number = (
        existing_documents[0][0]
    )

    previous_document = (
        existing_documents[0][1]
    )

    new_version = (
        f"v{latest_version_number + 1}"
    )

    return {
        "content_hash": content_hash,
        "is_duplicate": False,
        "duplicate_document": None,
        "version": new_version,
        "previous_document": previous_document,
    }


# ===========================================================================
# Mark Stale Requirements
# ===========================================================================

def mark_stale_requirements(
    db,
    document: Document,
    previous_doc: Document | None,
) -> str:
    """
    Compare current requirement nodes against the previous version.

    Returns:

        "first"
        "unchanged"
        "changed"

    A requirement is stale when its content hash differs from the
    corresponding requirement in the previous version.
    """

    # -----------------------------------------------------------------------
    # First document version
    # -----------------------------------------------------------------------

    if previous_doc is None:

        document.version_status = "first"

        documents_collection.update_one(
            {
                "_id": _get_object_id(
                    document.id
                )
            },
            {
                "$set": {
                    "version_status": "first"
                }
            },
        )

        return "first"

    # -----------------------------------------------------------------------
    # Previous version requirements
    # -----------------------------------------------------------------------

    previous_nodes_cursor = (
        requirements_collection.find(
            {
                "document_id": previous_doc.id
            }
        )
    )

    previous_nodes = {}

    for raw_node in previous_nodes_cursor:

        node = RequirementNode.model_validate(
            {
                "id": str(
                    raw_node["_id"]
                ),

                "node_id": raw_node[
                    "node_id"
                ],

                "document_id": raw_node[
                    "document_id"
                ],

                "parent_node_id": raw_node.get(
                    "parent_node_id"
                ),

                "number": raw_node.get(
                    "number"
                ),

                "level": raw_node.get(
                    "level"
                ),

                "title": raw_node.get(
                    "title"
                ),

                "text": raw_node.get(
                    "text"
                ),

                "tables": raw_node.get(
                    "tables",
                    []
                ),

                "content_hash": raw_node.get(
                    "content_hash"
                ),

                "page": raw_node.get(
                    "page"
                ),

                "source_y": raw_node.get(
                    "source_y"
                ),

                "is_stale": raw_node.get(
                    "is_stale",
                    False,
                ),
            }
        )

        previous_nodes[
            node.node_id
        ] = node

    # -----------------------------------------------------------------------
    # Current version requirements
    # -----------------------------------------------------------------------

    current_nodes_cursor = (
        requirements_collection.find(
            {
                "document_id": document.id
            }
        )
    )

    changed = False

    for raw_node in current_nodes_cursor:

        node = RequirementNode.model_validate(
            {
                "id": str(
                    raw_node["_id"]
                ),

                "node_id": raw_node[
                    "node_id"
                ],

                "document_id": raw_node[
                    "document_id"
                ],

                "parent_node_id": raw_node.get(
                    "parent_node_id"
                ),

                "number": raw_node.get(
                    "number"
                ),

                "level": raw_node.get(
                    "level"
                ),

                "title": raw_node.get(
                    "title"
                ),

                "text": raw_node.get(
                    "text"
                ),

                "tables": raw_node.get(
                    "tables",
                    []
                ),

                "content_hash": raw_node.get(
                    "content_hash"
                ),

                "page": raw_node.get(
                    "page"
                ),

                "source_y": raw_node.get(
                    "source_y"
                ),

                "is_stale": raw_node.get(
                    "is_stale",
                    False,
                ),
            }
        )

        old = previous_nodes.get(
            node.node_id
        )

        if (
            old is not None
            and old.content_hash
            != node.content_hash
        ):
            is_stale = True
            changed = True

        else:
            is_stale = False

        requirements_collection.update_one(
            {
                "_id": raw_node["_id"]
            },
            {
                "$set": {
                    "is_stale": is_stale
                }
            },
        )

    # -----------------------------------------------------------------------
    # Update current document version status
    # -----------------------------------------------------------------------

    status = (
        "changed"
        if changed
        else "unchanged"
    )

    document.version_status = status

    documents_collection.update_one(
        {
            "_id": _get_object_id(
                document.id
            )
        },
        {
            "$set": {
                "version_status": status
            }
        },
    )

    return status


# ===========================================================================
# ObjectId Utility
# ===========================================================================

def _get_object_id(
    document_id: str,
):
    """
    Convert application-level string ID back into MongoDB ObjectId.
    """

    from bson import ObjectId

    try:
        return ObjectId(
            document_id
        )

    except Exception:
        return document_id