"""
Qdrant vector store for CT-200 requirements.

MongoDB remains the source of truth.
Qdrant stores derived vector embeddings for semantic retrieval.

Each requirement vector is uniquely identified by:
    document_id + requirement node_id

This ensures different document versions can coexist in Qdrant
without overwriting each other's vectors.
"""

import hashlib

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.database import requirements_collection


COLLECTION_NAME = "ct200_requirements"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_SIZE = 384


# ---------------------------------------------------------------------------
# Local Qdrant storage
# ---------------------------------------------------------------------------

qdrant_client = QdrantClient(
    path="./qdrant_data"
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def create_collection():
    """
    Create the Qdrant collection if it does not already exist.
    """

    collections = qdrant_client.get_collections()

    existing_names = {
        collection.name
        for collection in collections.collections
    }

    if COLLECTION_NAME not in existing_names:

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


# ---------------------------------------------------------------------------
# Requirement text
# ---------------------------------------------------------------------------

def build_requirement_text(
    requirement: dict,
) -> str:
    title = requirement.get("title", "")
    text = requirement.get("text", "")
    tables = requirement.get("tables", [])

    table_text = " ".join(
        " ".join(str(cell or "") for cell in row)
        for table in tables
        for row in table
    )

    return f"{title}. {text} {table_text}".strip()

# ---------------------------------------------------------------------------
# Point identity
# ---------------------------------------------------------------------------

def build_point_id(
    document_id: str,
    node_id: str,
) -> int:
    """
    Create a deterministic Qdrant point ID.

    The point identity is based on both:
        document_id
        requirement node_id

    This prevents the same requirement number appearing in
    different document versions from overwriting each other.
    """

    point_key = (
        f"{document_id}:"
        f"{node_id}"
    )

    return int(
        hashlib.sha256(
            point_key.encode(
                "utf-8"
            )
        ).hexdigest()[:15],
        16,
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_document_requirements(
    document_id: str,
):
    """
    Read all requirements belonging to one document version
    from MongoDB and index them in Qdrant.

    MongoDB remains the source of truth.
    Qdrant is only a derived semantic retrieval index.
    """

    create_collection()

    requirements = list(
        requirements_collection.find(
            {
                "document_id": document_id,
            }
        )
    )

    if not requirements:
        return 0

    texts = [
        build_requirement_text(
            requirement
        )
        for requirement in requirements
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
    )

    points = []

    for requirement, embedding in zip(
        requirements,
        embeddings,
    ):

        document_id_value = requirement[
            "document_id"
        ]

        node_id = requirement[
            "node_id"
        ]

        point_id = build_point_id(
            document_id=document_id_value,
            node_id=node_id,
        )

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "document_id": document_id_value,

                    "node_id": node_id,

                    "title": requirement.get(
                        "title",
                        "",
                    ),

                    "text": requirement.get(
                        "text",
                        "",
                    ),

                    "content_hash": requirement.get(
                        "content_hash",
                        "",
                    ),

                    "page": requirement.get(
                        "page"
                    ),
                },
            )
        )

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(points)


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------

def search_requirements(
    query_text: str,
    document_id: str,
    top_k: int = 5,
) -> list:
    """
    Perform semantic vector search over requirements
    belonging to a specific document version.

    The document_id filter ensures that retrieval never
    mixes requirements from different document versions.
    """

    create_collection()

    query_embedding = embedding_model.encode(
        query_text,
        normalize_embeddings=True,
    )

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id,
                    ),
                )
            ]
        ),
        limit=top_k,
    )

    return [
        {
            "node_id": point.payload.get(
                "node_id"
            ),

            "title": point.payload.get(
                "title",
                "",
            ),

            "text": (
                point.payload.get(
                    "text",
                    "",
                ) or ""
            )[:600],

            "score": round(
                point.score,
                4,
            ),

            "page": point.payload.get(
                "page"
            ),
        }
        for point in results.points
    ]