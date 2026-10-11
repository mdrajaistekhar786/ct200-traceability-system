"""
Evidence Pack builder for CT-200.

Builds a structured evidence package from:
1. The selected requirement
2. Hybrid retrieval
3. CrossEncoder reranking

MongoDB remains the source of truth.
"""

from app.agents.retrieval import retrieve_filtered_reranked
from app.database import requirements_collection


def build_evidence_pack(
    document_id: str,
    requirement: dict,
    top_k: int = 5,
) -> dict:
    """
    Build structured evidence for a selected requirement.
    """

    requirement_id = requirement["node_id"]

    primary_evidence = {
        "node_id": requirement_id,
        "title": requirement.get(
            "title",
            "",
        ),
        "text": requirement.get(
            "text",
            [],
        ),
        "tables": requirement.get(
            "tables",
            [],
        ),
        "page": requirement.get(
            "page",
        ),
        "content_hash": requirement.get(
            "content_hash",
            "",
        ),
    }

    query_text = (
        f'{requirement.get("title", "")} '
        f'{" ".join(requirement.get("text", []))}'
    ).strip()

    related_requirements = retrieve_filtered_reranked(
        document_id=document_id,
        current_node_id=requirement_id,
        query_text=query_text,
        top_k=top_k,
        retrieval_k=max(top_k, 10),
    )

    related_evidence = []

    for result in related_requirements:

        raw_requirement = (
            requirements_collection.find_one(
                {
                    "document_id": document_id,
                    "node_id": result["node_id"],
                }
            )
        )

        related_evidence.append(
            {
                "node_id": result["node_id"],
                "title": result.get(
                    "title",
                    "",
                ),
                "text": result.get(
                    "text",
                    "",
                ),
                "page": (
                    raw_requirement.get("page")
                    if raw_requirement
                    else None
                ),
                "content_hash": (
                    raw_requirement.get(
                        "content_hash",
                        "",
                    )
                    if raw_requirement
                    else ""
                ),
                "bm25_score": result.get(
                    "bm25_score",
                    0.0,
                ),
                "vector_score": result.get(
                    "vector_score",
                    0.0,
                ),
                "rrf_score": result.get(
                    "rrf_score",
                    0.0,
                ),
                "reranker_score": result.get(
                    "reranker_score",
                    0.0,
                ),
            }
        )

    return {
        "document_id": document_id,
        "requirement_id": requirement_id,
        "primary_evidence": primary_evidence,
        "related_evidence": related_evidence,
    }