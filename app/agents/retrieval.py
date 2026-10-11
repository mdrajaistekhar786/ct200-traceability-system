"""
Hybrid retrieval and reranking for CT-200 requirements.

Retrieval branches:

1. BM25          -> lexical / exact technical matching
2. Qdrant        -> semantic vector matching
3. Knowledge Graph -> graph-based requirement retrieval
4. RRF           -> combines retrieval rankings
5. CrossEncoder  -> reranks the fused candidates

MongoDB remains the source of truth.
"""

import re
import json

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sklearn.linear_model import LogisticRegression

from app.database import requirements_collection
from app.agents.vector_store import search_requirements
from app.agents.knowledge_graph import (
    create_knowledge_graph,
    search_graph,
)


STOPWORDS = set("""
a an the and or of to in on for with is are be shall must should this that
these those as by from at into your device unit system it its not may can
will if then than when where which who whom while such per
""".split())


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(
    RERANKER_MODEL
)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    """
    Tokenize text for BM25 retrieval.
    """

    words = re.findall(
        r"[a-zA-Z][a-zA-Z0-9\-]{2,}",
        (text or "").lower(),
    )

    return [
        word
        for word in words
        if word not in STOPWORDS
    ]


# ---------------------------------------------------------------------------
# BM25 Retrieval
# ---------------------------------------------------------------------------

def retrieve_bm25(
    document_id: str,
    current_node_id: str,
    query_text: str,
    top_k: int = 5,
) -> list:
    """
    Retrieve requirements using BM25.

    Searches:
        title + text + table content

    The selected requirement itself is excluded from BM25 retrieval.
    """

    query_tokens = _tokenize(query_text)

    if not query_tokens:
        return []

    candidates = list(
        requirements_collection.find(
            {
                "document_id": document_id,
                "node_id": {
                    "$ne": current_node_id,
                },
            }
        )
    )

    if not candidates:
        return []

    candidate_tokens = []

    for candidate in candidates:

        title = candidate.get("title", "")
        text = candidate.get("text", "")

        # Extract table content
        tables = candidate.get("tables", [])

        table_text = " ".join(
            " ".join(str(cell or "") for cell in row)
            for table in tables
            for row in table
        )

        candidate_tokens.append(
            _tokenize(
                f"{title} {text} {table_text}"
            )
        )

    bm25 = BM25Okapi(candidate_tokens)

    scores = bm25.get_scores(query_tokens)

    scored = list(
        zip(
            scores,
            candidates,
        )
    )

    scored.sort(
        key=lambda pair: pair[0],
        reverse=True,
    )

    results = []

    for score, candidate in scored[:top_k]:

        if score <= 0:
            continue

        results.append(
            {
                "node_id": candidate.get("node_id"),
                "title": candidate.get("title", ""),
                "text": (
                    candidate.get("text", "") or ""
                )[:600],
                "score": round(
                    float(score),
                    3,
                ),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Graph Retrieval
# ---------------------------------------------------------------------------

def retrieve_graph(
    document_id: str,
    current_node_id: str,
    query_text: str,
    top_k: int = 5,
) -> list:
    """
    Retrieve related requirement candidates using the Knowledge Graph.

    The selected requirement itself is never returned as an RRF candidate.

    If the selected requirement is the graph search match, its connected
    requirement nodes are used as graph candidates.
    """

    graph = create_knowledge_graph(
        document_id=document_id
    )

    if graph.number_of_nodes() == 0:
        return []

    graph_results = search_graph(
        graph=graph,
        query=query_text,
        document_id=document_id,
        max_hops=1,
    )

    results = []
    seen = set()

    for result in graph_results:

        matched_node_id = result.get(
            "node_id"
        )

        if not matched_node_id:
            continue

        # ---------------------------------------------------------------
        # Case 1:
        # The graph directly matched another requirement.
        # ---------------------------------------------------------------

        if matched_node_id != current_node_id:

            graph_node_id = (
                f"requirement:"
                f"{document_id}:"
                f"{matched_node_id}"
            )

            node_data = graph.nodes.get(
                graph_node_id,
                {},
            )

            if (
                node_data.get("type")
                == "requirement"
            ):

                if matched_node_id not in seen:

                    results.append(
                        {
                            "node_id": matched_node_id,
                            "title": node_data.get(
                                "title",
                                result.get(
                                    "title",
                                    "",
                                ),
                            ),
                            "text": (
                                node_data.get(
                                    "text",
                                    result.get(
                                        "text",
                                        "",
                                    ),
                                )
                                or ""
                            )[:600],
                            "score": result.get(
                                "score",
                                0.0,
                            ),
                            "related_nodes": result.get(
                                "related_nodes",
                                [],
                            ),
                        }
                    )

                    seen.add(
                        matched_node_id
                    )

        # ---------------------------------------------------------------
        # Case 2:
        # The selected requirement itself matched.
        #
        # Use its graph-connected requirements
        # as candidates.
        # ---------------------------------------------------------------

        if matched_node_id == current_node_id:

            related_nodes = result.get(
                "related_nodes",
                [],
            )

            for related in related_nodes:

                if related.get(
                    "type"
                ) != "requirement":
                    continue

                related_node_id = related.get(
                    "node_id"
                )

                if not related_node_id:
                    continue

                if (
                    related_node_id
                    == current_node_id
                ):
                    continue

                if related_node_id in seen:
                    continue

                results.append(
                    {
                        "node_id": related_node_id,
                        "title": related.get(
                            "title",
                            "",
                        ),
                        "text": (
                            related.get(
                                "text",
                                "",
                            ) or ""
                        )[:600],
                        "score": round(
                            float(
                                result.get(
                                    "score",
                                    0.0,
                                )
                            ),
                            4,
                        ),
                        "related_nodes": [
                            related
                        ],
                    }
                )

                seen.add(
                    related_node_id
                )

    return results[:top_k]


# ---------------------------------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------------------------------

def retrieve_hybrid(
    document_id: str,
    current_node_id: str,
    query_text: str,
    top_k: int = 5,
    retrieval_k: int = 10,
) -> list:
    """
    Combine BM25, Qdrant, and Knowledge Graph retrieval
    using Reciprocal Rank Fusion (RRF).

    Recall-protection mechanism:

        RRF Top-K
             +
        Top-3 BM25
             +
        Top-3 Qdrant
             +
        Top-3 Graph

    The selected requirement is allowed to remain in Qdrant results.

    This is important because the selected requirement can itself contain
    the exact evidence needed for the query, especially table-based evidence.
    """

    # -----------------------------------------------------------------------
    # BM25 retrieval
    # -----------------------------------------------------------------------

    bm25_results = retrieve_bm25(
        document_id=document_id,
        current_node_id=current_node_id,
        query_text=query_text,
        top_k=retrieval_k,
    )

    # -----------------------------------------------------------------------
    # Qdrant vector retrieval
    # -----------------------------------------------------------------------

    vector_results = search_requirements(
        query_text=query_text,
        document_id=document_id,
        top_k=retrieval_k,
    )

    # IMPORTANT:
    #
    # Do NOT remove current_node_id here.
    #
    # The selected requirement itself can contain the required evidence.
    #
    # Example:
    #   current_node_id = REQ-4.2
    #
    # Qdrant can correctly retrieve REQ-4.2 because its Error Codes table
    # contains the required E6/E3/etc. information.
    #
    # Therefore there is intentionally NO filtering block here.

    # -----------------------------------------------------------------------
    # Knowledge Graph retrieval
    # -----------------------------------------------------------------------

    graph_results = retrieve_graph(
        document_id=document_id,
        current_node_id=current_node_id,
        query_text=query_text,
        top_k=retrieval_k,
    )

    # -----------------------------------------------------------------------
    # RRF Fusion
    # -----------------------------------------------------------------------

    fused = {}

    rrf_k = 60

    # -----------------------------------------------------------------------
    # BM25 branch
    # -----------------------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        node_id = result["node_id"]

        if node_id not in fused:

            fused[node_id] = {
                "node_id": node_id,
                "title": result.get(
                    "title",
                    "",
                ),
                "text": result.get(
                    "text",
                    "",
                ),
                "bm25_score": result.get(
                    "score",
                    0.0,
                ),
                "vector_score": 0.0,
                "graph_score": 0.0,
                "rrf_score": 0.0,
                "graph_related_nodes": [],
            }

        fused[node_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # -----------------------------------------------------------------------
    # Qdrant branch
    # -----------------------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):

        node_id = result["node_id"]

        if node_id not in fused:

            fused[node_id] = {
                "node_id": node_id,
                "title": result.get(
                    "title",
                    "",
                ),
                "text": result.get(
                    "text",
                    "",
                ),
                "bm25_score": 0.0,
                "vector_score": result.get(
                    "score",
                    0.0,
                ),
                "graph_score": 0.0,
                "rrf_score": 0.0,
                "graph_related_nodes": [],
            }

        else:

            fused[node_id]["vector_score"] = (
                result.get(
                    "score",
                    0.0,
                )
            )

        fused[node_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # -----------------------------------------------------------------------
    # Knowledge Graph branch
    # -----------------------------------------------------------------------

    for rank, result in enumerate(
        graph_results,
        start=1,
    ):

        node_id = result["node_id"]

        if node_id not in fused:

            fused[node_id] = {
                "node_id": node_id,
                "title": result.get(
                    "title",
                    "",
                ),
                "text": result.get(
                    "text",
                    "",
                ),
                "bm25_score": 0.0,
                "vector_score": 0.0,
                "graph_score": result.get(
                    "score",
                    0.0,
                ),
                "rrf_score": 0.0,
                "graph_related_nodes": result.get(
                    "related_nodes",
                    [],
                ),
            }

        else:

            fused[node_id]["graph_score"] = (
                result.get(
                    "score",
                    0.0,
                )
            )

            fused[node_id][
                "graph_related_nodes"
            ] = result.get(
                "related_nodes",
                [],
            )

        fused[node_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # -----------------------------------------------------------------------
    # Rank fused candidates
    # -----------------------------------------------------------------------

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    # -----------------------------------------------------------------------
    # Recall Protection Safeguard — Hybrid 1
    #
    # Hybrid 1 = BM25 + Qdrant + Knowledge Graph
    #
    # Keep:
    #   1. RRF Top-K
    #   2. Top-3 BM25
    #   3. Top-3 Qdrant
    #   4. Top-3 Graph
    #
    # This protects strong candidates from any individual retriever
    # that RRF may push below Top-K.
    #
    # It does NOT modify their RRF score.
    # -----------------------------------------------------------------------

    PROTECTED_RANK_K = 3

    protected_node_ids = set()

    # Top-3 BM25
    for result in bm25_results[:PROTECTED_RANK_K]:
        protected_node_ids.add(
            result["node_id"]
        )

    # Top-3 Qdrant
    for result in vector_results[:PROTECTED_RANK_K]:
        protected_node_ids.add(
            result["node_id"]
        )

    # Top-3 Knowledge Graph
    for result in graph_results[:PROTECTED_RANK_K]:
        protected_node_ids.add(
            result["node_id"]
        )

    # Normal RRF Top-K
    selected_node_ids = {
        result["node_id"]
        for result in ranked[:top_k]
    }

    # Candidates protected by individual retrievers
    # but missing from normal RRF Top-K
    protected_candidates = [
        result
        for result in ranked
        if (
            result["node_id"]
            in protected_node_ids
        )
        and (
            result["node_id"]
            not in selected_node_ids
        )
    ]

    # Final candidate pool
    final_candidates = (
        ranked[:top_k]
        + protected_candidates
    )

    return [
        {
            "node_id": result["node_id"],
            "title": result["title"],
            "text": result["text"][:600],
            "bm25_score": round(
                result["bm25_score"],
                3,
            ),
            "vector_score": round(
                result["vector_score"],
                4,
            ),
            "graph_score": round(
                result["graph_score"],
                4,
            ),
            "rrf_score": round(
                result["rrf_score"],
                6,
            ),
            "graph_related_nodes": result.get(
                "graph_related_nodes",
                [],
            ),
        }
        for result in final_candidates
    ]


# ---------------------------------------------------------------------------
# Hybrid Retrieval: BM25 + Qdrant
# ---------------------------------------------------------------------------

def retrieve_hybrid_bm25_qdrant(
    document_id: str,
    current_node_id: str,
    query_text: str,
    top_k: int = 5,
    retrieval_k: int = 10,
) -> list:
    """
    Combine BM25 and Qdrant vector retrieval
    using Reciprocal Rank Fusion (RRF).

    This is the active hybrid strategy for the calibrated
    cost-sensitive retrieval pipeline.
    existing BM25 + Qdrant + Knowledge Graph hybrid.
    """

    # -----------------------------------------------------------------------
    # BM25 retrieval
    # -----------------------------------------------------------------------

    bm25_results = retrieve_bm25(
        document_id=document_id,
        current_node_id=current_node_id,
        query_text=query_text,
        top_k=retrieval_k,
    )

    # -----------------------------------------------------------------------
    # Qdrant vector retrieval
    # -----------------------------------------------------------------------

    vector_results = search_requirements(
        query_text=query_text,
        document_id=document_id,
        top_k=retrieval_k,
    )

    # IMPORTANT:
    #
    # Do not remove current_node_id here either.
    #
    # The selected requirement itself may contain the required evidence.

    # -----------------------------------------------------------------------
    # RRF Fusion
    # -----------------------------------------------------------------------

    fused = {}

    rrf_k = 60

    # -----------------------------------------------------------------------
    # BM25 branch
    # -----------------------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        node_id = result["node_id"]

        if node_id not in fused:

            fused[node_id] = {
                "node_id": node_id,
                "title": result.get(
                    "title",
                    "",
                ),
                "text": result.get(
                    "text",
                    "",
                ),
                "bm25_score": result.get(
                    "score",
                    0.0,
                ),
                "vector_score": 0.0,
                "rrf_score": 0.0,
            }

        fused[node_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # -----------------------------------------------------------------------
    # Qdrant branch
    # -----------------------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):

        node_id = result["node_id"]

        if node_id not in fused:

            fused[node_id] = {
                "node_id": node_id,
                "title": result.get(
                    "title",
                    "",
                ),
                "text": result.get(
                    "text",
                    "",
                ),
                "bm25_score": 0.0,
                "vector_score": result.get(
                    "score",
                    0.0,
                ),
                "rrf_score": 0.0,
            }

        else:

            fused[node_id]["vector_score"] = (
                result.get(
                    "score",
                    0.0,
                )
            )

        fused[node_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # -----------------------------------------------------------------------
    # Rank fused candidates
    # -----------------------------------------------------------------------

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    # -----------------------------------------------------------------------
    # Recall Protection Safeguard — Hybrid 2
    #
    # Hybrid 2 = BM25 + Qdrant only.
    #
    # Keep RRF Top-K plus:
    #   Top-3 BM25
    #   Top-3 Qdrant
    #
    # No Knowledge Graph candidates are used here.
    # -----------------------------------------------------------------------

    PROTECTED_RANK_K = 3

    protected_node_ids = set()

    # Top-3 BM25
    for result in bm25_results[:PROTECTED_RANK_K]:
        protected_node_ids.add(
            result["node_id"]
        )

    # Top-3 Qdrant
    for result in vector_results[:PROTECTED_RANK_K]:
        protected_node_ids.add(
            result["node_id"]
        )

    selected_node_ids = {
        result["node_id"]
        for result in ranked[:top_k]
    }

    protected_candidates = [
        result
        for result in ranked
        if (
            result["node_id"]
            in protected_node_ids
        )
        and (
            result["node_id"]
            not in selected_node_ids
        )
    ]

    final_candidates = (
        ranked[:top_k]
        + protected_candidates
    )

    # -----------------------------------------------------------------------
    # Mark recall-protected candidates before passing them downstream.
    # This metadata is intentionally preserved through CrossEncoder reranking
    # and is used by the calibrated filter as a recall safeguard.
    # -----------------------------------------------------------------------
    for result in final_candidates:
        result["recall_protected"] = (
            result["node_id"] in protected_node_ids
        )

    return [
        {
            "node_id": result["node_id"],
            "title": result["title"],
            "text": result["text"][:600],
            "bm25_score": round(
                result["bm25_score"],
                3,
            ),
            "vector_score": round(
                result["vector_score"],
                4,
            ),
            "rrf_score": round(
                result["rrf_score"],
                6,
            ),
            "recall_protected": result.get(
                "recall_protected",
                False,
            ),
        }
        for result in final_candidates
    ]


# ---------------------------------------------------------------------------
# CrossEncoder Reranking
# ---------------------------------------------------------------------------

def rerank_results(
    query_text: str,
    candidates: list,
    top_k: int = 5,
) -> list:
    """
    Rerank hybrid retrieval candidates using
    a CrossEncoder.
    """

    if not candidates:
        return []

    pairs = [
        [
            query_text,
            f"{candidate.get('title', '')}. "
            f"{candidate.get('text', '')}",
        ]
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs
    )

    reranked = []

    for candidate, score in zip(
        candidates,
        scores,
    ):

        result = dict(candidate)

        result["reranker_score"] = round(
            float(score),
            4,
        )

        reranked.append(
            result
        )

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]


# ---------------------------------------------------------------------------
# Full Retrieval + Reranking Pipeline
# ---------------------------------------------------------------------------

def retrieve_reranked(
    document_id: str,
    current_node_id: str,
    query_text: str,
    top_k: int = 5,
    retrieval_k: int = 10,
) -> list:
    """
    Full retrieval pipeline:

    BM25 + Qdrant + Knowledge Graph
            ↓
          RRF
            ↓
       Recall Protection
            ↓
       CrossEncoder
            ↓
          Top-K
    """

    hybrid_results = retrieve_hybrid(
        document_id=document_id,
        current_node_id=current_node_id,
        query_text=query_text,
        top_k=retrieval_k,
        retrieval_k=retrieval_k,
    )

    return rerank_results(
        query_text=query_text,
        candidates=hybrid_results,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Backward-Compatible Interface
# ---------------------------------------------------------------------------

def retrieve_related_requirements(
    db,
    document_id: str,
    current_node_id: str,
    query_text: str,
    top_k: int = 3,
) -> list:
    """
    Backward-compatible retrieval interface.

    Uses the full hybrid + reranking pipeline.

    The legacy validation agent expects a generic
    `score` field. The current retrieval pipeline
    produces `reranker_score`, so this interface
    exposes `reranker_score` as `score`.
    """

    results = retrieve_reranked(
        document_id=document_id,
        current_node_id=current_node_id,
        query_text=query_text,
        top_k=top_k,
        retrieval_k=max(
            top_k,
            10,
        ),
    )

    compatible_results = []

    for result in results:

        compatible_result = dict(result)

        compatible_result["score"] = result.get(
            "reranker_score",
            0.0,
        )

        compatible_results.append(
            compatible_result
        )

    return compatible_results


# ============================================================
# CALIBRATED COST-SENSITIVE FILTERING
# ============================================================

# Pipeline:
#
# Hybrid Retrieval
#       ↓
# CrossEncoder Reranking
#       ↓
# Platt Calibration
#       ↓
# P(Relevant)
# P(Irrelevant)
#       ↓
# Cost-Sensitive Filtering
#       ↓
# Surviving Evidence
#       ↓
# Second CrossEncoder Reranking
#       ↓
# Final Top-K


# ------------------------------------------------------------
# CALIBRATION MODEL
# ------------------------------------------------------------

# These values are learned offline from the Gold Standard
# calibration dataset.
#
# P(relevant) =
# sigmoid(0.212899 * score - 0.404584)

CALIBRATION_COEFFICIENT = 0.212899
CALIBRATION_INTERCEPT = -0.404584


# ------------------------------------------------------------
# COST CONFIGURATION
# ------------------------------------------------------------

FALSE_NEGATIVE_COST = 4.0
FALSE_POSITIVE_COST = 1.0


# ------------------------------------------------------------
# FILTER CONFIGURATION
# ------------------------------------------------------------

DEFAULT_TOP_K = 5


# ------------------------------------------------------------
# COST-SENSITIVE THRESHOLD
# ------------------------------------------------------------

# Remove when:
#
# P(relevant) × FN_cost
# <
# P(irrelevant) × FP_cost
#
# Therefore:
#
# P(irrelevant) >
# FN_cost / (FN_cost + FP_cost)
#
# With:
#
# FN = 4
# FP = 1
#
# threshold = 4 / 5 = 0.80

REMOVAL_THRESHOLD = (
    FALSE_NEGATIVE_COST
    / (
        FALSE_NEGATIVE_COST
        + FALSE_POSITIVE_COST
    )
)


# ------------------------------------------------------------
# CALIBRATE CROSSENCODER SCORE
# ------------------------------------------------------------

def calibrate_crossencoder_score(
    score: float,
) -> tuple:

    linear_score = (
        CALIBRATION_COEFFICIENT
        * score
        + CALIBRATION_INTERCEPT
    )

    prob_relevant = (
        1.0
        / (
            1.0
            + np.exp(-linear_score)
        )
    )

    prob_irrelevant = (
        1.0
        - prob_relevant
    )

    return (
        float(prob_relevant),
        float(prob_irrelevant),
    )


# ------------------------------------------------------------
# COST-SENSITIVE FILTER
# ------------------------------------------------------------

def cost_sensitive_filter(
    candidates: list,
) -> list:
    """
    Remove contexts whose calibrated probability
    of being irrelevant is greater than or equal
    to the cost-sensitive threshold.
    """

    filtered = []

    for candidate in candidates:

        score = float(
            candidate.get(
                "reranker_score",
                0.0,
            )
        )

        (
            prob_relevant,
            prob_irrelevant,
        ) = calibrate_crossencoder_score(
            score
        )

        result = dict(candidate)

        result["prob_relevant"] = round(
            prob_relevant,
            6,
        )

        result["prob_irrelevant"] = round(
            prob_irrelevant,
            6,
        )

        # ----------------------------------------------------
        # Expected cost of KEEP
        # ----------------------------------------------------

        result["cost_keep"] = round(
            prob_irrelevant
            * FALSE_POSITIVE_COST,
            6,
        )

        # ----------------------------------------------------
        # Expected cost of REMOVE
        # ----------------------------------------------------

        result["cost_remove"] = round(
            prob_relevant
            * FALSE_NEGATIVE_COST,
            6,
        )

        # ----------------------------------------------------
        # ----------------------------------------------------
        # Final decision with recall protection
        # ----------------------------------------------------
        # Protected candidates are strong retrieval candidates from
        # the individual retrievers. Never remove them solely because
        # calibration assigns a high probability of irrelevance.
        # Non-protected candidates still use the original
        # cost-sensitive threshold.

        if candidate.get("recall_protected", False):
            result["filter_decision"] = (
                "KEEP_RECALL_PROTECTED"
            )
        elif prob_irrelevant >= REMOVAL_THRESHOLD:
            result["filter_decision"] = "REMOVE"
        else:
            result["filter_decision"] = "KEEP"

        if result["filter_decision"] != "REMOVE":
            filtered.append(result)

    return filtered


# ------------------------------------------------------------
# SECOND RERANKING
# ------------------------------------------------------------

def rerank_surviving_evidence(
    query_text: str,
    surviving_candidates: list,
    top_k: int = DEFAULT_TOP_K,
) -> list:
    """
    Rerank only the contexts that survived
    the cost-sensitive filter.
    """

    if not surviving_candidates:
        return []

    pairs = [
        [
            query_text,
            f"{candidate.get('title', '')}. "
            f"{candidate.get('text', '')}",
        ]
        for candidate in surviving_candidates
    ]

    scores = reranker.predict(
        pairs
    )

    reranked = []

    for candidate, score in zip(
        surviving_candidates,
        scores,
    ):

        result = dict(candidate)

        # Preserve the original CrossEncoder score
        result[
            "original_reranker_score"
        ] = result.get(
            "reranker_score",
            0.0,
        )

        # New score after filtering
        result[
            "final_reranker_score"
        ] = round(
            float(score),
            4,
        )

        reranked.append(result)

    # --------------------------------------------------------
    # Rank surviving evidence again
    # --------------------------------------------------------

    reranked.sort(
        key=lambda item:
        item["final_reranker_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Final Top-K
    # --------------------------------------------------------

    return reranked[:top_k]


# ============================================================
# COMPLETE RETRIEVAL PIPELINE
# ============================================================

def retrieve_filtered_reranked(
    document_id: str,
    current_node_id: str,
    query_text: str,
    top_k: int = 5,
    retrieval_k: int = 10,
) -> list:
    """
    Complete retrieval pipeline:

    BM25 + Qdrant
            ↓
           RRF
            ↓
       Recall Protection
            ↓
       CrossEncoder
            ↓
         Calibration
            ↓
      Cost-Sensitive Filter
            ↓
      Surviving Evidence
            ↓
       CrossEncoder
            ↓
          Top-K
    """

    # --------------------------------------------------------
    # STEP 1: Hybrid Retrieval
    # --------------------------------------------------------
    # Uses Hybrid 2 only: BM25 + Qdrant. Hybrid 1 remains unchanged.

    hybrid_results = retrieve_hybrid_bm25_qdrant(
        document_id=document_id,
        current_node_id=current_node_id,
        query_text=query_text,
        top_k=retrieval_k,
        retrieval_k=retrieval_k,
    )

    if not hybrid_results:
        return []

    # --------------------------------------------------------
    # STEP 2: FIRST CROSSENCODER RERANK
    # --------------------------------------------------------

    reranked_results = rerank_results(
        query_text=query_text,
        candidates=hybrid_results,
        top_k=retrieval_k,
    )

    if not reranked_results:
        return []

    # --------------------------------------------------------
    # STEP 3: CALIBRATED COST-SENSITIVE FILTER
    # --------------------------------------------------------

    filtered_results = cost_sensitive_filter(
        reranked_results
    )

    # --------------------------------------------------------
    # STEP 4: SECOND CROSSENCODER RERANK
    # --------------------------------------------------------

    final_results = rerank_surviving_evidence(
        query_text=query_text,
        surviving_candidates=filtered_results,
        top_k=top_k,
    )

    return final_results
