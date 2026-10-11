# retrieval_evaluation.py

"""
CT-200 Retrieval Evaluation

Evaluation design
-----------------
Retrieval baselines are evaluated using their natural candidate sets:
    1. BM25
    2. Knowledge Graph
    3. Hybrid 1 (BM25 + Qdrant + Graph) -- unchanged baseline
    4. Hybrid 2 (BM25 + Qdrant + Recall Protection) -- active retrieval

For the active Hybrid 2 pipeline, every downstream stage starts from the
EXACT SAME Hybrid-2 candidate pool for each Gold Standard case:
    5. Hybrid 2 + CrossEncoder #1
    6. Hybrid 2 + Calibration / Cost-Sensitive Filter
    7. Hybrid 2 + CrossEncoder #2 -- final Top-K

This avoids a candidate-set confound when comparing the ranking/filtering
stages. CrossEncoder #1 only changes ordering of the fixed Hybrid-2 pool;
the filter only removes candidates from that same pool; CrossEncoder #2
only reorders the surviving candidates.

Hybrid 1 is not modified by this evaluation script.
The Gold Standard is NOT modified by this script.
"""

import sys
from pathlib import Path


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from gold_standard import GOLD_STANDARD

from app.database import documents_collection

from app.agents.retrieval import (
    retrieve_bm25,
    retrieve_graph,
    retrieve_hybrid,
    retrieve_hybrid_bm25_qdrant,
    rerank_results,
    cost_sensitive_filter,
    rerank_surviving_evidence,
)


# ============================================================
# CONFIGURATION
# ============================================================

EVAL_TOP_K = 5
RETRIEVAL_K = 10


# ============================================================
# DOCUMENT
# ============================================================

def get_document_id(document_name):
    document = documents_collection.find_one(
        {"name": document_name}
    )

    if not document:
        raise ValueError(
            f"Document not found in MongoDB: {document_name}"
        )

    return str(document["_id"])


# ============================================================
# METRICS
# ============================================================

def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved = retrieved_ids[:k]
    relevant = set(relevant_ids)

    if not retrieved:
        return 0.0

    return len(set(retrieved) & relevant) / len(retrieved)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    for rank, node_id in enumerate(retrieved_ids, start=1):
        if node_id in relevant:
            return 1 / rank

    return 0.0


def evaluate_stage(results, relevant_ids, k=EVAL_TOP_K):
    retrieved_ids = [
        result.get("node_id")
        for result in results
    ]

    return {
        "recall": recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        "precision": precision_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        "mrr": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
    }


# ============================================================
# PRINT RETRIEVED RESULTS
# ============================================================

def print_results(stage_name, results, relevant_ids):
    print("\n" + "-" * 80)
    print(stage_name)
    print("-" * 80)

    print(f"Expected Relevant Nodes: {relevant_ids}")

    if not results:
        print("No results returned.")
        return

    for rank, result in enumerate(results, start=1):
        node_id = result.get("node_id")
        title = result.get("title", "")
        text = result.get("text", "")

        score = result.get(
            "final_reranker_score",
            result.get(
                "reranker_score",
                result.get(
                    "rrf_score",
                    result.get("score", None),
                ),
            ),
        )

        protected = result.get("recall_protected", False)
        decision = result.get("filter_decision", "")

        marker = " <-- RELEVANT" if node_id in relevant_ids else ""

        print(
            f"{rank}. {node_id} | "
            f"score={score} | "
            f"protected={protected} | "
            f"decision={decision} | "
            f"{title}{marker}"
        )

        print(
            f"   {text[:180].replace(chr(10), ' ')}"
        )


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(stage_name, metrics):
    print(
        f"{stage_name:<45} "
        f"Recall@5={metrics['recall']:.3f} | "
        f"Precision@5={metrics['precision']:.3f} | "
        f"MRR={metrics['mrr']:.3f}"
    )


# ============================================================
# OVERALL AVERAGE
# ============================================================

def calculate_average(all_results, stage):
    count = len(all_results)

    if count == 0:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "mrr": 0.0,
        }

    return {
        "recall": sum(
            item[stage]["recall"]
            for item in all_results
        ) / count,
        "precision": sum(
            item[stage]["precision"]
            for item in all_results
        ) / count,
        "mrr": sum(
            item[stage]["mrr"]
            for item in all_results
        ) / count,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():
    print("\n" + "=" * 100)
    print("CT-200 RETRIEVAL EVALUATION")
    print("=" * 100)

    print(
        """
Evaluation Metrics:

Recall@5
    = relevant retrieved in Top-5 / total relevant

Precision@5
    = relevant retrieved in Top-5 / retrieved Top-5

MRR
    = reciprocal rank of the first relevant result

Evaluation Methodology:

A. Retrieval comparison
   BM25, Graph, Hybrid 1 and Hybrid 2 are evaluated using their
   natural retrieval outputs. This preserves the meaning of measuring
   retrieval capability itself.

B. Active Hybrid-2 pipeline comparison
   For every Gold Standard case, the Hybrid-2 output is frozen as the
   candidate pool. CrossEncoder #1, filtering and CrossEncoder #2 then
   operate on that same pool (or its explicitly filtered survivors).

   Therefore, any ranking change from Hybrid 2 -> CE #1 -> CE #2 is
   attributable to reranking/filtering rather than a fresh retrieval call.

Active Final Pipeline:

Hybrid 2 (BM25 + Qdrant)
        |
        v
RRF + Recall Protection
        |
        v
[FROZEN CANDIDATE POOL]
        |
        v
CrossEncoder #1
        |
        v
Calibration + Cost-Sensitive Filter
        |
        v
Surviving Evidence
        |
        v
CrossEncoder #2
        |
        v
Final Top-K

Hybrid 1 is evaluated only as an unchanged baseline.
"""
    )

    all_results = []

    # ========================================================
    # EACH GOLD STANDARD SAMPLE
    # ========================================================

    for sample in GOLD_STANDARD:
        document_id = get_document_id(sample["document"])

        current_node_id = sample.get(
            "source_node_id",
            "__EVALUATION__",
        )

        query = sample["query"]
        relevant_ids = sample["relevant_node_ids"]

        print("\n\n" + "=" * 100)
        print(f"SAMPLE: {sample['id']}")
        print("=" * 100)
        print(f"Document : {sample['document']}")
        print(f"Query    : {query}")
        print(f"Expected : {relevant_ids}")

        # ====================================================
        # 1. BM25
        # ====================================================

        bm25_results = retrieve_bm25(
            document_id=document_id,
            current_node_id=current_node_id,
            query_text=query,
            top_k=RETRIEVAL_K,
        )

        bm25_metrics = evaluate_stage(
            bm25_results,
            relevant_ids,
        )

        # ====================================================
        # 2. KNOWLEDGE GRAPH
        # ====================================================

        graph_results = retrieve_graph(
            document_id=document_id,
            current_node_id=current_node_id,
            query_text=query,
            top_k=RETRIEVAL_K,
        )

        graph_metrics = evaluate_stage(
            graph_results,
            relevant_ids,
        )

        # ====================================================
        # 3. HYBRID 1
        #
        # UNCHANGED BASELINE
        # BM25 + Qdrant + Knowledge Graph
        # ====================================================

        hybrid1_results = retrieve_hybrid(
            document_id=document_id,
            current_node_id=current_node_id,
            query_text=query,
            top_k=RETRIEVAL_K,
            retrieval_k=RETRIEVAL_K,
        )

        hybrid1_metrics = evaluate_stage(
            hybrid1_results,
            relevant_ids,
        )

        # ====================================================
        # 4. HYBRID 2
        #
        # ACTIVE RETRIEVAL STRATEGY
        # BM25 + Qdrant + RRF + Recall Protection
        # ====================================================

        hybrid2_results = retrieve_hybrid_bm25_qdrant(
            document_id=document_id,
            current_node_id=current_node_id,
            query_text=query,
            top_k=RETRIEVAL_K,
            retrieval_k=RETRIEVAL_K,
        )

        # IMPORTANT:
        # Freeze the Hybrid-2 output for ALL downstream stages.
        # No downstream stage performs another retrieval call.
        fixed_hybrid2_pool = [dict(result) for result in hybrid2_results]

        hybrid2_metrics = evaluate_stage(
            fixed_hybrid2_pool,
            relevant_ids,
        )

        # ====================================================
        # 5. CROSSENCODER #1
        #
        # SAME FIXED HYBRID-2 CANDIDATE POOL
        # Only the ordering is changed.
        # ====================================================

        reranked_results = rerank_results(
            query_text=query,
            candidates=[dict(result) for result in fixed_hybrid2_pool],
            top_k=len(fixed_hybrid2_pool),
        )

        reranked_metrics = evaluate_stage(
            reranked_results,
            relevant_ids,
        )

        # ====================================================
        # 6. CALIBRATION + COST-SENSITIVE FILTER
        #
        # SAME CE #1 RESULT SET.
        # Filtering may remove nodes, but it does NOT retrieve
        # any new nodes.
        # ====================================================

        filtered_results = cost_sensitive_filter(
            [dict(result) for result in reranked_results]
        )

        filtered_metrics = evaluate_stage(
            filtered_results,
            relevant_ids,
        )

        # ====================================================
        # 7. CROSSENCODER #2
        #
        # SAME FILTER SURVIVORS.
        # Only final ordering/top-k selection is performed.
        # ====================================================

        final_results = rerank_surviving_evidence(
            query_text=query,
            surviving_candidates=[dict(result) for result in filtered_results],
            top_k=EVAL_TOP_K,
        )

        final_metrics = evaluate_stage(
            final_results,
            relevant_ids,
        )

        # ====================================================
        # PRINT RETRIEVAL RESULTS
        # ====================================================

        print_results(
            "BM25 RESULTS",
            bm25_results,
            relevant_ids,
        )

        print_results(
            "GRAPH RESULTS",
            graph_results,
            relevant_ids,
        )

        print_results(
            "HYBRID 1 RESULTS -- UNCHANGED",
            hybrid1_results,
            relevant_ids,
        )

        print_results(
            "HYBRID 2 RESULTS -- FROZEN CANDIDATE POOL",
            fixed_hybrid2_pool,
            relevant_ids,
        )

        print_results(
            "CROSSENCODER #1 -- SAME HYBRID 2 POOL",
            reranked_results,
            relevant_ids,
        )

        print_results(
            "FILTERED SURVIVING EVIDENCE",
            filtered_results,
            relevant_ids,
        )

        print_results(
            "FINAL RESULTS -- CROSSENCODER #2",
            final_results,
            relevant_ids,
        )

        # ====================================================
        # SAMPLE METRICS
        # ====================================================

        print("\n" + "-" * 100)
        print(f"METRICS FOR {sample['id']}")
        print("-" * 100)

        print_metrics("BM25", bm25_metrics)
        print_metrics("Graph", graph_metrics)
        print_metrics("Hybrid 1", hybrid1_metrics)
        print_metrics("Hybrid 2", hybrid2_metrics)
        print_metrics("Hybrid 2 + CE #1", reranked_metrics)
        print_metrics("Hybrid 2 + Calibration + Filter", filtered_metrics)
        print_metrics("Final Hybrid 2 + CE #2", final_metrics)

        print(
            f"Fixed Hybrid-2 candidate pool size: "
            f"{len(fixed_hybrid2_pool)}"
        )

        # ====================================================
        # STORE RESULTS
        # ====================================================

        all_results.append({
            "id": sample["id"],
            "bm25": bm25_metrics,
            "graph": graph_metrics,
            "hybrid1": hybrid1_metrics,
            "hybrid2": hybrid2_metrics,
            "reranked": reranked_metrics,
            "filtered": filtered_metrics,
            "final": final_metrics,
            "candidate_pool_size": len(fixed_hybrid2_pool),
        })

    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    print("\n\n" + "=" * 100)
    print("OVERALL RETRIEVAL PERFORMANCE")
    print("=" * 100)
    print(f"Total Gold Standard Cases: {len(all_results)}")

    print(
        "\n"
        f"{'Pipeline / Method':<45}"
        f"{'Recall@5':<12}"
        f"{'Precision@5':<15}"
        f"{'MRR':<10}"
    )
    print("-" * 90)

    for stage, name in [
        ("bm25", "BM25"),
        ("graph", "Graph"),
        ("hybrid1", "Hybrid 1"),
        ("hybrid2", "Hybrid 2 + Recall Protection"),
        ("reranked", "Hybrid 2 + CE #1"),
        ("filtered", "H2 + Calibration + Cost-Sensitive Filter"),
        ("final", "H2 + Recall Protection + Calibration + Filter + CE #2"),
    ]:
        avg = calculate_average(all_results, stage)

        print(
            f"{name:<45}"
            f"{avg['recall']:<12.3f}"
            f"{avg['precision']:<15.3f}"
            f"{avg['mrr']:<10.3f}"
        )

    print("\n" + "=" * 100)
    print("IMPORTANT EVALUATION NOTE")
    print("=" * 100)
    print(
        "BM25 / Graph / Hybrid 1 / Hybrid 2 measure retrieval capability "
        "using their natural outputs."
    )
    print(
        "For the active Hybrid-2 pipeline, CE #1, filtering and CE #2 "
        "operate on the same Hybrid-2 candidate pool for each Gold Standard case."
    )
    print(
        "Therefore, downstream ranking changes are not caused by a fresh "
        "retrieval candidate set."
    )
    print("\nRETRIEVAL EVALUATION COMPLETE")
    print("=" * 100)

    return all_results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_evaluation()
