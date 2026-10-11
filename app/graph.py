"""
LangGraph Orchestrator.

Two workflows:

1. Document Ingestion:
   Document Identity -> Document Gate -> Version Detection
   -> Requirement Agent -> Derived Indexes -> END

2. Selected Requirement QA:
   Evidence Pack -> Test Case Agent -> Validation Agent
   -> Retry if needed -> Traceability Agent -> END
"""

import time
from datetime import datetime

from langgraph.graph import StateGraph, END

from app.graph_state import QAGraphState

from app.agents.document_gate import check_document_relevance

from app.agents.version_control import (
    compute_file_hash,
    resolve_document_version,
    mark_stale_requirements,
)

from app.agents.requirement_agent import (
    extract_requirements,
    save_requirement_tree,
)

from app.agents.test_case_agent import (
    generate_test_cases,
    save_test_cases,
    get_cached_test_cases,
)

from app.agents.evidence_pack import (
    build_evidence_pack,
)

from app.agents.validation_agent import validate_node

from app.agents.traceability_agent import build_traceability

from app.agents.vector_store import (
    index_document_requirements,
)

from app.agents.knowledge_graph import (
    create_knowledge_graph,
    save_graph,
)

from app.models import Document

from app.database import (
    documents_collection,
    requirements_collection,
)

from app.database import (
    documents_collection,
    requirements_collection,
    test_cases_collection,
)


DEFAULT_MAX_RETRIES = 2


# ===========================================================================
# Timing helpers
# ===========================================================================

def log_start(name: str):
    print(f"\n[START] {name}")
    return time.perf_counter()


def log_end(name: str, start_time: float):
    elapsed = time.perf_counter() - start_time
    print(f"[DONE]  {name}: {elapsed:.2f}s")


# ===========================================================================
# DOCUMENT INGESTION NODES
# ===========================================================================

def document_identity_node(
    state: QAGraphState,
) -> QAGraphState:
    """
    Establish document identity before any expensive PDF processing.

    SHA-256 of the actual PDF bytes is used as the exact document
    identity.

    If the hash already exists in MongoDB:

        - mark upload as duplicate
        - reuse existing document
        - skip parsing
        - skip CRAG
        - skip requirement extraction
        - skip version creation
        - skip index rebuilding

    If the hash is new:

        - continue to Document Gate
    """

    start = log_start(
        "Document Identity"
    )

    pdf_path = state["pdf_path"]

    content_hash = compute_file_hash(
        pdf_path
    )

    state["content_hash"] = content_hash

    print(
        f"[INFO] Content hash: "
        f"{content_hash}"
    )

    resolution = resolve_document_version(
        state["db"],
        state["filename"],
        content_hash,
    )

    state["is_duplicate"] = bool(
        resolution["is_duplicate"]
    )

    if resolution["is_duplicate"]:

        existing_document = (
            resolution["duplicate_document"]
        )

        state["document"] = existing_document

        state["previous_document"] = None

        state["previous_document_id"] = None

        state["version_status"] = "duplicate"

        state["document_id"] = (
            existing_document.id
        )

        state["is_relevant"] = (
            existing_document.is_relevant
        )

        state["relevance_reason"] = (
            existing_document.relevance_reason
            or ""
        )

        print(
            "[INFO] Duplicate document detected."
        )

        print(
            f"[INFO] Existing document ID: "
            f"{existing_document.id}"
        )

        print(
            f"[INFO] Existing version: "
            f"{existing_document.version}"
        )

        log_end(
            "Document Identity",
            start,
        )

        return state

    state["is_duplicate"] = False

    state["previous_document"] = (
        resolution["previous_document"]
    )

    previous_document = (
        resolution["previous_document"]
    )

    if previous_document:

        state["previous_document_id"] = (
            previous_document.id
        )

        print(
            f"[INFO] Previous version: "
            f"{previous_document.version}"
        )

    else:

        state["previous_document_id"] = None

        print(
            "[INFO] No previous document version."
        )

    print(
        f"[INFO] New document version will be: "
        f"{resolution['version']}"
    )

    log_end(
        "Document Identity",
        start,
    )

    return state


# ===========================================================================
# Document Gate
# ===========================================================================

def document_gate_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Document Gate"
    )

    if state.get("is_duplicate"):

        print(
            "[INFO] Duplicate upload. "
            "Skipping Document Gate."
        )

        log_end(
            "Document Gate",
            start,
        )

        return state

    tree = extract_requirements(
        state["pdf_path"]
    )

    sample_text = (
        " ".join(
            " ".join(
                n.get("text", [])
            )
            for n in tree[:3]
        )
        if tree
        else ""
    )

    gate = check_document_relevance(
        tree,
        state["filename"],
        sample_text,
    )

    state["tree"] = tree

    state["is_relevant"] = bool(
        gate.get("is_relevant")
    )

    state["relevance_reason"] = gate.get(
        "reason",
        "",
    )

    print(
        f"[INFO] Top-level requirements extracted: "
        f"{len(tree)}"
    )

    print(
        f"[INFO] Document relevant: "
        f"{state['is_relevant']}"
    )

    log_end(
        "Document Gate",
        start,
    )

    return state


# ===========================================================================
# Reject
# ===========================================================================

def reject_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Reject"
    )

    state["rejected"] = True

    state["error"] = (
        f"Document rejected by Document Gate: "
        f"{state.get('relevance_reason')}"
    )

    log_end(
        "Reject",
        start,
    )

    return state


# ===========================================================================
# Version Detection
# ===========================================================================

def version_detection_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Version Detection"
    )

    if state.get("is_duplicate"):

        print(
            "[INFO] Duplicate document. "
            "Skipping Version Detection."
        )

        log_end(
            "Version Detection",
            start,
        )

        return state

    db = state["db"]

    content_hash = state[
        "content_hash"
    ]

    resolution = resolve_document_version(
        db,
        state["filename"],
        content_hash,
    )

    version = resolution[
        "version"
    ]

    previous_doc = resolution[
        "previous_document"
    ]

    if resolution["is_duplicate"]:

        existing_document = (
            resolution["duplicate_document"]
        )

        state["is_duplicate"] = True

        state["document"] = (
            existing_document
        )

        state["document_id"] = (
            existing_document.id
        )

        state["version_status"] = (
            "duplicate"
        )

        print(
            "[INFO] Duplicate detected during "
            "Version Detection safety check."
        )

        log_end(
            "Version Detection",
            start,
        )

        return state

    document_data = {

        "name": state["filename"],

        "version": version,

        "content_hash": content_hash,

        "previous_document_id": (
            previous_doc.id
            if previous_doc
            else None
        ),

        "created_at": datetime.utcnow(),

        "is_relevant": True,

        "relevance_reason": state.get(
            "relevance_reason",
            "",
        ),

        "version_status": None,
    }

    result = documents_collection.insert_one(
        document_data
    )

    document_id = str(
        result.inserted_id
    )

    state["document_id"] = document_id

    document = Document.model_validate(
        {
            "id": document_id,

            "name": document_data[
                "name"
            ],

            "version": document_data[
                "version"
            ],

            "content_hash": document_data[
                "content_hash"
            ],

            "previous_document_id": (
                document_data[
                    "previous_document_id"
                ]
            ),

            "created_at": document_data[
                "created_at"
            ],

            "is_relevant": document_data[
                "is_relevant"
            ],

            "relevance_reason": (
                document_data[
                    "relevance_reason"
                ]
            ),

            "version_status": (
                document_data[
                    "version_status"
                ]
            ),
        }
    )

    state["document"] = document

    state["previous_document"] = (
        previous_doc
    )

    print(
        f"[INFO] Current version: "
        f"{version}"
    )

    print(
        f"[INFO] Document ID: "
        f"{document_id}"
    )

    if previous_doc:

        print(
            f"[INFO] Previous version: "
            f"{previous_doc.version}"
        )

        print(
            f"[INFO] Previous document ID: "
            f"{previous_doc.id}"
        )

    else:

        print(
            "[INFO] No previous version."
        )

    log_end(
        "Version Detection",
        start,
    )

    return state


# ===========================================================================
# Requirement Agent
# ===========================================================================

def requirement_agent_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Requirement Agent"
    )

    if state.get("is_duplicate"):

        print(
            "[INFO] Duplicate document. "
            "Skipping Requirement Agent."
        )

        log_end(
            "Requirement Agent",
            start,
        )

        return state

    db = state["db"]

    document = state["document"]

    tree = state.get("tree") or []

    save_requirement_tree(
        db,
        tree,
        document.id,
    )

    version_status = (
        mark_stale_requirements(
            db,
            document,
            state.get(
                "previous_document"
            ),
        )
    )

    state["version_status"] = (
        version_status
    )

    print(
        f"[INFO] Version status: "
        f"{version_status}"
    )

    stored_count = (
        requirements_collection.count_documents(
            {
                "document_id": document.id
            }
        )
    )

    print(
        f"[INFO] Requirements stored: "
        f"{stored_count}"
    )

    log_end(
        "Requirement Agent",
        start,
    )

    return state


# ===========================================================================
# Derived Retrieval Indexes
# ===========================================================================

def build_indexes_node(
    state: QAGraphState,
) -> QAGraphState:
    """
    Build derived retrieval indexes for a newly ingested document.

    MongoDB remains the source of truth.

    Derived indexes:
        1. Qdrant vector index
        2. Knowledge Graph

    Duplicate documents never reach this node.
    """

    start = log_start(
        "Derived Indexes"
    )

    if state.get("is_duplicate"):

        print(
            "[INFO] Duplicate document. "
            "Skipping Qdrant and Knowledge Graph."
        )

        log_end(
            "Derived Indexes",
            start,
        )

        return state

    document_id = state["document_id"]

    document = state["document"]

    vector_count = index_document_requirements(
        document_id=document_id
    )

    print(
        f"[INFO] Qdrant vectors indexed: "
        f"{vector_count}"
    )

    graph = create_knowledge_graph(
        document_id=document_id
    )

    graph_path = (
        f"data/"
        f"ct200_knowledge_graph_"
        f"{document.version}.graphml"
    )

    saved_graph = save_graph(
        graph,
        output_path=graph_path,
    )

    print(
        f"[INFO] Knowledge Graph nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"[INFO] Knowledge Graph edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"[INFO] Knowledge Graph saved: "
        f"{saved_graph}"
    )

    state["evidence_pack"] = {
        **state.get(
            "evidence_pack",
            {}
        ),

        "vector_count": vector_count,
        "knowledge_graph_path": saved_graph,
    }

    log_end(
        "Derived Indexes",
        start,
    )

    return state


# ===========================================================================
# INGESTION ROUTERS
# ===========================================================================

def route_after_identity(
    state: QAGraphState,
) -> str:

    if state.get("is_duplicate"):

        return "duplicate"

    return "document_gate"


def route_after_gate(
    state: QAGraphState,
) -> str:

    if state.get("is_relevant"):

        return "version_detection"

    return "reject"


# ===========================================================================
# SELECTED REQUIREMENT QA
# ===========================================================================

def evidence_pack_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Evidence Pack"
    )

    document = state["document"]

    requirement = state[
        "selected_requirement"
    ]

    evidence_pack = build_evidence_pack(
        document_id=document.id,
        requirement=requirement,
        top_k=5,
    )

    state["evidence_pack"] = (
        evidence_pack
    )

    print(
        f"[INFO] Evidence Pack created for: "
        f"{requirement['node_id']}"
    )

    print(
        f"[INFO] Related evidence: "
        f"{len(evidence_pack.get('related_evidence', []))}"
    )

    log_end(
        "Evidence Pack",
        start,
    )

    return state


def selected_test_case_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Selected Requirement - Test Case Agent"
    )

    db = state["db"]
    document = state["document"]
    node = state["selected_requirement"]

    feedback = state.get(
        "validation_feedback"
    )

    print(
        f"[INFO] Selected requirement: "
        f"{node['node_id']}"
    )

    print(
        f"[INFO] Title: "
        f"{node['title']}"
    )

    # -----------------------------------------------------------------------
    # Check cached test cases
    # -----------------------------------------------------------------------

    cached_test_cases = get_cached_test_cases(
        document_id=document.id,
        requirement_node_id=node["id"],
        source_content_hash=node["content_hash"],
    )

    if cached_test_cases:

        print(
            "[CACHE] Valid approved test cases found."
        )

        test_cases = cached_test_cases

        node["test_cases"] = test_cases

        state["selected_requirement"] = node
        state["selected_test_cases"] = test_cases
        state["cache_hit"] = True
        state["error"] = None

        print(
            f"[CACHE] Returning {len(test_cases)} "
            "cached test case(s)."
        )

        log_end(
            "Selected Requirement - Test Case Agent",
            start,
        )

        return state

    # -----------------------------------------------------------------------
    # Evidence Pack -> Test Case Agent
    # -----------------------------------------------------------------------

    print(
        "[CACHE] No valid cache found. "
        "Generating new test cases."
    )

    test_cases = generate_test_cases(
        requirement=node,
        evidence_pack=state.get(
            "evidence_pack",
            {}
        ),
        feedback=feedback,
    )

    node["test_cases"] = test_cases

    save_test_cases(
        db,
        node["id"],
        test_cases,
        document_id=document.id,
        source_version=document.version,
        source_content_hash=node[
            "content_hash"
        ],
    )

    state["selected_requirement"] = node

    state["selected_test_cases"] = (
        test_cases
    )

    # -----------------------------------------------------------------------
    # Generation failure handling
    # -----------------------------------------------------------------------

    if not test_cases:

        state["error"] = (
            "Test case generation failed: "
            "LLM returned no valid test cases."
        )

        print(
            "[ERROR] Test case generation failed."
        )

        print(
            "[INFO] Validation Agent will be skipped."
        )

    else:

        state["error"] = None

        print(
            f"[INFO] Generated test cases: "
            f"{len(test_cases)}"
        )

    log_end(
        "Selected Requirement - Test Case Agent",
        start,
    )

    return state


def selected_validation_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Selected Requirement - Validation Agent"
    )

    db = state["db"]

    document = state["document"]

    node = state["selected_requirement"]

    attempt = (
        state.get("retry_count", 0) + 1
    )

    print(
        f"[INFO] Validation attempt: "
        f"{attempt}"
    )

    result = validate_node(
        db,
        document.id,
        node,
        node["test_cases"],
        attempt=attempt,
    )

    verdict = result.get(
        "verdict",
        "fail",
    )

    print(
        f"[INFO] Validation verdict: "
        f"{verdict}"
    )

    state["validation_result"] = result

    if verdict != "pass":

        state["retry_count"] = (
            state.get("retry_count", 0) + 1
        )

        state["validation_feedback"] = (
            result.get(
                "feedback",
                "",
            )
        )

    else:

        state["validation_feedback"] = ""

    log_end(
        "Selected Requirement - Validation Agent",
        start,
    )

    return state



def human_approval_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Human-in-the-Loop Approval"
    )

    test_cases = state.get(
        "selected_test_cases", []
    )

    if not test_cases:
        state["error"] = (
            "No test cases available for human approval."
        )
        return state

    # ==========================================================
    # 1. SELECTED REQUIREMENT
    # ==========================================================

    requirement = state.get(
        "selected_requirement", {}
    )

    print("\n" + "=" * 80)
    print("SELECTED REQUIREMENT - SOURCE")
    print("=" * 80)

    print(
        f"Requirement ID: "
        f"{requirement.get('id', '')}"
    )

    print(
        f"Title: "
        f"{requirement.get('title', '')}"
    )

    print(
        f"Requirement Text: "
        f"{requirement.get('text', '')}"
    )

    print(
        f"Source Page: "
        f"{requirement.get('page', '')}"
    )

    print("=" * 80)

    # ==========================================================
    # 2. RETRIEVED EVIDENCE
    # ==========================================================

    evidence_pack = state.get(
        "evidence_pack", {}
    )

    print("\n" + "=" * 80)
    print("RETRIEVED EVIDENCE")
    print("=" * 80)

    if evidence_pack:

        # Show complete evidence pack
        if isinstance(evidence_pack, dict):

            for key, value in evidence_pack.items():
                print(f"\n{key}:")
                print(value)

        else:
            print(evidence_pack)

    else:
        print("No evidence pack available.")

    print("=" * 80)

    # ==========================================================
    # 3. LLM GENERATED TEST CASES
    # ==========================================================

    print(
        "\n[HITL] Generated test cases require human approval."
    )

    for tc in test_cases:

        print("\n" + "=" * 80)
        print("LLM GENERATED TEST CASE")
        print("=" * 80)

        print(
            f"Test Case ID: "
            f"{tc.get('id', '')}"
        )

        print(
            f"Title: "
            f"{tc.get('title', '')}"
        )

        print(
            f"Requirement: "
            f"{tc.get('requirement', '')}"
        )

        print(
            f"Preconditions: "
            f"{tc.get('preconditions', '')}"
        )

        print(
            f"Test Steps: "
            f"{tc.get('test_steps', '')}"
        )

        print(
            f"Expected Result: "
            f"{tc.get('expected_result', '')}"
        )

        print(
            f"Source Evidence: "
            f"{tc.get('source_evidence', '')}"
        )

        print("=" * 80)

    # ==========================================================
    # 4. HUMAN DECISION
    # ==========================================================

    approval = input(
        "\nApprove test cases? (yes/no): "
    ).strip().lower()

    if approval == "yes":

        state["human_approval"] = "APPROVED"
        approval_status = "APPROVED"

        print(
            "[HITL] Test cases approved."
        )

    else:

        state["human_approval"] = "REJECTED"
        approval_status = "REJECTED"

        # Increment retry count for HITL rejection
        state["retry_count"] = (
            state.get("retry_count", 0) + 1
        )

        print(
            f"[HITL] Test cases rejected. "
            f"Retry count: {state['retry_count']}"
        )

    # ==========================================================
    # 5. SAVE HITL DECISION
    # ==========================================================

    document = state["document"]
    node = state["selected_requirement"]

    for tc in test_cases:

        test_cases_collection.update_one(
            {
                "document_id": document.id,
                "requirement_node_id": node["id"],
                "test_case_id": tc["id"],
            },
            {
                "$set": {
                    "human_approval": approval_status,
                    "validation_status": "PASS",
                }
            },
        )

    log_end(
        "Human-in-the-Loop Approval",
        start,
    )

    return state



def selected_traceability_node(
    state: QAGraphState,
) -> QAGraphState:

    start = log_start(
        "Selected Requirement - Traceability"
    )

    db = state["db"]

    document = state["document"]

    node = state["selected_requirement"]

    build_traceability(
        db,
        document.id,
        [node["id"]],
    )

    print(
        f"[INFO] Traceability created for: "
        f"{node['node_id']}"
    )

    log_end(
        "Selected Requirement - Traceability",
        start,
    )

    return state


# ===========================================================================
# Selected Requirement Routers
# ===========================================================================

def route_after_test_case_generation(
    state: QAGraphState,
) -> str:

    test_cases = state.get(
        "selected_test_cases",
        []
    )

    if not test_cases:
        return "generation_failed"

    if state.get("cache_hit"):
        return "traceability"

    return "validation"


def route_after_selected_validation(
    state: QAGraphState,
) -> str:

    result = state.get(
        "validation_result",
        {},
    )

    verdict = result.get(
        "verdict",
        "fail",
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    max_retries = state.get(
        "max_retries",
        DEFAULT_MAX_RETRIES,
    )

    if verdict == "pass":

        return "human_approval"

    if retry_count <= max_retries:

        return "regenerate"

    return "validation_failed"


# ===========================================================================
# INGESTION GRAPH
# ===========================================================================

def build_ingestion_graph():

    graph = StateGraph(
        QAGraphState
    )

    graph.add_node(
        "document_identity",
        document_identity_node,
    )

    graph.add_node(
        "document_gate",
        document_gate_node,
    )

    graph.add_node(
        "reject",
        reject_node,
    )

    graph.add_node(
        "version_detection",
        version_detection_node,
    )

    graph.add_node(
        "requirement_agent",
        requirement_agent_node,
    )

    graph.add_node(
        "build_indexes",
        build_indexes_node,
    )

    graph.set_entry_point(
        "document_identity"
    )

    graph.add_conditional_edges(
        "document_identity",
        route_after_identity,
        {
            "duplicate": END,
            "document_gate": "document_gate",
        },
    )

    graph.add_conditional_edges(
        "document_gate",
        route_after_gate,
        {
            "version_detection":
                "version_detection",

            "reject":
                "reject",
        },
    )

    graph.add_edge(
        "reject",
        END,
    )

    graph.add_edge(
        "version_detection",
        "requirement_agent",
    )

    graph.add_edge(
        "requirement_agent",
        "build_indexes",
    )

    graph.add_edge(
        "build_indexes",
        END,
    )

    return graph.compile()
#Route_after_human_approval
def route_after_human_approval(state: QAGraphState):
    if state.get("human_approval") == "APPROVED":
        return "traceability"

    retry_count = state.get("retry_count", 0)

    if retry_count < state.get("max_retries", DEFAULT_MAX_RETRIES):
        return "regenerate"

    return "validation_failed"
# ===========================================================================
# SELECTED REQUIREMENT GRAPH
# ===========================================================================

def build_selected_requirement_graph():

    graph = StateGraph(QAGraphState)

    graph.add_node("evidence_pack", evidence_pack_node)
    graph.add_node("test_case_agent", selected_test_case_node)
    graph.add_node("validation_agent", selected_validation_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("traceability_agent", selected_traceability_node)

    graph.set_entry_point("evidence_pack")

    graph.add_edge("evidence_pack", "test_case_agent")

    graph.add_conditional_edges(
        "test_case_agent",
        route_after_test_case_generation,
        {
            "validation": "validation_agent",
            "traceability":"traceability_agent",
            "generation_failed": END,
        },
    )

    graph.add_conditional_edges(
        "validation_agent",
        route_after_selected_validation,
        {
            "regenerate": "evidence_pack",
            "human_approval": "human_approval",
            "validation_failed": END,
        },
    )

    graph.add_conditional_edges(
    "human_approval",
    route_after_human_approval,
    {
        "traceability": "traceability_agent",
        "regenerate": "evidence_pack",
        "validation_failed":END,
    },
    )

    graph.add_edge("traceability_agent", END)
    

    return graph.compile()


# ===========================================================================
# Compile both graphs
# ===========================================================================

ingestion_graph = (
    build_ingestion_graph()
)

selected_requirement_graph = (
    build_selected_requirement_graph()
)


# ===========================================================================
# PUBLIC PIPELINE 1
# ===========================================================================

def run_pipeline(
    db,
    filename: str,
    pdf_path: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> QAGraphState:

    start = time.perf_counter()

    print("\n" + "=" * 70)
    print("STARTING DOCUMENT INGESTION")
    print(f"Document: {filename}")
    print("=" * 70)

    state: QAGraphState = {
        "db": db,
        "filename": filename,
        "pdf_path": pdf_path,
        "retry_count": 0,
        "max_retries": max_retries,
    }

    result = ingestion_graph.invoke(
        state
    )

    elapsed = (
        time.perf_counter() - start
    )

    print("\n" + "=" * 70)
    print(
        f"INGESTION COMPLETE: "
        f"{elapsed:.2f}s"
    )
    print("=" * 70)

    return result


# ===========================================================================
# PUBLIC PIPELINE 2
# ===========================================================================

def run_selected_requirement(
    db,
    document_id: str,
    node_id: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> QAGraphState:

    start = time.perf_counter()

    print("\n" + "=" * 70)
    print("STARTING SELECTED REQUIREMENT QA")
    print(
        f"Document ID: {document_id}"
    )
    print(
        f"Requirement: {node_id}"
    )
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load document from MongoDB
    # -----------------------------------------------------------------------

    raw_document = documents_collection.find_one(
        {
            "id": document_id
        }
    )

    if not raw_document:

        from bson import ObjectId

        try:

            raw_document = (
                documents_collection.find_one(
                    {
                        "_id": ObjectId(
                            document_id
                        )
                    }
                )
            )

        except Exception:

            raw_document = None

    if not raw_document:

        raise ValueError(
            f"Document {document_id} not found"
        )

    # -----------------------------------------------------------------------
    # Reconstruct Pydantic document model
    # -----------------------------------------------------------------------

    created_at = raw_document.get(
        "created_at"
    )

    if created_at is None:

        created_at = datetime.utcnow()

        documents_collection.update_one(
            {
                "_id": raw_document["_id"]
            },
            {
                "$set": {
                    "created_at": created_at
                }
            },
        )

    document = Document.model_validate(
        {
            "id": str(
                raw_document["_id"]
            ),

            "name": raw_document["name"],

            "version": raw_document["version"],

            "content_hash": raw_document.get(
                "content_hash",
                "",
            ),

            "previous_document_id": (
                raw_document.get(
                    "previous_document_id"
                )
            ),

            "created_at": created_at,

            "is_relevant": raw_document.get(
                "is_relevant",
                True,
            ),

            "relevance_reason": (
                raw_document.get(
                    "relevance_reason"
                )
            ),

            "version_status": (
                raw_document.get(
                    "version_status"
                )
            ),
        }
    )

    # -----------------------------------------------------------------------
    # Load requirement from MongoDB
    # -----------------------------------------------------------------------

    raw_node = requirements_collection.find_one(
        {
            "document_id": document.id,
            "node_id": node_id,
        }
    )

    if not raw_node:

        raise ValueError(
            f"Requirement {node_id} "
            f"not found in document "
            f"{document_id}"
        )

    selected_requirement = {

        # Backward compatibility for existing
        # Test Case and Traceability code.
        "id": raw_node["node_id"],

        # Canonical MongoDB requirement identifier.
        "node_id": raw_node["node_id"],

        "number": raw_node.get(
            "number"
        ),

        "title": raw_node.get(
            "title",
            "",
        ),

        "text": (
            [raw_node["text"]]
            if raw_node.get("text")
            else []
        ),

        "tables": raw_node.get(
            "tables",
            [],
        ),

        "children": [],

        "test_cases": [],

        "content_hash": raw_node.get(
            "content_hash"
        ),

        "page": raw_node.get(
            "page"
        ),
    }

    state: QAGraphState = {

        "db": db,

        "document": document,

        "selected_requirement":
            selected_requirement,

        "selected_test_cases": [],

        "retry_count": 0,

        "max_retries": max_retries,

        "validation_feedback": "",
    }

    result = (
        selected_requirement_graph.invoke(
            state
        )
    )

    elapsed = (
        time.perf_counter() - start
    )

    print("\n" + "=" * 70)

    print(
        f"SELECTED REQUIREMENT QA COMPLETE: "
        f"{elapsed:.2f}s"
    )

    print("=" * 70)

    return result 