"""
LangGraph Orchestrator.

Wires the Document Gate (CRAG) -> Version Detection -> Requirement Agent ->
Test Case Agent -> Validation Agent (Self-RAG) -> Traceability Agent
pipeline, with a regeneration loop back to the Test Case Agent on
validation failure, matching the target architecture diagram.
"""

from langgraph.graph import StateGraph, END

from app.graph_state import QAGraphState
from app.agents.document_gate import check_document_relevance
from app.agents.version_control import resolve_document_version, mark_stale_requirements
from app.agents.requirement_agent import extract_requirements, save_requirement_tree
from app.agents.test_case_agent import generate_test_cases, save_test_cases
from app.agents.validation_agent import validate_node
from app.agents.traceability_agent import build_traceability
from app.models import Document

DEFAULT_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def document_gate_node(state: QAGraphState) -> QAGraphState:
    tree = extract_requirements(state["pdf_path"])

    sample_text = " ".join(
        " ".join(n.get("text", [])) for n in tree[:3]
    ) if tree else ""

    gate = check_document_relevance(tree, state["filename"], sample_text)

    state["tree"] = tree
    state["is_relevant"] = bool(gate.get("is_relevant"))
    state["relevance_reason"] = gate.get("reason", "")
    return state


def reject_node(state: QAGraphState) -> QAGraphState:
    state["rejected"] = True
    state["error"] = f"Document rejected by Document Gate: {state.get('relevance_reason')}"
    return state


def version_detection_node(state: QAGraphState) -> QAGraphState:
    db = state["db"]
    version, previous_doc = resolve_document_version(db, state["filename"])

    document = Document(name=state["filename"], version=version)
    db.add(document)
    db.commit()
    db.refresh(document)

    state["document"] = document
    state["previous_document"] = previous_doc
    return state


def requirement_agent_node(state: QAGraphState) -> QAGraphState:
    db = state["db"]
    document = state["document"]

    if state.get("tree"):
        save_requirement_tree(db, state["tree"], document.id)

    state["version_status"] = mark_stale_requirements(db, document, state.get("previous_document"))
    return state


def test_case_agent_node(state: QAGraphState) -> QAGraphState:
    db = state["db"]
    tree = state.get("tree") or []

    feedback = None
    if state.get("retry_count"):
        feedback = state.get("validation_summary", {}).get("feedback")

    def walk(nodes):
        for node in nodes:
            if node.get("text") or node.get("tables"):
                node["test_cases"] = generate_test_cases(node, feedback=feedback)
                save_test_cases(db, node["id"], node["test_cases"])
            else:
                node["test_cases"] = []

            if node.get("children"):
                walk(node["children"])

    walk(tree)
    return state


def validation_agent_node(state: QAGraphState) -> QAGraphState:
    db = state["db"]
    document = state["document"]
    tree = state.get("tree") or []
    attempt = state.get("retry_count", 0) + 1

    failures = []

    def walk(nodes):
        for node in nodes:
            if node.get("test_cases"):
                result = validate_node(db, document.id, node, node["test_cases"], attempt=attempt)
                if result.get("verdict") != "pass":
                    failures.append({
                        "node_id": node["id"],
                        "feedback": result.get("feedback", ""),
                    })

            if node.get("children"):
                walk(node["children"])

    walk(tree)

    state["validation_summary"] = {
        "failures": failures,
        "feedback": " | ".join(f["feedback"] for f in failures if f["feedback"]),
    }

    if failures:
        state["retry_count"] = state.get("retry_count", 0) + 1

    return state


def traceability_agent_node(state: QAGraphState) -> QAGraphState:
    db = state["db"]
    document = state["document"]
    tree = state.get("tree") or []

    node_ids = []

    def walk(nodes):
        for node in nodes:
            node_ids.append(node["id"])
            if node.get("children"):
                walk(node["children"])

    walk(tree)

    build_traceability(db, document.id, node_ids)
    return state


# ---------------------------------------------------------------------------
# Conditional routers
# ---------------------------------------------------------------------------
def route_after_gate(state: QAGraphState) -> str:
    return "version_detection" if state.get("is_relevant") else "reject"


def route_after_validation(state: QAGraphState) -> str:
    failures = state.get("validation_summary", {}).get("failures", [])
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if failures and state.get("retry_count", 0) <= max_retries:
        return "regenerate"

    return "traceability"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(QAGraphState)

    graph.add_node("document_gate", document_gate_node)
    graph.add_node("reject", reject_node)
    graph.add_node("version_detection", version_detection_node)
    graph.add_node("requirement_agent", requirement_agent_node)
    graph.add_node("test_case_agent", test_case_agent_node)
    graph.add_node("validation_agent", validation_agent_node)
    graph.add_node("traceability_agent", traceability_agent_node)

    graph.set_entry_point("document_gate")

    graph.add_conditional_edges(
        "document_gate",
        route_after_gate,
        {"version_detection": "version_detection", "reject": "reject"},
    )

    graph.add_edge("reject", END)
    graph.add_edge("version_detection", "requirement_agent")
    graph.add_edge("requirement_agent", "test_case_agent")
    graph.add_edge("test_case_agent", "validation_agent")

    graph.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {"regenerate": "test_case_agent", "traceability": "traceability_agent"},
    )

    graph.add_edge("traceability_agent", END)

    return graph.compile()


qa_graph = build_graph()


def run_pipeline(db, filename: str, pdf_path: str, max_retries: int = DEFAULT_MAX_RETRIES) -> QAGraphState:
    initial_state: QAGraphState = {
        "db": db,
        "filename": filename,
        "pdf_path": pdf_path,
        "retry_count": 0,
        "max_retries": max_retries,
    }
    return qa_graph.invoke(initial_state)
