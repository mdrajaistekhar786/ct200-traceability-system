"""
Validation Agent -- Self-RAG (with cross-requirement retrieval).

For each requirement, retrieves two kinds of grounding evidence:
  1. The requirement's own extracted text/tables (primary evidence).
  2. The most lexically-related OTHER requirements in the same document
     (cross-requirement evidence), so the agent can catch a test case that
     is actually grounded in a different section, or a requirement whose
     correct testing depends on a neighboring one.

Persists one ValidationResult row per attempt, including which related
requirements were retrieved, so failed attempts stay auditable.
"""

from app.llm_client import call_llm_json
from app.models import ValidationResult
from app.agents.retrieval import retrieve_related_requirements


def validate_node(db, document_id: int, requirement: dict, test_cases: list, attempt: int = 1) -> dict:
    """
    Returns {"verdict": "pass"|"fail", "per_test_case": [...], "feedback": str}
    """
    if not test_cases:
        result = {
            "verdict": "fail",
            "per_test_case": [],
            "feedback": "No test cases were generated for this requirement.",
        }
        _save_validation_result(db, requirement["id"], result, attempt, related=[])
        return result

    evidence = " ".join(requirement.get("text", [])) or "(no descriptive text extracted for this section)"

    related = retrieve_related_requirements(
        db=db,
        document_id=document_id,
        current_node_id=requirement["id"],
        query_text=f'{requirement.get("title", "")} {evidence}',
        top_k=3,
    )

    if related:
        cross_evidence = "\n".join(
            f'- {r["node_id"]} "{r["title"]}" (similarity {r["score"]}): {r["text"]}'
            for r in related
        )
    else:
        cross_evidence = "(no closely related requirements found elsewhere in the document)"

    prompt = f"""
You are a QA Validation Agent performing Self-RAG style checking: you must
ground every judgment in the retrieved evidence below, not in outside
assumptions.

Requirement being validated:
ID: {requirement["id"]}
Title: {requirement.get("title", "")}

Primary evidence (this requirement's own source manual text):
{evidence}

Cross-requirement evidence (the most closely related OTHER sections
retrieved from elsewhere in the manual -- use this to catch a test case
that is actually testing a different requirement, or that depends on
something defined in a neighboring section):
{cross_evidence}

Test cases to validate:
{test_cases}

For each test case, judge whether it is:
  1. Clearly grounded in the primary evidence above (not hallucinated).
  2. NOT actually testing one of the cross-requirement evidence sections
     instead of this requirement (misattribution).
  3. Consistent with any related requirement it depends on or overlaps with.

Return ONLY valid JSON in this exact shape:
{{
  "verdict": "pass" or "fail",
  "per_test_case": [
    {{"id": "TC-001", "grounded": true, "misattributed_to": null, "comment": "short reason"}}
  ],
  "feedback": "concise actionable feedback for regeneration if verdict is fail, else empty string"
}}

Rules: verdict is "fail" if ANY test case is not grounded, or is better
grounded in one of the cross-requirement sections than in this requirement.
"""

    result = call_llm_json(prompt)

    if not isinstance(result, dict) or "verdict" not in result:
        # LLM unavailable/malformed -> fail safe so a human can review,
        # rather than silently accepting ungrounded test cases.
        result = {
            "verdict": "fail",
            "per_test_case": [],
            "feedback": "Validator could not parse a response; treating as fail-safe for manual review.",
        }

    _save_validation_result(db, requirement["id"], result, attempt, related)

    return result


def _save_validation_result(db, requirement_node_id: str, result: dict, attempt: int, related: list):
    try:
        row = ValidationResult(
            requirement_node_id=requirement_node_id,
            verdict=result.get("verdict", "fail"),
            feedback=result.get("feedback", ""),
            evidence_detail=str(result.get("per_test_case", [])),
            related_requirement_ids=",".join(r["node_id"] for r in related) if related else None,
            attempt=attempt,
        )
        db.add(row)
        db.commit()
    except Exception as e:
        db.rollback()
        print("DATABASE ERROR while saving validation result:", e)
