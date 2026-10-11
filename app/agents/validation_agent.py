"""
Validation Agent -- Self-RAG (with cross-requirement retrieval).

For each requirement, retrieves two kinds of grounding evidence:
1. The requirement's own extracted text/tables (primary evidence).
2. The most lexically-related OTHER requirements in the same document
   (cross-requirement evidence).

The validator checks whether generated test cases are actually supported
by the requirement and whether they introduce unsupported assumptions.

Validation output is validated using Pydantic before being persisted.
"""

from datetime import datetime

from pydantic import ValidationError

from app.llm_client import call_llm_json
from app.schemas import ValidationAgentOutput
from app.agents.retrieval import retrieve_related_requirements
from app.database import validation_results_collection


def validate_node(
    db,
    document_id: str,
    requirement: dict,
    test_cases: list,
    attempt: int = 1
) -> dict:
    """
    Validate generated test cases against requirement evidence.

    Returns:
        {
            "verdict": "pass" | "fail",
            "per_test_case": [...],
            "feedback": "..."
        }
    """

    if not test_cases:
        result = {
            "verdict": "fail",
            "per_test_case": [],
            "feedback": (
                "No test cases were generated for this requirement."
            ),
        }

        _save_validation_result(
            db,
            requirement["node_id"],
            result,
            attempt,
            related=[]
        )

        return result

    evidence = " ".join(
        requirement.get("text", [])
    ) or "(no descriptive text extracted for this section)"

    related = retrieve_related_requirements(
        db=db,
        document_id=document_id,
        current_node_id=requirement["node_id"],
        query_text=(
            f'{requirement.get("title", "")} '
            f'{evidence}'
        ),
        top_k=3,
    )

    if related:
        cross_evidence = "\n".join(
            f'- {r["node_id"]} "{r["title"]}" '
            f'(similarity {r["score"]}): {r["text"]}'
            for r in related
        )
    else:
        cross_evidence = (
            "(no closely related requirements found elsewhere "
            "in the document)"
        )

    prompt = f"""
You are a Senior QA Validation Agent performing strict
Self-RAG-style grounding validation.

Your job is NOT to decide whether a test case sounds reasonable.

Your job is to determine whether EVERY claim made by each test case
is directly supported by the retrieved source evidence.

============================================================
REQUIREMENT BEING VALIDATED
============================================================

Requirement ID:
{requirement["node_id"]}

Requirement Title:
{requirement.get("title", "")}

============================================================
PRIMARY EVIDENCE
============================================================

This is the authoritative evidence for this requirement:

{evidence}

============================================================
CROSS-REQUIREMENT EVIDENCE
============================================================

These are other sections from the same document.

They may provide supporting context, but they MUST NOT be treated
as evidence that the current requirement contains a fact that it
does not actually state.

{cross_evidence}

============================================================
TEST CASES TO VALIDATE
============================================================

{test_cases}

============================================================
STRICT VALIDATION RULES
============================================================

For EACH test case:

1. Check whether the test case is directly grounded in the
   PRIMARY EVIDENCE.

2. Every important claim in:
   - preconditions
   - steps
   - expected result
   must be supported by the evidence.

3. Do NOT infer unstated relationships.

Example:

If the requirement says:

"approximately 300 measurement cycles"

and separately says:

"low-battery icon appears below 15% capacity"

you MUST NOT assume:

"300 cycles means below 15% capacity"

unless the requirement explicitly establishes that relationship.

Likewise, do NOT assume:

"299 cycles means above 15% capacity"

unless the source explicitly states that.

4. Words such as:
   - exactly
   - always
   - never
   - before
   - after
   - greater than
   - less than
   - equal to
   - at the limit
   - immediately
   - within
   - below
   - above

must NOT introduce a condition that the source does not establish.

5. Numeric values must be used carefully.

If the source says "approximately 300", a test case must not
silently convert that into an exact threshold of 300 unless
the source explicitly defines 300 as a hard boundary.

6. A test case may test a condition only when the requirement
provides enough information to establish the expected result.

7. If a test case requires information that is missing from the
requirement, mark it as NOT GROUNDED.

8. Do not use general QA knowledge or outside knowledge to fill
missing information.

9. Do not mark a test case grounded merely because it is
plausible or technically reasonable.

10. Check for misattribution.

A test case should fail if it is actually testing a different
requirement retrieved in the cross-requirement evidence.

11. Cross-requirement evidence can clarify relationships, but it
cannot override or contradict the primary requirement.

12. The validator must fail safe.

If there is meaningful uncertainty about whether a claim is
supported, mark the test case as NOT GROUNDED.

============================================================
VERDICT RULE
============================================================

Return:

"pass"

ONLY when ALL test cases are adequately grounded.

Return:

"fail"

when ANY test case:

- contains an unsupported claim,
- introduces an unstated assumption,
- uses an unsupported numeric boundary,
- has an expected result not justified by the evidence,
- is misattributed to another requirement,
- or cannot be confidently grounded.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
  "verdict": "pass",
  "per_test_case": [
    {{
      "id": "TC-001",
      "grounded": true,
      "misattributed_to": null,
      "comment": "The test case is directly supported by the requirement."
    }}
  ],
  "feedback": ""
}}

For a failed test case:

{{
  "id": "TC-001",
  "grounded": false,
  "misattributed_to": null,
  "comment": "The test assumes that 300 cycles corresponds to battery capacity below 15%, but the requirement does not establish this relationship."
}}

The "feedback" field must contain concise, actionable instructions
for the Test Case Agent to correct the failed test cases.

Do not include Markdown.
Do not include explanations outside the JSON.
"""

    result = call_llm_json(prompt)

    if result is None:
        result = {
            "verdict": "fail",
            "per_test_case": [],
            "feedback": (
                "Validator returned no valid JSON response. "
                "Treat the validation attempt as failed and "
                "regenerate after retry."
            ),
        }

    else:
        try:
            validated_output = (
                ValidationAgentOutput.model_validate(
                    result
                )
            )

            result = validated_output.model_dump()

        except ValidationError as exc:
            print(
                "VALIDATION AGENT OUTPUT FAILED "
                "Pydantic validation:"
            )
            print(exc)

            result = {
                "verdict": "fail",
                "per_test_case": [],
                "feedback": (
                    "Validator response failed Pydantic validation. "
                    "Regenerate the test cases after retry."
                ),
            }

    _save_validation_result(
        db,
        requirement["node_id"],
        result,
        attempt,
        related
    )

    return result


def _save_validation_result(
    db,
    requirement_node_id: str,
    result: dict,
    attempt: int,
    related: list
):
    """
    Save validation result to MongoDB.

    Previous validation attempts are preserved as history,
    while only the newest result is marked as is_latest=True.
    """

    try:
        # Mark previous validation results for this requirement
        # as historical.
        validation_results_collection.update_many(
            {
                "requirement_node_id": requirement_node_id,
                "is_latest": True,
            },
            {
                "$set": {
                    "is_latest": False,
                }
            }
        )

        validation_document = {
            "requirement_node_id": requirement_node_id,

            "verdict": result.get(
                "verdict",
                "fail"
            ),

            "feedback": result.get(
                "feedback",
                ""
            ),

            "evidence_detail": str(
                result.get(
                    "per_test_case",
                    []
                )
            ),

            "related_requirement_ids": (
                [
                    r["node_id"]
                    for r in related
                ]
                if related
                else []
            ),

            "attempt": attempt,

            "is_latest": True,

            "created_at": datetime.utcnow(),
        }

        validation_results_collection.insert_one(
            validation_document
        )

    except Exception as e:

        print(
            "DATABASE ERROR while saving "
            "validation result:",
            e
        )