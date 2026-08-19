"""
Test Case Agent.

Generates high-quality test cases for a requirement node.
Accepts optional feedback from the Validation Agent so it can
regenerate test cases when the previous attempt is rejected.
"""

import json

from app.llm_client import call_llm_json
from app.models import TestCase


def build_prompt(requirement: dict, feedback: str | None = None) -> str:
    feedback_block = ""

    if feedback:
        feedback_block = f"""
The previous test-case generation attempt was rejected by the
Validation Agent.

Validation feedback:
{feedback}

Regenerate the test cases and specifically fix every issue identified
in the feedback. Do not repeat the previous mistakes.
"""

    return f"""
You are a Senior QA Engineer responsible for creating precise,
requirement-based software test cases.

Your task is to generate test cases ONLY from the information contained
in the requirement below.

IMPORTANT RULES:

1. Every test case must directly validate the stated requirement.
2. Do not invent functionality, UI elements, APIs, hardware features,
   error messages, limits, or system behavior that is not supported by
   the requirement.
3. Cover the requirement's important acceptance conditions.
4. Include both:
   - Positive/normal behavior
   - Negative, boundary, or failure scenarios when they are logically
     applicable from the requirement.
5. Steps must be concrete, executable, and ordered.
6. Expected results must be observable and directly tied to the
   requirement.
7. Preconditions should contain only conditions necessary to execute
   the test.
8. Do not create duplicate or substantially overlapping test cases.
9. If the requirement contains a numeric limit, threshold, duration,
   range, condition, or constraint, explicitly test that condition.
10. Maintain traceability: every generated test case must be clearly
    related to Requirement ID {requirement["id"]}.
11. Do not add assumptions merely to make a test case more detailed.
12. Generate a reasonable number of test cases based on the complexity
    of the requirement. Do not generate unnecessary test cases.

REQUIREMENT

Requirement ID:
{requirement["id"]}

Title:
{requirement["title"]}

Description:
{" ".join(requirement.get("text", []))}

Tables:
{json.dumps(requirement.get("tables", []), indent=2)}

{feedback_block}

OUTPUT REQUIREMENTS

Return ONLY valid JSON.

Return a JSON array using exactly this structure:

[
  {{
    "id": "TC-001",
    "title": "Clear and specific test case title",
    "preconditions": "Required preconditions",
    "steps": [
      "Step 1",
      "Step 2",
      "Step 3"
    ],
    "expected_result": "Specific observable expected result",
    "priority": "High"
  }}
]

Use priority values only from:
- High
- Medium
- Low

The test cases must be:
- specific
- executable
- non-duplicative
- requirement-driven
- traceable
- suitable for review by a professional QA engineer.
"""


def generate_test_cases(
    requirement: dict,
    feedback: str | None = None
) -> list:
    prompt = build_prompt(requirement, feedback)

    result = call_llm_json(prompt)

    return result if isinstance(result, list) else []


def save_test_cases(db, requirement_node_id: str, test_cases: list):
    saved = []

    for tc in test_cases:
        try:
            db_test_case = TestCase(
                requirement_node_id=requirement_node_id,
                test_case_id=tc["id"],
                title=tc["title"],
                preconditions=tc.get("preconditions", ""),
                steps="\n".join(tc.get("steps", [])),
                expected_result=tc.get("expected_result", ""),
                priority=tc.get("priority", ""),
            )

            db.add(db_test_case)
            db.commit()
            db.refresh(db_test_case)

            saved.append(db_test_case)

        except Exception as e:
            db.rollback()
            print("DATABASE ERROR while saving test case:", e)

    return saved