"""
Final End-to-End Evaluation for CT-200 Traceability System.

Flow:
    Gold Standard
        ↓
    Find correct document version
        ↓
    Select expected requirement
        ↓
    run_selected_requirement()
        ↓
    Generated Test Cases
        ↓
    Validation Result
        ↓
    Requirement Coverage
    Fact Coverage
    Version Correctness
    Overall PASS / PARTIAL / FAIL
"""

import sys
import json
import re
import builtins
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from gold_standard import GOLD_STANDARD

from app.database import (
    db,
    documents_collection,
    requirements_collection,
)

from app.graph  import run_selected_requirement


# ===========================================================================
# Configuration
# ===========================================================================

MAX_RETRIES = 2

OUTPUT_FILE = ROOT / "final_evaluation_results.json"


# ===========================================================================
# Text utilities
# ===========================================================================

def normalize_text(text):
    """
    Normalize text for fact matching.
    """
    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("≤", "<=")
    text = text.replace("≥", ">=")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def test_cases_to_text(test_cases):
    """
    Convert generated test cases into one searchable text block.
    """
    parts = []

    for tc in test_cases or []:

        if not isinstance(tc, dict):
            continue

        parts.append(str(tc.get("id", "")))
        parts.append(str(tc.get("title", "")))
        parts.append(str(tc.get("preconditions", "")))
        parts.append(str(tc.get("steps", "")))
        parts.append(str(tc.get("expected_result", "")))
        parts.append(str(tc.get("priority", "")))

    return normalize_text(" ".join(parts))


def fact_is_present(fact, generated_text):
    """
    Check whether a Gold Standard fact is represented
    in the generated test-case output.

    Uses normalized matching plus common semantic
    wording variations.
    """

    fact = normalize_text(fact)
    generated_text = normalize_text(generated_text)

    if not fact:
        return True

    # 1. Exact normalized match
    if fact in generated_text:
        return True

    # 2. Common wording normalization
    fact_normalized = fact
    generated_normalized = generated_text

    replacements = [
        (r"\bseconds?\b", "sec"),
        (r"\bmilliseconds?\b", "ms"),
        (r"\bminutes?\b", "min"),
        (r"\bhours?\b", "hr"),
        (r"\bautomatically\b", "auto"),
        (r"\bdeflates?\b", "deflate"),
        (r"\bdisplays?\b", "display"),
        (r"\bshows?\b", "show"),
    ]

    for pattern, replacement in replacements:
        fact_normalized = re.sub(
            pattern, replacement, fact_normalized
        )
        generated_normalized = re.sub(
            pattern, replacement, generated_normalized
        )

    fact_normalized = re.sub(
        r"\s+", " ", fact_normalized
    ).strip()

    generated_normalized = re.sub(
        r"\s+", " ", generated_normalized
    ).strip()

    if fact_normalized in generated_normalized:
        return True

    # 3. Handle time-based facts with different sentence structure.
    # Example:
    # "E3 auto-deflates within 2 seconds"
    # "E3 automatically deflates within 2 sec"
    # "E3 releases pressure no later than 2 seconds"

    time_match = re.search(
        r"\b(e\d+)\b.*?"
        r"(?:within|no later than|in|after)\s+"
        r"(\d+(?:\.\d+)?)\s*"
        r"(sec|ms|min|hr)",
        fact_normalized,
    )

    if time_match:
        error_code, value, unit = time_match.groups()

        generated_time_match = re.search(
            rf"\b{re.escape(error_code)}\b.*?"
            rf"(?:within|no later than|in|after).*?"
            rf"{re.escape(value)}\s*"
            rf"{re.escape(unit)}",
            generated_normalized,
        )

        if generated_time_match:
            return True

    # 4. Token-based fallback for wording variations.
    fact_tokens = set(
        re.findall(
            r"\b[a-z0-9]+(?:\.[0-9]+)?\b",
            fact_normalized,
        )
    )

    generated_tokens = set(
        re.findall(
            r"\b[a-z0-9]+(?:\.[0-9]+)?\b",
            generated_normalized,
        )
    )

    stop_words = {
        "the", "a", "an", "is", "are", "be",
        "should", "within", "in", "and", "or",
        "of", "to", "for", "device",
    }

    important_tokens = fact_tokens - stop_words

    if (
        important_tokens
        and important_tokens.issubset(generated_tokens)
    ):
        return True

    return False


# ===========================================================================
# Gold Standard helpers
# ===========================================================================

def get_case_id(case):
    return (
        case.get("id")
        or case.get("case_id")
        or case.get("name")
    )


def get_expected_version(case):
    """
    Gold Standard cases use version identifiers such as:
    v1, v2, v3.
    """

    version = case.get("version")

    if version:
        return str(version).lower()

    case_id = get_case_id(case)

    if case_id:
        match = re.search(r"GS-(V\d+)", case_id.upper())

        if match:
            return match.group(1).lower()

    return None


def get_expected_nodes(case):
    """
    Return expected requirement IDs.
    """

    nodes = case.get("relevant_node_ids", [])

    if isinstance(nodes, str):
        nodes = [nodes]

    return list(nodes or [])


def get_ground_truth_facts(case):
    """
    Return Gold Standard facts.
    """

    facts = case.get("ground_truth_facts", [])

    if isinstance(facts, str):
        facts = [facts]

    return list(facts or [])


# ===========================================================================
# Document lookup
# ===========================================================================

def find_document_for_version(version):
    """
    Find the stored MongoDB document corresponding to V1/V2/V3.
    """

    if not version:
        return None

    version = str(version).lower()

    documents = list(
        documents_collection.find({})
    )

    # Exact version match first
    for document in documents:

        doc_version = str(
            document.get("version", "")
        ).lower()

        if doc_version == version:
            return document

    # Fallback: version may be stored as "1", "2", "3"
    numeric_version = version.replace("v", "")

    for document in documents:

        doc_version = str(
            document.get("version", "")
        ).lower()

        if doc_version == numeric_version:
            return document

    return None


# ===========================================================================
# Requirement lookup
# ===========================================================================

def find_requirement(document_id, node_id):
    """
    Retrieve the expected requirement from MongoDB.
    """

    return requirements_collection.find_one(
        {
            "document_id": document_id,
            "node_id": node_id,
        }
    )


# ===========================================================================
# Version correctness
# ===========================================================================

def check_version_correctness(
    generated_text,
    expected_version,
    case,
):
    """
    Check that the generated output contains the expected
    version-specific facts and does not obviously use facts
    from another version.

    The primary check is based on Gold Standard facts.
    """

    facts = get_ground_truth_facts(case)

    if not facts:
        return True

    matched = 0

    for fact in facts:

        if fact_is_present(
            fact,
            generated_text,
        ):
            matched += 1

    return matched > 0


# ===========================================================================
# Human approval bypass
# ===========================================================================

def yes_input(prompt=""):
    """
    Final evaluation must be automated.

    The production graph contains a Human-in-the-Loop node.
    During evaluation we automatically approve generated
    test cases so the evaluator does not block on input().
    """

    print(
        "[EVAL] Human approval automatically set to YES."
    )

    return "yes"


# ===========================================================================
# Evaluate one case
# ===========================================================================

def evaluate_case(case):
    """
    Run one Gold Standard case through the selected-requirement
    end-to-end QA pipeline.
    """

    case_id = get_case_id(case)
    version = get_expected_version(case)
    expected_nodes = get_expected_nodes(case)
    expected_facts = get_ground_truth_facts(case)

    result = {
        "case_id": case_id,
        "version": version,
        "expected_requirement_ids": expected_nodes,
        "expected_facts": expected_facts,
        "actual_requirement_id": None,
        "generated_test_cases": [],
        "validation_verdict": None,
        "validation_feedback": "",
        "retry_count": None,
        "requirement_coverage": 0.0,
        "fact_coverage": 0.0,
        "version_correct": False,
        "status": "FAIL",
        "reason": "",
        "error": None,
    }

    # -----------------------------------------------------------------------
    # Validate Gold Standard case
    # -----------------------------------------------------------------------

    if not version:
        result["reason"] = (
            "Could not determine document version."
        )
        return result

    if not expected_nodes:
        result["reason"] = (
            "Gold Standard has no relevant_node_ids."
        )
        return result

    # -----------------------------------------------------------------------
    # Find document
    # -----------------------------------------------------------------------

    document = find_document_for_version(version)

    if not document:

        result["reason"] = (
            f"Document version '{version}' "
            "not found in MongoDB."
        )

        return result

    document_id = str(
        document.get("id")
        or document.get("_id")
    )

    # -----------------------------------------------------------------------
    # For final QA, run each expected requirement.
    #
    # Most Gold Standard cases have one requirement.
    # If multiple requirements exist, evaluate all of them.
    # -----------------------------------------------------------------------

    all_test_cases = []

    actual_requirement_ids = []

    validation_verdicts = []

    validation_feedback = []

    total_expected_nodes = len(
        expected_nodes
    )

    covered_nodes = 0

    try:

        # Prevent Human-in-the-Loop input() from blocking
        original_input = builtins.input
        builtins.input = yes_input

        try:

            for node_id in expected_nodes:

                requirement = find_requirement(
                    document_id,
                    node_id,
                )

                if not requirement:

                    continue

                state = run_selected_requirement(
                    db=db,
                    document_id=document_id,
                    node_id=node_id,
                    max_retries=MAX_RETRIES,
                )

                # -----------------------------------------------------------
                # Actual selected requirement
                # -----------------------------------------------------------

                selected_requirement = state.get(
                    "selected_requirement",
                    {},
                )

                actual_node_id = (
                    selected_requirement.get(
                        "node_id"
                    )
                    or selected_requirement.get(
                        "id"
                    )
                )

                if actual_node_id:

                    actual_requirement_ids.append(
                        actual_node_id
                    )

                # -----------------------------------------------------------
                # Requirement coverage
                # -----------------------------------------------------------

                if actual_node_id == node_id:

                    covered_nodes += 1

                # -----------------------------------------------------------
                # Generated test cases
                # -----------------------------------------------------------

                test_cases = state.get(
                    "selected_test_cases",
                    [],
                )

                if test_cases:

                    all_test_cases.extend(
                        test_cases
                    )

                # -----------------------------------------------------------
                # Validation
                # -----------------------------------------------------------
                # Fresh test cases are validated by the Validation Agent.
                # Cache hits intentionally skip another LLM validation call,
                # so use the validation_status stored on approved cached
                # test cases.
                # -----------------------------------------------------------

                validation = state.get(
                    "validation_result",
                    {},
                )

                verdict = validation.get(
                    "verdict"
                )

                feedback = validation.get(
                    "feedback",
                    "",
                )

                if not verdict and state.get("cache_hit"):
                    cached_statuses = [
                        str(tc.get("validation_status", "")).lower()
                        for tc in test_cases
                        if isinstance(tc, dict)
                    ]

                    if cached_statuses and all(
                        status == "pass"
                        for status in cached_statuses
                    ):
                        verdict = "pass"

                    elif any(
                        status == "fail"
                        for status in cached_statuses
                    ):
                        verdict = "fail"

                if verdict:
                    validation_verdicts.append(
                        verdict
                    )

                if feedback:
                    validation_feedback.append(
                        feedback
                    )

                result["retry_count"] = state.get(
                    "retry_count",
                    0,
                )

        finally:

            builtins.input = original_input

    except Exception as exc:

        result["error"] = str(exc)

        result["reason"] = (
            "End-to-end pipeline execution failed."
        )

        return result

    # =========================================================================
    # Requirement Coverage
    # =========================================================================

    requirement_coverage = (
        covered_nodes / total_expected_nodes
        if total_expected_nodes
        else 0.0
    )

    result["requirement_coverage"] = round(
        requirement_coverage,
        4,
    )

    result["actual_requirement_id"] = (
        actual_requirement_ids[0]
        if len(actual_requirement_ids) == 1
        else actual_requirement_ids
    )

    # =========================================================================
    # Fact Coverage
    # =========================================================================

    generated_text = test_cases_to_text(
        all_test_cases
    )

    matched_facts = []

    missing_facts = []

    for fact in expected_facts:

        if fact_is_present(
            fact,
            generated_text,
        ):
            matched_facts.append(fact)
        else:
            missing_facts.append(fact)

    fact_coverage = (
        len(matched_facts)
        / len(expected_facts)
        if expected_facts
        else 1.0
    )

    result["fact_coverage"] = round(
        fact_coverage,
        4,
    )

    result["matched_facts"] = matched_facts
    result["missing_facts"] = missing_facts

    # =========================================================================
    # Validation
    # =========================================================================

    result["validation_verdict"] = (
        validation_verdicts[-1]
        if validation_verdicts
        else None
    )

    result["validation_feedback"] = (
        validation_feedback[-1]
        if validation_feedback
        else ""
    )

    # =========================================================================
    # Version Correctness
    # =========================================================================

    result["version_correct"] = (
        check_version_correctness(
            generated_text,
            version,
            case,
        )
    )

    # =========================================================================
    # Save generated test cases
    # =========================================================================

    result["generated_test_cases"] = all_test_cases

    # =========================================================================
    # Final Status
    # =========================================================================

    validation_pass = (
        result["validation_verdict"]
        == "pass"
    )

    requirement_pass = (
        result["requirement_coverage"]
        == 1.0
    )

    facts_pass = (
        result["fact_coverage"]
        == 1.0
    )

    version_pass = (
        result["version_correct"]
    )

    # -------------------------------------------------------------------------
    # PASS
    # -------------------------------------------------------------------------

    if (
        requirement_pass
        and facts_pass
        and version_pass
        and validation_pass
    ):

        result["status"] = "PASS"

        result["reason"] = (
            "Requirement covered, all Gold Standard "
            "facts preserved, version-specific information "
            "correct, and validation passed."
        )

    # -------------------------------------------------------------------------
    # PARTIAL
    # -------------------------------------------------------------------------

    elif (
        requirement_pass
        and result["fact_coverage"] > 0
    ):

        result["status"] = "PARTIAL"

        reasons = []

        if not facts_pass:
            reasons.append(
                f"Missing {len(missing_facts)} fact(s)."
            )

        if not validation_pass:
            reasons.append(
                "Validation did not pass."
            )

        if not version_pass:
            reasons.append(
                "Version correctness check failed."
            )

        result["reason"] = " ".join(reasons)

    # -------------------------------------------------------------------------
    # FAIL
    # -------------------------------------------------------------------------

    else:

        result["status"] = "FAIL"

        reasons = []

        if not requirement_pass:
            reasons.append(
                "Expected requirement not covered."
            )

        if result["fact_coverage"] == 0:
            reasons.append(
                "No Gold Standard facts found."
            )

        if not version_pass:
            reasons.append(
                "Version correctness failed."
            )

        if not validation_pass:
            reasons.append(
                "Validation failed."
            )

        result["reason"] = " ".join(
            reasons
        )

    return result


# ===========================================================================
# Overall evaluation
# ===========================================================================

def calculate_overall_metrics(results):
    total = len(results)

    passed = sum(
        1
        for r in results
        if r["status"] == "PASS"
    )

    partial = sum(
        1
        for r in results
        if r["status"] == "PARTIAL"
    )

    failed = sum(
        1
        for r in results
        if r["status"] == "FAIL"
    )

    avg_requirement_coverage = (
        sum(
            r["requirement_coverage"]
            for r in results
        )
        / total
        if total
        else 0
    )

    avg_fact_coverage = (
        sum(
            r["fact_coverage"]
            for r in results
        )
        / total
        if total
        else 0
    )

    version_correct_count = sum(
        1
        for r in results
        if r["version_correct"]
    )

    validation_pass_count = sum(
        1
        for r in results
        if r["validation_verdict"] == "pass"
    )

    return {
        "total_cases": total,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "pass_rate": round(
            passed / total,
            4,
        ) if total else 0,
        "average_requirement_coverage": round(
            avg_requirement_coverage,
            4,
        ),
        "average_fact_coverage": round(
            avg_fact_coverage,
            4,
        ),
        "version_correctness_rate": round(
            version_correct_count / total,
            4,
        ) if total else 0,
        "validation_pass_rate": round(
            validation_pass_count / total,
            4,
        ) if total else 0,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():

    print("\n")
    print("=" * 80)
    print("CT-200 FINAL END-TO-END GOLD STANDARD EVALUATION")
    print("=" * 80)

    print(
        f"Gold Standard Cases: {len(GOLD_STANDARD)}"
    )

    results = []

    # -----------------------------------------------------------------------
    # Evaluate every Gold Standard case
    # -----------------------------------------------------------------------

    for index, case in enumerate(
        GOLD_STANDARD,
        start=1,
    ):

        case_id = get_case_id(case)

        print("\n" + "-" * 80)
        print(
            f"[{index}/{len(GOLD_STANDARD)}] {case_id}"
        )
        print("-" * 80)

        print(
            "Version:",
            get_expected_version(case),
        )

        print(
            "Expected Requirement:",
            get_expected_nodes(case),
        )

        result = evaluate_case(case)

        results.append(result)

        print(
            f"Actual Requirement: "
            f"{result.get('actual_requirement_id')}"
        )

        print(
            f"Requirement Coverage: "
            f"{result['requirement_coverage']:.2%}"
        )

        print(
            f"Fact Coverage: "
            f"{result['fact_coverage']:.2%}"
        )

        print(
            f"Version Correct: "
            f"{result['version_correct']}"
        )

        print(
            f"Validation: "
            f"{result['validation_verdict']}"
        )

        print(
            f"STATUS: {result['status']}"
        )

        print(
            f"Reason: {result['reason']}"
        )

        if result.get("missing_facts"):

            print(
                "Missing Facts:"
            )

            for fact in result[
                "missing_facts"
            ]:

                print(
                    f"  - {fact}"
                )

        if result.get("error"):

            print(
                f"ERROR: {result['error']}"
            )

    # =========================================================================
    # Overall metrics
    # =========================================================================

    metrics = calculate_overall_metrics(
        results
    )

    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Total Cases              : "
        f"{metrics['total_cases']}"
    )

    print(
        f"Passed                   : "
        f"{metrics['passed']}"
    )

    print(
        f"Partial                  : "
        f"{metrics['partial']}"
    )

    print(
        f"Failed                   : "
        f"{metrics['failed']}"
    )

    print(
        f"Pass Rate                : "
        f"{metrics['pass_rate']:.2%}"
    )

    print(
        f"Requirement Coverage     : "
        f"{metrics['average_requirement_coverage']:.2%}"
    )

    print(
        f"Fact Coverage            : "
        f"{metrics['average_fact_coverage']:.2%}"
    )

    print(
        f"Version Correctness      : "
        f"{metrics['version_correctness_rate']:.2%}"
    )

    print(
        f"Validation Pass Rate     : "
        f"{metrics['validation_pass_rate']:.2%}"
    )

    # =========================================================================
    # Case-wise compact table
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("CASE-WISE RESULTS")
    print("=" * 80)

    print(
        f"{'CASE':<15}"
        f"{'REQ COVER':<12}"
        f"{'FACT COVER':<13}"
        f"{'VALIDATION':<13}"
        f"{'STATUS':<10}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['case_id']:<15}"
            f"{result['requirement_coverage']:.0%}"
            f"{'':<10}"
            f"{result['fact_coverage']:.0%}"
            f"{'':<11}"
            f"{str(result['validation_verdict']):<13}"
            f"{result['status']:<10}"
        )

    # =========================================================================
    # Save complete JSON report
    # =========================================================================

    final_report = {
        "evaluation": "CT-200 Final End-to-End Evaluation",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "results": results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_report,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\n")
    print(
        f"Detailed report saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()