import csv
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.database import (
    create_tables,
    documents_collection,
    requirements_collection,
    test_cases_collection,
    validation_results_collection,
)

from app.graph import (
    run_pipeline,
    run_selected_requirement,
)

from app.schemas import (
    TraceabilityRow,
    CoverageResponse,
    AnalyzeResponse,
    ReportResponse,
    ValidationSummaryResponse,
)


app = FastAPI(
    title="CT200 Traceability System",
    description=(
        "LangGraph-orchestrated pipeline: CRAG document gate, version "
        "detection, requirement extraction, selected-section AI test case "
        "generation, Self-RAG validation with a regeneration loop, and "
        "traceability mapping, backed by coverage/staleness reporting."
    ),
    version="2.0.0",
)


create_tables()


# -----------------------------------------------------------------------
# Analyze / Document Ingestion
# -----------------------------------------------------------------------

@app.post(
    "/analyze",
    tags=["QA Analysis"],
    response_model=AnalyzeResponse,
    summary="Upload a PDF and extract requirements",
)
async def analyze(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(
            await file.read()
        )

        pdf_path = temp_file.name

    db = create_tables()

    try:

        final_state = run_pipeline(
            db=db,
            filename=file.filename,
            pdf_path=pdf_path,
        )

        # -------------------------------------------------------------------
        # Rejected document
        # -------------------------------------------------------------------

        if final_state.get("rejected"):

            return AnalyzeResponse(
                filename=file.filename,
                accepted=False,
                rejection_reason=(
                    final_state.get(
                        "relevance_reason"
                    )
                ),
            )

        # -------------------------------------------------------------------
        # Document returned by identity/version pipeline
        # -------------------------------------------------------------------

        document = final_state.get(
            "document"
        )

        if document is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Document processing completed "
                    "without a document."
                ),
            )

        # -------------------------------------------------------------------
        # Duplicate upload
        # -------------------------------------------------------------------

        is_duplicate = bool(
            final_state.get(
                "is_duplicate",
                False,
            )
        )

        # -------------------------------------------------------------------
        # For a duplicate, requirements are already stored in MongoDB.
        #
        # Do not return an empty requirement list just because the
        # parser was intentionally skipped.
        # -------------------------------------------------------------------

        if is_duplicate:

            stored_requirements = list(
                requirements_collection.find(
                    {
                        "document_id": document.id
                    }
                )
            )

            requirements = [
                {
                    "id": requirement.get(
                        "node_id"
                    ),
                    "number": requirement.get(
                        "number"
                    ),
                    "title": requirement.get(
                        "title"
                    ),
                    "parent": requirement.get(
                        "parent_node_id"
                    ),
                    "source": {
                        "page": requirement.get(
                            "page"
                        ),
                        "y": requirement.get(
                            "source_y"
                        ),
                    },
                    "text": (
                        [requirement["text"]]
                        if requirement.get("text")
                        else []
                    ),
                    "tables": requirement.get(
                        "tables",
                        [],
                    ),
                    "children": [],
                    "test_cases": [],
                    "version": document.version,
                    "content_hash": requirement.get(
                        "content_hash"
                    ),
                }
                for requirement in stored_requirements
            ]

        else:

            requirements = final_state.get(
                "tree",
                []
            )

        # -------------------------------------------------------------------
        # Return result
        # -------------------------------------------------------------------

        return AnalyzeResponse(
            document_id=document.id,
            filename=file.filename,
            accepted=True,
            version=document.version,
            version_status=(
                "duplicate"
                if is_duplicate
                else final_state.get(
                    "version_status"
                )
            ),
            retry_count=(
                final_state.get(
                    "retry_count",
                    0,
                )
            ),
            requirements=requirements,
        )

    finally:

        pass


# -----------------------------------------------------------------------
# Selected Requirement Test Case Generation
# -----------------------------------------------------------------------

@app.post(
    "/documents/{document_id}/requirements/{node_id}/generate-test-cases",
    tags=["QA Analysis"],
    summary="Generate and validate test cases for a selected requirement",
)
def generate_test_cases_for_requirement(
    document_id: str,
    node_id: str,
):

    db = create_tables()

    document = documents_collection.find_one(
        {
            "_id": _to_object_id(
                document_id
            )
        }
    )

    if not document:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Document {document_id} "
                "not found"
            ),
        )

    requirement = requirements_collection.find_one(
        {
            "document_id": document_id,
            "node_id": node_id,
        }
    )

    if not requirement:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Requirement {node_id} "
                f"not found in document "
                f"{document_id}"
            ),
        )

    final_state = run_selected_requirement(
        db=db,
        document_id=document_id,
        node_id=node_id,
    )

    selected_requirement = (
        final_state.get(
            "selected_requirement",
            {},
        )
    )

    validation_result = (
        final_state.get(
            "validation_result",
            {},
        )
    )

    for tc in selected_requirement.get("test_cases", []):
        tc.pop("_id", None)

    return {
        "document_id": document_id,
        "requirement_id": node_id,
        "requirement_title": requirement.get(
            "title"
        ),
        "validation": validation_result,
        "retry_count": (
            final_state.get(
                "retry_count",
                0,
            )
        ),
        "test_cases": (
            selected_requirement.get(
                "test_cases",
                [],
            )
        ),
        "traceability_created": (
            validation_result.get("verdict") == "pass"
            or bool(selected_requirement.get("test_cases"))
        ),
    }


# -----------------------------------------------------------------------
# Backward-compatible alias
# -----------------------------------------------------------------------

@app.post(
    "/generate-test-cases",
    tags=["QA Analysis"],
    response_model=AnalyzeResponse,
    summary="[Deprecated] Alias for /analyze",
)
async def generate_from_pdf(
    file: UploadFile = File(...)
):
    return await analyze(file)


# -----------------------------------------------------------------------
# Traceability
# -----------------------------------------------------------------------

@app.get(
    "/traceability-matrix/{document_id}",
    tags=["Traceability"],
    response_model=List[TraceabilityRow],
    summary="Get Traceability Matrix",
)
def get_traceability_matrix(
    document_id: str
):

    requirements = list(
        requirements_collection.find(
            {
                "document_id": document_id
            }
        )
    )

    matrix = []

    for req in requirements:

        test_cases = list(
            test_cases_collection.find(
                {
                    "document_id": document_id,
                    "requirement_node_id": req[
                        "node_id"
                    ],
                }
            )
        )

        matrix.append(
            {
                "requirement_id": req[
                    "node_id"
                ],
                "requirement_title": req.get(
                    "title"
                ),
                "is_stale": bool(
                    req.get(
                        "is_stale",
                        False,
                    )
                ),
                "test_case_count": len(
                    test_cases
                ),
                "test_cases": [
                    tc.get(
                        "test_case_id"
                    )
                    for tc in test_cases
                ],
            }
        )

    return matrix


# -----------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------

@app.get(
    "/coverage/{document_id}",
    tags=["Coverage"],
    response_model=CoverageResponse,
    summary="Get Coverage",
)
def get_coverage(
    document_id: str
):

    requirements = list(
        requirements_collection.find(
            {
                "document_id": document_id
            }
        )
    )

    total_requirements = len(
        requirements
    )

    covered = 0

    for req in requirements:

        count = test_cases_collection.count_documents(
            {
                "document_id": document_id,
                "requirement_node_id": req[
                    "node_id"
                ],
            }
        )

        if count > 0:
            covered += 1

    coverage = (
        covered / total_requirements * 100
        if total_requirements > 0
        else 0
    )

    return {
        "document_id": document_id,
        "total_requirements": total_requirements,
        "covered_requirements": covered,
        "uncovered_requirements": (
            total_requirements - covered
        ),
        "coverage_percentage": round(
            coverage,
            2,
        ),
    }


# -----------------------------------------------------------------------
# Validation Results
# -----------------------------------------------------------------------

@app.get(
    "/validation/{document_id}",
    tags=["Validation"],
    response_model=List[
        ValidationSummaryResponse
    ],
    summary="Get Self-RAG validation results for a document",
)
def get_validation_results(
    document_id: str
):

    node_ids = [
        n["node_id"]
        for n in requirements_collection.find(
            {
                "document_id": document_id
            }
        )
    ]

    results = list(
        validation_results_collection.find(
            {
                "requirement_node_id": {
                    "$in": node_ids
                },
                "is_latest": True,
            }
        ).sort(
            [
                (
                    "requirement_node_id",
                    1,
                ),
                (
                    "attempt",
                    -1,
                ),
            ]
        )
    )

    return [
        {
            "requirement_id": (
                r.get(
                    "requirement_node_id"
                )
            ),
            "verdict": r.get(
                "verdict"
            ),
            "attempt": r.get(
                "attempt",
                1,
            ),
            "feedback": r.get(
                "feedback"
            ),
            "related_requirement_ids": (
                r.get(
                    "related_requirement_ids",
                    [],
                )
                if isinstance(
                    r.get(
                        "related_requirement_ids",
                        [],
                    ),
                    list,
                )
                else (
                    r.get(
                        "related_requirement_ids"
                    ).split(",")
                    if r.get(
                        "related_requirement_ids"
                    )
                    else []
                )
            ),
        }
        for r in results
    ]


# -----------------------------------------------------------------------
# Combined QA Report
# -----------------------------------------------------------------------

@app.get(
    "/reports/{document_id}",
    tags=["Reports"],
    response_model=ReportResponse,
    summary="Get the final QA report",
)
def get_report(
    document_id: str
):

    document = documents_collection.find_one(
        {
            "_id": _to_object_id(
                document_id
            )
        }
    )

    if not document:

        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    coverage = get_coverage(
        document_id
    )

    stale_ids = [
        n["node_id"]
        for n in requirements_collection.find(
            {
                "document_id": document_id,
                "is_stale": True,
            }
        )
    ]

    node_ids = [
        n["node_id"]
        for n in requirements_collection.find(
            {
                "document_id": document_id
            }
        )
    ]

    failed = list(
        validation_results_collection.find(
            {
                "requirement_node_id": {
                    "$in": node_ids
                },
                "verdict": "fail",
                "is_latest": True,
            }
        )
    )

    return {
        "document_id": document_id,
        "filename": document.get(
            "name"
        ),
        "version": document.get(
            "version"
        ),
        "version_status": document.get(
            "version_status"
        ),
        "coverage": coverage,
        "stale_requirement_ids": stale_ids,
        "failed_validations": [
            {
                "requirement_id": (
                    f.get(
                        "requirement_node_id"
                    )
                ),
                "verdict": f.get(
                    "verdict"
                ),
                "attempt": f.get(
                    "attempt",
                    1,
                ),
                "feedback": f.get(
                    "feedback"
                ),
                "related_requirement_ids": (
                    f.get(
                        "related_requirement_ids",
                        [],
                    )
                    if isinstance(
                        f.get(
                            "related_requirement_ids",
                            [],
                        ),
                        list,
                    )
                    else (
                        f.get(
                            "related_requirement_ids"
                        ).split(",")
                        if f.get(
                            "related_requirement_ids"
                        )
                        else []
                    )
                ),
            }
            for f in failed
        ],
    }


# -----------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------

@app.get(
    "/export/{document_id}",
    tags=["Export"],
    summary="Export Traceability Matrix as CSV",
)
def export_traceability_matrix(
    document_id: str
):

    filename = (
        f"traceability_matrix_"
        f"{document_id}.csv"
    )

    requirements = list(
        requirements_collection.find(
            {
                "document_id": document_id
            }
        )
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.writer(
            csvfile
        )

        writer.writerow(
            [
                "Requirement ID",
                "Requirement Title",
                "Is Stale",
                "Test Case Count",
                "Test Cases",
            ]
        )

        for req in requirements:

            test_cases = list(
                test_cases_collection.find(
                    {
                        "document_id": document_id,
                        "requirement_node_id": req[
                            "node_id"
                        ],
                    }
                )
            )

            writer.writerow(
                [
                    req.get(
                        "node_id"
                    ),
                    req.get(
                        "title"
                    ),
                    req.get(
                        "is_stale",
                        False,
                    ),
                    len(test_cases),
                    ", ".join(
                        tc.get(
                            "test_case_id",
                            "",
                        )
                        for tc in test_cases
                    ),
                ]
            )

    return FileResponse(
        filename,
        media_type="text/csv",
        filename=filename,
    )


# -----------------------------------------------------------------------
# Requirements Lookup
# -----------------------------------------------------------------------

@app.get(
    "/requirements/{node_id}",
    tags=["Requirements"],
    summary="Get Requirement by ID",
)
def get_requirement(
    node_id: str
):

    node = requirements_collection.find_one(
        {
            "node_id": node_id
        }
    )

    if not node:

        raise HTTPException(
            status_code=404,
            detail="Requirement not found",
        )

    document = documents_collection.find_one(
        {
            "_id": _to_object_id(
                node["document_id"]
            )
        }
    )

    if not document:

        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return {
        "node_id": node.get(
            "node_id"
        ),
        "title": node.get(
            "title"
        ),
        "text": node.get(
            "text"
        ),
        "page": node.get(
            "page"
        ),
        "version": document.get(
            "version"
        ),
        "is_stale": node.get(
            "is_stale",
            False,
        ),
    }


@app.get(
    "/requirements/search/{keyword}",
    tags=["Requirements"],
    summary="Search Requirements",
)
def search_requirements(
    keyword: str
):

    results = requirements_collection.find(
        {
            "text": {
                "$regex": keyword,
                "$options": "i",
            }
        }
    )

    response = []

    for r in results:

        document = documents_collection.find_one(
            {
                "_id": _to_object_id(
                    r["document_id"]
                )
            }
        )

        response.append(
            {
                "node_id": r.get(
                    "node_id"
                ),
                "title": r.get(
                    "title"
                ),
                "page": r.get(
                    "page"
                ),
                "version": (
                    document.get(
                        "version"
                    )
                    if document
                    else None
                ),
                "is_stale": r.get(
                    "is_stale",
                    False,
                ),
            }
        )

    return response


# -----------------------------------------------------------------------
# MongoDB ID helper
# -----------------------------------------------------------------------

def _to_object_id(
    document_id: str,
):
    """
    Convert string document ID to MongoDB ObjectId.
    """

    from bson import ObjectId

    try:

        return ObjectId(
            document_id
        )

    except Exception:

        return document_id