import csv
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.database import create_tables, SessionLocal
from app.models import Document, RequirementNode, TestCase, ValidationResult, Traceability
from app.graph import run_pipeline
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
        "detection, requirement extraction, AI test case generation, "
        "Self-RAG validation with a regeneration loop, and traceability "
        "mapping, backed by coverage/staleness reporting."
    ),
    version="2.0.0",
)

create_tables()


@app.get("/")
def home():
    return {"message": "CT200 Traceability System is running!"}


# -----------------------------------------------------------------------
# Analyze -- runs the full LangGraph pipeline
# -----------------------------------------------------------------------
@app.post(
    "/analyze",
    tags=["QA Analysis"],
    response_model=AnalyzeResponse,
    summary="Upload a PDF and run the full QA analysis pipeline",
)
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        pdf_path = temp_file.name

    db = SessionLocal()

    try:
        final_state = run_pipeline(db=db, filename=file.filename, pdf_path=pdf_path)

        if final_state.get("rejected"):
            return AnalyzeResponse(
                filename=file.filename,
                accepted=False,
                rejection_reason=final_state.get("relevance_reason"),
            )

        document = final_state["document"]

        return AnalyzeResponse(
            document_id=document.id,
            filename=file.filename,
            accepted=True,
            version=document.version,
            version_status=final_state.get("version_status"),
            retry_count=final_state.get("retry_count", 0),
            requirements=final_state.get("tree"),
        )

    finally:
        db.close()


# Backward-compatible alias for the old endpoint name.
@app.post(
    "/generate-test-cases",
    tags=["QA Analysis"],
    response_model=AnalyzeResponse,
    summary="[Deprecated] Alias for /analyze",
)
async def generate_from_pdf(file: UploadFile = File(...)):
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
def get_traceability_matrix(document_id: int):
    db = SessionLocal()

    try:
        requirements = (
            db.query(RequirementNode)
            .filter(RequirementNode.document_id == document_id)
            .all()
        )

        matrix = []

        for req in requirements:
            test_cases = (
                db.query(TestCase)
                .filter(TestCase.requirement_node_id == req.node_id)
                .all()
            )

            matrix.append({
                "requirement_id": req.node_id,
                "requirement_title": req.title,
                "is_stale": bool(req.is_stale),
                "test_case_count": len(test_cases),
                "test_cases": [tc.test_case_id for tc in test_cases],
            })

        return matrix

    finally:
        db.close()


# -----------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------
@app.get(
    "/coverage/{document_id}",
    tags=["Coverage"],
    response_model=CoverageResponse,
    summary="Get Coverage",
)
def get_coverage(document_id: int):
    db = SessionLocal()

    try:
        requirements = (
            db.query(RequirementNode)
            .filter(RequirementNode.document_id == document_id)
            .all()
        )

        total_requirements = len(requirements)
        covered = 0

        for req in requirements:
            count = (
                db.query(TestCase)
                .filter(TestCase.requirement_node_id == req.node_id)
                .count()
            )
            if count > 0:
                covered += 1

        coverage = (covered / total_requirements * 100) if total_requirements > 0 else 0

        return {
            "document_id": document_id,
            "total_requirements": total_requirements,
            "covered_requirements": covered,
            "uncovered_requirements": total_requirements - covered,
            "coverage_percentage": round(coverage, 2),
        }

    finally:
        db.close()


# -----------------------------------------------------------------------
# Validation results (Self-RAG)
# -----------------------------------------------------------------------
@app.get(
    "/validation/{document_id}",
    tags=["Validation"],
    response_model=List[ValidationSummaryResponse],
    summary="Get Self-RAG validation results for a document",
)
def get_validation_results(document_id: int):
    db = SessionLocal()

    try:
        node_ids = [
            n.node_id
            for n in db.query(RequirementNode).filter(RequirementNode.document_id == document_id).all()
        ]

        results = (
            db.query(ValidationResult)
            .filter(ValidationResult.requirement_node_id.in_(node_ids))
            .order_by(ValidationResult.requirement_node_id, ValidationResult.attempt)
            .all()
        )

        return [
            {
                "requirement_id": r.requirement_node_id,
                "verdict": r.verdict,
                "attempt": r.attempt,
                "feedback": r.feedback,
                "related_requirement_ids": (
                    r.related_requirement_ids.split(",") if r.related_requirement_ids else []
                ),
            }
            for r in results
        ]

    finally:
        db.close()


# -----------------------------------------------------------------------
# Combined QA report
# -----------------------------------------------------------------------
@app.get(
    "/reports/{document_id}",
    tags=["Reports"],
    response_model=ReportResponse,
    summary="Get the final QA report (coverage + stale + failed validations)",
)
def get_report(document_id: int):
    db = SessionLocal()

    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        coverage = get_coverage(document_id)

        stale_ids = [
            n.node_id
            for n in db.query(RequirementNode)
            .filter(RequirementNode.document_id == document_id, RequirementNode.is_stale.is_(True))
            .all()
        ]

        node_ids = [n.node_id for n in document.nodes]

        failed = (
            db.query(ValidationResult)
            .filter(
                ValidationResult.requirement_node_id.in_(node_ids),
                ValidationResult.verdict == "fail",
            )
            .all()
        )

        return {
            "document_id": document.id,
            "filename": document.name,
            "version": document.version,
            "version_status": document.version_status,
            "coverage": coverage,
            "stale_requirement_ids": stale_ids,
            "failed_validations": [
                {
                    "requirement_id": f.requirement_node_id,
                    "verdict": f.verdict,
                    "attempt": f.attempt,
                    "feedback": f.feedback,
                    "related_requirement_ids": (
                        f.related_requirement_ids.split(",") if f.related_requirement_ids else []
                    ),
                }
                for f in failed
            ],
        }

    finally:
        db.close()


# -----------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------
@app.get(
    "/export/{document_id}",
    tags=["Export"],
    summary="Export Traceability Matrix as CSV",
)
def export_traceability_matrix(document_id: int):
    db = SessionLocal()

    try:
        filename = f"traceability_matrix_{document_id}.csv"

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([
                "Requirement ID",
                "Requirement Title",
                "Is Stale",
                "Test Case Count",
                "Test Cases",
            ])

            requirements = (
                db.query(RequirementNode)
                .filter(RequirementNode.document_id == document_id)
                .all()
            )

            for req in requirements:
                test_cases = (
                    db.query(TestCase)
                    .filter(TestCase.requirement_node_id == req.node_id)
                    .all()
                )

                writer.writerow([
                    req.node_id,
                    req.title,
                    req.is_stale,
                    len(test_cases),
                    ", ".join(tc.test_case_id for tc in test_cases),
                ])

        return FileResponse(filename, media_type="text/csv", filename=filename)

    finally:
        db.close()


# -----------------------------------------------------------------------
# Requirements lookup
# -----------------------------------------------------------------------
@app.get(
    "/requirements/{node_id}",
    tags=["Requirements"],
    summary="Get Requirement by ID",
)
def get_requirement(node_id: str):
    db = SessionLocal()

    try:
        node = db.query(RequirementNode).filter(RequirementNode.node_id == node_id).first()

        if not node:
            raise HTTPException(status_code=404, detail="Requirement not found")

        return {
            "node_id": node.node_id,
            "title": node.title,
            "text": node.text,
            "page": node.page,
            "version": node.document.version,
            "is_stale": node.is_stale,
        }

    finally:
        db.close()


@app.get(
    "/requirements/search/{keyword}",
    tags=["Requirements"],
    summary="Search Requirements",
)
def search_requirements(keyword: str):
    db = SessionLocal()

    try:
        results = (
            db.query(RequirementNode)
            .filter(RequirementNode.text.ilike(f"%{keyword}%"))
            .all()
        )

        return [
            {
                "node_id": r.node_id,
                "title": r.title,
                "page": r.page,
                "version": r.document.version,
                "is_stale": r.is_stale,
            }
            for r in results
        ]

    finally:
        db.close()
