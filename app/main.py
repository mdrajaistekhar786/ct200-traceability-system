from fastapi import FastAPI, UploadFile, File
import tempfile
import hashlib
from fastapi import HTTPException

from app.parser import parse_manual
from app.llm import generate_test_cases
from app.database import create_tables, SessionLocal
from app.models import Document, RequirementNode, TestCase
from sqlalchemy.orm import Session
import csv
from fastapi.responses import FileResponse
from app.schemas import TraceabilityRow, CoverageResponse
from typing import List

app = FastAPI(
    title="CT200 Traceability System",
    description="AI-powered requirement extraction, automated test case generation, traceability matrix creation, coverage analysis, and CSV export.",
    version="1.0.0"
)

create_tables()


@app.get("/")
def home():
    return {"message": "CT200 Traceability System is running!"}


# -----------------------------------
# Save Requirement Nodes
# -----------------------------------
def save_nodes(db, nodes, document_id, parent_id=None):

    for node in nodes:

        db_node = RequirementNode(
            node_id=node["id"],
            document_id=document_id,
            parent_node_id=parent_id,
            number=node["number"],
            title=node["title"],
            text="\n".join(node.get("text", [])),
            content_hash=node.get("content_hash"),
            page=node["source"]["page"]
        )

        db.add(db_node)
        db.commit()
        db.refresh(db_node)

        if node.get("children"):
            save_nodes(
                db=db,
                nodes=node["children"],
                document_id=document_id,
                parent_id=db_node.node_id
            )

def mark_stale_requirements(db, document):

    if document.version == "v1":
        return

    previous_version = f"v{int(document.version[1:]) - 1}"

    previous_doc = (
        db.query(Document)
        .filter(
            Document.name == document.name,
            Document.version == previous_version
        )
        .first()
    )

    if not previous_doc:
        return

    previous_nodes = {
        n.node_id: n
        for n in previous_doc.nodes
    }

    current_nodes = (
        db.query(RequirementNode)
        .filter(
            RequirementNode.document_id == document.id
        )
        .all()
    )

    for node in current_nodes:

        old = previous_nodes.get(node.node_id)

        if old and old.content_hash != node.content_hash:
            node.is_stale = True
        else:
            node.is_stale = False

    db.commit()
# -----------------------------------
# Save Test Cases
# -----------------------------------
def save_test_cases(db, requirement_node_id, test_cases):

    for tc in test_cases:

        try:
            db_test_case = TestCase(
                requirement_node_id=requirement_node_id,
                test_case_id=tc["id"],
                title=tc["title"],
                preconditions=tc["preconditions"],
                steps="\n".join(tc["steps"]),
                expected_result=tc["expected_result"],
                priority=tc["priority"]
            )

            db.add(db_test_case)
            db.commit()

            print(f"Saved {tc['id']}")

        except Exception as e:
            db.rollback()
            print("DATABASE ERROR:", e)



@app.post(
    "/generate-test-cases",
    tags=["Test Case Generation"],
    summary="Generate Test Cases from PDF"
)
async def generate_from_pdf(file: UploadFile = File(...)):

    # Save uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        pdf_path = temp_file.name

    # Parse PDF
    tree = parse_manual(pdf_path)

    # Database session
    db = SessionLocal()

    try:

        # --------------------------
        # Save Document
        # --------------------------
        latest = (
            db.query(Document)
            .filter(Document.name == file.filename)
            .order_by(Document.id.desc())
            .first()
        )

        if latest:
            version_no = int(latest.version.replace("v", "")) + 1
            version = f"v{version_no}"
        else:
            version = "v1"

        document = Document(
            name=file.filename,
            version=version
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        # --------------------------
# Save Requirement Nodes
# --------------------------
        if tree:
            save_nodes(
            db=db,
            nodes=tree,
            document_id=document.id
    )

            mark_stale_requirements(db, document)

        # --------------------------
        # Generate & Save Test Cases
        # --------------------------
        def generate_for_tree(nodes):

            for node in nodes:

                if node.get("text") or node.get("tables"):

                    node["test_cases"] = generate_test_cases(node)

                    print(node["id"], len(node["test_cases"]))

                    save_test_cases(
                        db=db,
                        requirement_node_id=node["id"],
                        test_cases=node["test_cases"]
                    )

                else:
                    node["test_cases"] = []

                if node.get("children"):
                    generate_for_tree(node["children"])

        if tree:
            generate_for_tree(tree)

        return {
            "requirements": tree
        }

    finally:
        db.close()
@app.get(
    "/traceability-matrix/{document_id}",
    tags=["Traceability"],
    response_model=List[TraceabilityRow],
    summary="Get Traceability Matrix"

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
                "test_case_count": len(test_cases),
                "test_cases": [
                    tc.test_case_id for tc in test_cases
                ]
            })

        return matrix

    finally:
        db.close()

@app.get(
    "/coverage/{document_id}",
    tags=["Coverage"],
    response_model=CoverageResponse,
    summary="Get Coverage"
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

        coverage = (
            covered / total_requirements * 100
            if total_requirements > 0 else 0
        )

        return {
            "document_id": document_id,
            "total_requirements": total_requirements,
            "covered_requirements": covered,
            "uncovered_requirements": total_requirements - covered,
            "coverage_percentage": round(coverage, 2)
        }

    finally:
        db.close()




@app.get(
    "/export/{document_id}",
    tags=["Export"],
    summary="Export Traceability Matrix"
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
                "Test Case Count",
                "Test Cases"
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
                    len(test_cases),
                    ", ".join(tc.test_case_id for tc in test_cases)
                ])

        return FileResponse(
            filename,
            media_type="text/csv",
            filename=filename
        )

    finally:
        db.close()


@app.get(
    "/requirements/{node_id}",
    tags=["Requirements"],
    summary="Get Requirement by ID"
)
def get_requirement(node_id: str):

    db = SessionLocal()

    try:
        node = (
            db.query(RequirementNode)
            .filter(RequirementNode.node_id == node_id)
            .first()
        )

        if not node:
            raise HTTPException(
                status_code=404,
                detail="Requirement not found"
            )

        return {
            "node_id": node.node_id,
            "title": node.title,
            "text": node.text,
            "page": node.page,
            "version": node.document.version,
            "is_stale": node.is_stale
        }

    finally:
        db.close()


@app.get(
    "/requirements/search/{keyword}",
    tags=["Requirements"],
    summary="Search Requirements"
)
def search_requirements(keyword: str):

    db = SessionLocal()

    try:

        results = (
            db.query(RequirementNode)
            .filter(
                RequirementNode.text.ilike(f"%{keyword}%")
            )
            .all()
        )

        return [
            {
                "node_id": r.node_id,
                "title": r.title,
                "page": r.page,
                "version": r.document.version,
                "is_stale": r.is_stale
            }
            for r in results
        ]

    finally:
        db.close()

