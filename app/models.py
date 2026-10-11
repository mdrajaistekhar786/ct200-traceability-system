"""
MongoDB data models for the CT-200 Traceability System.

These Pydantic models define the structure of documents stored
in MongoDB collections.
"""

from datetime import datetime
from typing import Optional, List, Any

from pydantic import BaseModel, Field


# ===========================================================================
# Document
# ===========================================================================

class Document(BaseModel):
    """
    Represents an uploaded CT-200 document/version.

    Document identity is based on the SHA-256 hash of the actual
    PDF content, not on the filename.

    This allows the system to:
        - detect exact duplicate uploads
        - reuse the existing document_id
        - prevent duplicate processing
        - maintain stable version history
    """

    # MongoDB document ID
    id: Optional[str] = None

    # Original uploaded filename
    name: str

    # Logical document version
    # Example: v1, v2
    version: str

    # SHA-256 hash of the actual PDF bytes
    content_hash: str

    # ID of the immediately previous actual document version
    previous_document_id: Optional[str] = None

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    # Set by the Document Gate (CRAG) node
    is_relevant: bool = True
    relevance_reason: Optional[str] = None

    # "first" | "unchanged" | "changed" | "duplicate"
    version_status: Optional[str] = None


# ===========================================================================
# Requirement Node
# ===========================================================================

class RequirementNode(BaseModel):
    """
    Represents a requirement node belonging to a document version.

    Stores the structured information extracted by the PDF parser,
    including requirement text, tables, and source location.
    """

    id: Optional[str] = None

    # Example: REQ-2.1.3
    node_id: str

    # MongoDB document ID
    document_id: str

    # Parent requirement in the hierarchy
    parent_node_id: Optional[str] = None

    number: Optional[str] = None
    title: Optional[str] = None
    level: Optional[int] = None

    # Extracted requirement text
    text: Optional[str] = None

    # Structured tables extracted from the PDF
    tables: Optional[List[Any]] = None

    # Original PDF source location
    page: Optional[int] = None
    source_y: Optional[float] = None

    # Used for version comparison
    content_hash: Optional[str] = None

    # Set when compared against the previous document version
    is_stale: bool = False


# ===========================================================================
# Test Case
# ===========================================================================

class TestCase(BaseModel):
    """
    Represents a generated test case.

    Version-aware fields ensure that each test case can be traced
    back to the exact document version and requirement content
    used during generation.
    """

    id: Optional[str] = None

    requirement_node_id: str
    document_id: str

    source_version: str
    source_content_hash: str

    test_case_id: Optional[str] = None
    title: Optional[str] = None
    preconditions: Optional[str] = None
    steps: Optional[str] = None
    expected_result: Optional[str] = None
    priority: Optional[str] = None

    # HITL + validation status
    validation_status: str = "PENDING"
    human_approval: str = "PENDING"

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )


# ===========================================================================
# Validation Result
# ===========================================================================

class ValidationResult(BaseModel):
    """
    Stores Self-RAG validation results for generated test cases.
    """

    id: Optional[str] = None

    requirement_node_id: str

    verdict: str
    feedback: Optional[str] = None
    evidence_detail: Optional[str] = None

    # Cross-requirement evidence
    related_requirement_ids: Optional[List[str]] = None

    attempt: int = 1

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )


# ===========================================================================
# Traceability
# ===========================================================================

class Traceability(BaseModel):
    """
    Maps a requirement to its generated test case.
    """

    id: Optional[str] = None

    document_id: str

    requirement_node_id: str

    test_case_id: str

    status: str = "active"

    updated_at: datetime = Field(
        default_factory=datetime.utcnow
    )