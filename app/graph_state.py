from typing import TypedDict, Any, Optional, List, Dict


class QAGraphState(TypedDict, total=False):
    # Input / plumbing
    db: Any
    filename: str
    pdf_path: str

    # -----------------------------------------------------------------------
    # Document Identity
    # -----------------------------------------------------------------------
    # Unique identity of the uploaded PDF/document.
    document_id: Optional[str]

    # SHA-256 hash of the actual PDF content.
    # Used to detect duplicate uploads regardless of filename.
    content_hash: Optional[str]

    # True when the uploaded PDF already exists in MongoDB.
    is_duplicate: bool

    # -----------------------------------------------------------------------
    # Document Gate (CRAG)
    # -----------------------------------------------------------------------
    tree: List[dict]
    is_relevant: bool
    relevance_reason: str
    rejected: bool

    # -----------------------------------------------------------------------
    # Version Detection
    # -----------------------------------------------------------------------
    document: Any
    previous_document: Any

    # MongoDB document_id of the immediately previous
    # actual document version.
    previous_document_id: Optional[str]

    # "first" | "unchanged" | "changed"
    version_status: str

    # -----------------------------------------------------------------------
    # Selected Requirement
    # -----------------------------------------------------------------------
    selected_requirement: Dict[str, Any]
    selected_test_cases: List[dict]
    cache_hit:bool

    # -----------------------------------------------------------------------
    # Retrieval / Evidence
    # -----------------------------------------------------------------------
    evidence_pack: Dict[str, Any]

    # -----------------------------------------------------------------------
    # Validation / Regeneration
    # -----------------------------------------------------------------------
    validation_result: Dict[str, Any]
    validation_feedback: str
    retry_count: int
    max_retries: int
    validation_summary: Dict[str, Any]

    # -----------------------------------------------------------------------
    # Error handling
    # -----------------------------------------------------------------------
    error: Optional[str]