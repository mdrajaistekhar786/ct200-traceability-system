from typing import TypedDict, Any, Optional, List, Dict


class QAGraphState(TypedDict, total=False):
    # Input / plumbing
    db: Any
    filename: str
    pdf_path: str

    # Document Gate (CRAG)
    tree: List[dict]
    is_relevant: bool
    relevance_reason: str
    rejected: bool

    # Version Detection
    document: Any
    previous_document: Any
    version_status: str  # "first" | "unchanged" | "changed"

    # Regeneration loop
    retry_count: int
    max_retries: int
    validation_summary: Dict[str, Any]

    error: Optional[str]
