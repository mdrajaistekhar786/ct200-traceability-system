from pydantic import BaseModel
from typing import List, Optional


class TraceabilityRow(BaseModel):
    requirement_id: str
    requirement_title: str
    is_stale: bool
    test_case_count: int
    test_cases: List[str]


class CoverageResponse(BaseModel):
    document_id: int
    total_requirements: int
    covered_requirements: int
    uncovered_requirements: int
    coverage_percentage: float


class ValidationSummaryResponse(BaseModel):
    requirement_id: str
    verdict: str
    attempt: int
    feedback: Optional[str] = None
    related_requirement_ids: Optional[List[str]] = None


class AnalyzeResponse(BaseModel):
    document_id: Optional[int] = None
    filename: str
    accepted: bool
    rejection_reason: Optional[str] = None
    version: Optional[str] = None
    version_status: Optional[str] = None
    retry_count: Optional[int] = None
    requirements: Optional[list] = None


class ReportResponse(BaseModel):
    document_id: int
    filename: str
    version: str
    version_status: Optional[str]
    coverage: CoverageResponse
    stale_requirement_ids: List[str]
    failed_validations: List[ValidationSummaryResponse]
