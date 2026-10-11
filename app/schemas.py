from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ============================================================
# LLM Test Case Generation Schemas
# ============================================================

class GeneratedTestCase(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    preconditions: str
    steps: List[str] = Field(min_length=1)
    expected_result: str = Field(min_length=1)
    priority: Literal["High", "Medium", "Low"]


class TestCaseGenerationOutput(BaseModel):
    test_cases: List[GeneratedTestCase] = Field(
        min_length=1,
        max_length=2
    )


# ============================================================
# LLM Validation Schemas
# ============================================================

class TestCaseValidationResult(BaseModel):
    id: str = Field(min_length=1)
    grounded: bool
    misattributed_to: Optional[str] = None
    comment: str = Field(min_length=1)


class ValidationAgentOutput(BaseModel):
    verdict: Literal["pass", "fail"]
    per_test_case: List[TestCaseValidationResult]
    feedback: str


# ============================================================
# API Response Schemas
# ============================================================

class TraceabilityRow(BaseModel):
    requirement_id: str
    requirement_title: str
    is_stale: bool
    test_case_count: int
    test_cases: List[str]


class CoverageResponse(BaseModel):
    document_id: str
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
    document_id: Optional[str] = None
    filename: str
    accepted: bool
    rejection_reason: Optional[str] = None
    version: Optional[str] = None
    version_status: Optional[str] = None
    retry_count: Optional[int] = None
    requirements: Optional[list] = None


class ReportResponse(BaseModel):
    document_id: str
    filename: str
    version: str
    version_status: Optional[str]
    coverage: CoverageResponse
    stale_requirement_ids: List[str]
    failed_validations: List[ValidationSummaryResponse]