from pydantic import BaseModel
from typing import List


class TraceabilityRow(BaseModel):
    requirement_id: str
    requirement_title: str
    test_case_count: int
    test_cases: List[str]


class CoverageResponse(BaseModel):
    document_id: int
    total_requirements: int
    covered_requirements: int
    uncovered_requirements: int
    coverage_percentage: float