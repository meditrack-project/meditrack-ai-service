from typing import Optional
from pydantic import BaseModel, Field


class MedicationSummaryRequest(BaseModel):
    days: Optional[int] = Field(default=30, ge=1, le=365)


class SymptomAnalysisRequest(BaseModel):
    days: Optional[int] = Field(default=14, ge=1, le=365)


class InsightsRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    days: Optional[int] = Field(default=7, ge=1, le=365)
