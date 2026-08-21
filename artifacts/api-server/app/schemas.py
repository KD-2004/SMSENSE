from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

Label = Literal["LOW", "SUSPICIOUS", "HIGH RISK", "CRITICAL"]


class AnalyzeRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    sender: str | None = Field(default=None, max_length=320)
    timestamp: datetime | None = None


class EvidenceGroups(BaseModel):
    content: float = 0
    sender: float = 0
    url: float = 0
    behavior: float = 0
    obfuscation: float = 0
    similarity: float = 0


class HealthComponent(BaseModel):
    available: bool
    detail: str


class HealthResponse(BaseModel):
    status: str
    components: dict[str, HealthComponent]


class AnalysisResponse(BaseModel):
    id: int | None = None
    risk_score: int = Field(ge=0, le=100)
    label: Label
    category: str
    language: str
    raw_message: str
    normalized_message: str
    evidence: EvidenceGroups
    suspicious_phrases: list[str]
    url_analysis: dict[str, Any]
    obfuscation: dict[str, Any]
    explanation: dict[str, Any]
    post_hoc: dict[str, Any] | None = None
    created_at: datetime | None = None


class FeedbackRequest(BaseModel):
    analysis_id: int
    helpful: bool
    note: str | None = Field(default=None, max_length=2_000)


class AdversarialRequest(BaseModel):
    messages: list[AnalyzeRequest] = Field(min_length=1, max_length=20)


class AdversarialResponse(BaseModel):
    results: list[AnalysisResponse]
    invariants: list[str]
