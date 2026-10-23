from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

Label = Literal["LOW", "SUSPICIOUS", "HIGH RISK", "CRITICAL"]


class AnalyzeRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    sender: str | None = Field(default=None, max_length=320)
    timestamp: datetime | None = None

    @field_validator("message")
    @classmethod
    def reject_blank_messages(cls, value: str) -> str:
        """Reject blank-only input without mutating a valid original SMS."""
        if not value.strip():
            raise ValueError("message must contain at least one non-whitespace character")
        return value


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
    sender: str | None = None
    model_assessment: dict[str, Any] = Field(default_factory=dict)
    decision_diagnostics: dict[str, Any] = Field(default_factory=dict)
    similarity: dict[str, Any] = Field(default_factory=dict)
    explanation_status: dict[str, Any] | None = None
    evidence: EvidenceGroups
    suspicious_phrases: list[str]
    url_analysis: dict[str, Any]
    obfuscation: dict[str, Any]
    explanation: dict[str, Any]
    post_hoc: dict[str, Any] | None = None
    created_at: datetime | None = None


class ExplainRequest(BaseModel):
    """Bounded input for an optional explanation generated after risk analysis."""

    risk_score: int = Field(ge=0, le=100)
    label: Label
    category: str = Field(min_length=1, max_length=120)
    top_evidence: list[str] = Field(default_factory=list, max_length=12)
    raw_message: str = Field(min_length=1, max_length=10_000)
    normalized_message: str = Field(default="", max_length=10_000)
    suspicious_phrases: list[str] = Field(default_factory=list, max_length=40)

    @field_validator("raw_message")
    @classmethod
    def reject_blank_explanation_messages(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_message must contain non-whitespace text")
        return value


class ExplanationResponse(BaseModel):
    summary: str
    suspicious_phrases: list[str]
    recommended_action: str
    available: bool
    detail: str


class FeedbackRequest(BaseModel):
    analysis_id: int
    helpful: bool
    note: str | None = Field(default=None, max_length=2_000)


class AdversarialRequest(BaseModel):
    messages: list[AnalyzeRequest] = Field(min_length=1, max_length=20)


class AdversarialResponse(BaseModel):
    results: list[AnalysisResponse]
    invariants: list[str]
