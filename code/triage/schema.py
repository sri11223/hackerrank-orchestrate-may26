"""Pydantic schemas shared by the deterministic triage pipeline stages."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrapTag(str, Enum):
    """Fixed Stage 2 trap taxonomy from ARCHITECTURE.md.

    The enum values intentionally match the architecture document exactly so
    logs, traces, handler dispatch, and interview artifacts all speak the same
    language.
    """

    ACTION_REQUEST = "ACTION_REQUEST"
    SECURITY_DISCLOSURE = "SECURITY_DISCLOSURE"
    IDENTITY_FRAUD = "IDENTITY_FRAUD"
    PAYMENT_DISPUTE = "PAYMENT_DISPUTE"
    SCORE_DISPUTE = "SCORE_DISPUTE"
    ADMIN_ACTION = "ADMIN_ACTION"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    SYSTEM_HARM = "SYSTEM_HARM"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    COURTESY = "COURTESY"
    THIRD_PARTY = "THIRD_PARTY"
    NORMAL_FAQ = "NORMAL_FAQ"


TRAP_PRIORITY: tuple[TrapTag, ...] = (
    TrapTag.SYSTEM_HARM,
    TrapTag.PROMPT_INJECTION,
    TrapTag.IDENTITY_FRAUD,
    TrapTag.ACTION_REQUEST,
    TrapTag.PAYMENT_DISPUTE,
    TrapTag.SCORE_DISPUTE,
    TrapTag.INSUFFICIENT_INFO,
    TrapTag.OUT_OF_SCOPE,
    TrapTag.COURTESY,
    TrapTag.SECURITY_DISCLOSURE,
    TrapTag.ADMIN_ACTION,
    TrapTag.THIRD_PARTY,
    TrapTag.NORMAL_FAQ,
)


class TrapResult(BaseModel):
    """Stage 2 output: one or more taxonomy tags plus compact reasoning."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    tags: list[TrapTag] = Field(
        ...,
        min_length=1,
        description="One or more tags from the fixed 13-category trap taxonomy.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        max_length=600,
        description="Short classifier rationale grounded only in the ticket text.",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> list[Any]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("tags must be a list of trap tag strings")

        cleaned: list[str] = []
        for item in value:
            if isinstance(item, TrapTag):
                cleaned.append(item.value)
            elif isinstance(item, str):
                cleaned.append(item.strip().upper())
            else:
                raise TypeError(f"unsupported trap tag value: {item!r}")
        return cleaned

    @field_validator("tags")
    @classmethod
    def _dedupe_and_order_tags(cls, value: list[TrapTag]) -> list[TrapTag]:
        seen = set(value)
        ordered = [tag for tag in TRAP_PRIORITY if tag in seen]
        return ordered or [TrapTag.NORMAL_FAQ]

    @field_validator("reasoning")
    @classmethod
    def _compact_reasoning(cls, value: str) -> str:
        return " ".join(value.split())


class Ticket(BaseModel):
    """Normalized input ticket carried between stages 1-7."""

    model_config = ConfigDict(extra="forbid")

    issue: str = Field(default="", description="Unicode-normalized ticket body.")
    subject: str = Field(default="", description="Unicode-normalized ticket subject.")
    company: str | None = Field(default=None, description="Stated company/domain, if provided.")
    language: str = Field(default="unknown", description="Detected ISO-639 language code.")

    @property
    def text(self) -> str:
        """Combined support text for classification and retrieval."""

        return "\n".join(part for part in (self.subject, self.issue) if part).strip()


class ChunkRecord(BaseModel):
    """One line from data/processed/chunks.jsonl."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    domain: str
    url_or_file: str
    heading_path: str
    text: str


class RetrievedChunk(ChunkRecord):
    """A retrieved chunk with fused and source-specific ranking metadata."""

    rrf_score: float = Field(ge=0.0)
    normalized_score: float = Field(ge=0.0, le=1.0)
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    dense_score: float | None = None


Status = Literal["replied", "escalated"]
RequestType = Literal["product_issue", "feature_request", "bug", "invalid"]


class TriageDecision(BaseModel):
    """Final row written to output.csv."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1)
    product_area: str = Field(default="general_support")
    status: Status
    request_type: RequestType
    justification: str = Field(min_length=1)

    @field_validator("product_area", mode="before")
    @classmethod
    def _default_product_area(cls, value: object) -> str:
        compact = " ".join(str(value or "").split())
        return compact or "general_support"

    @field_validator("response", "justification")
    @classmethod
    def _compact_output_text(cls, value: str) -> str:
        compact = " ".join(value.split())
        if not compact:
            raise ValueError("output field must not be blank")
        return compact
