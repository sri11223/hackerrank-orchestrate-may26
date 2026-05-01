"""Runtime configuration for the support triage agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_CHUNKS_JSONL = PROCESSED_DATA_DIR / "chunks.jsonl"
SUPPORT_TICKETS_DIR = REPO_ROOT / "support_tickets"
DEFAULT_INPUT_CSV = SUPPORT_TICKETS_DIR / "support_tickets.csv"
DEFAULT_OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"
LOG_PATH = Path.home() / "hackerrank_orchestrate" / "log.txt"
DENSE_RETRIEVAL_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60
DENSE_MAX_SEQ_LENGTH = 128


@dataclass(frozen=True)
class Settings:
    """Environment-backed settings.

    API keys are intentionally optional at import time so deterministic,
    non-LLM pipeline pieces can run without secrets.
    """

    openai_api_key: str | None
    groq_api_key: str | None
    openai_model: str
    groq_model: str
    dense_retrieval_model: str
    dense_max_seq_length: int
    request_timeout_seconds: float
    max_retries: int
    retry_base_delay_seconds: float
    retry_max_delay_seconds: float

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment and optional .env files."""

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(CODE_DIR / ".env")

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        dense_retrieval_model=os.getenv("DENSE_RETRIEVAL_MODEL", DENSE_RETRIEVAL_MODEL),
        dense_max_seq_length=_env_int("DENSE_MAX_SEQ_LENGTH", DENSE_MAX_SEQ_LENGTH),
        request_timeout_seconds=_env_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0),
        max_retries=_env_int("LLM_MAX_RETRIES", 4),
        retry_base_delay_seconds=_env_float("LLM_RETRY_BASE_DELAY_SECONDS", 0.75),
        retry_max_delay_seconds=_env_float("LLM_RETRY_MAX_DELAY_SECONDS", 12.0),
    )


settings = get_settings()
