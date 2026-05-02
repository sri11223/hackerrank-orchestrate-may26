"""Runtime configuration for the support triage agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_RESOURCE_DIR = PACKAGE_DIR / "resources"
PACKAGE_DATA_DIR = PACKAGE_RESOURCE_DIR
PACKAGE_SUPPORT_TICKETS_DIR = PACKAGE_RESOURCE_DIR / "support_tickets"


def _find_project_root() -> Path:
    """Prefer a real checkout when present, otherwise fall back to install root."""

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    package_checkout_root = PACKAGE_DIR.parents[1]
    candidates.append(package_checkout_root)

    for candidate in candidates:
        if (candidate / "AGENTS.md").exists() and (candidate / "code" / "triage").exists():
            return candidate
    return package_checkout_root


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return Path(raw).expanduser().resolve()


def _first_existing(paths: list[Path], fallback: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return fallback


def _first_valid_data_dir(paths: list[Path], fallback: Path) -> Path:
    for path in paths:
        if (path / "processed" / "chunks.jsonl").exists() or any(
            (path / domain).exists() for domain in ("hackerrank", "claude", "visa")
        ):
            return path
    return fallback


def _first_valid_support_dir(paths: list[Path], fallback: Path) -> Path:
    for path in paths:
        if (path / "support_tickets.csv").exists():
            return path
    return fallback


REPO_ROOT = _find_project_root()
CODE_DIR = REPO_ROOT / "code" if (REPO_ROOT / "code").exists() else PACKAGE_DIR.parent

_repo_data_dir = REPO_ROOT / "data"
DATA_DIR = _env_path("TRIAGE_DATA_DIR") or _first_valid_data_dir(
    [
        _repo_data_dir,
        Path.cwd() / "data",
        PACKAGE_DATA_DIR,
    ],
    PACKAGE_DATA_DIR,
)
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_CHUNKS_JSONL = _env_path("TRIAGE_CHUNKS_JSONL") or _first_existing(
    [
        PROCESSED_DATA_DIR / "chunks.jsonl",
        PACKAGE_DATA_DIR / "processed" / "chunks.jsonl",
    ],
    PROCESSED_DATA_DIR / "chunks.jsonl",
)

_repo_support_tickets_dir = REPO_ROOT / "support_tickets"
SUPPORT_TICKETS_DIR = _env_path("TRIAGE_SUPPORT_TICKETS_DIR") or _first_valid_support_dir(
    [
        _repo_support_tickets_dir,
        Path.cwd() / "support_tickets",
        PACKAGE_SUPPORT_TICKETS_DIR,
    ],
    Path.cwd() / "support_tickets",
)
DEFAULT_INPUT_CSV = _env_path("TRIAGE_INPUT_CSV") or _first_existing(
    [
        SUPPORT_TICKETS_DIR / "support_tickets.csv",
        PACKAGE_SUPPORT_TICKETS_DIR / "support_tickets.csv",
    ],
    SUPPORT_TICKETS_DIR / "support_tickets.csv",
)
DEFAULT_OUTPUT_CSV = _env_path("TRIAGE_OUTPUT_CSV") or (
    SUPPORT_TICKETS_DIR / "output.csv"
    if SUPPORT_TICKETS_DIR != PACKAGE_SUPPORT_TICKETS_DIR
    else Path.cwd() / "support_tickets" / "output.csv"
)
DEFAULT_TRACES_DIR = _env_path("TRIAGE_TRACES_DIR") or (
    REPO_ROOT / "traces" if (REPO_ROOT / "AGENTS.md").exists() else Path.cwd() / "traces"
)
LOG_PATH = Path.home() / "hackerrank_orchestrate" / "log.txt"
DENSE_RETRIEVAL_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
    cross_encoder_enabled: bool
    cross_encoder_model: str
    cross_encoder_top_n: int
    llm_cache_enabled: bool
    llm_cache_dir: Path
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment and optional .env files."""

    env_file = _env_path("TRIAGE_ENV_FILE")
    if env_file is not None:
        load_dotenv(env_file)
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(CODE_DIR / ".env")

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        dense_retrieval_model=os.getenv("DENSE_RETRIEVAL_MODEL", DENSE_RETRIEVAL_MODEL),
        dense_max_seq_length=_env_int("DENSE_MAX_SEQ_LENGTH", DENSE_MAX_SEQ_LENGTH),
        cross_encoder_enabled=_env_bool("CROSS_ENCODER_ENABLED", True),
        cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", CROSS_ENCODER_MODEL),
        cross_encoder_top_n=_env_int("CROSS_ENCODER_TOP_N", 20),
        llm_cache_enabled=_env_bool("LLM_CACHE_ENABLED", True),
        llm_cache_dir=Path(os.getenv("LLM_CACHE_DIR", str(PROCESSED_DATA_DIR / "llm_cache"))),
        request_timeout_seconds=_env_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0),
        max_retries=_env_int("LLM_MAX_RETRIES", 4),
        retry_base_delay_seconds=_env_float("LLM_RETRY_BASE_DELAY_SECONDS", 0.75),
        retry_max_delay_seconds=_env_float("LLM_RETRY_MAX_DELAY_SECONDS", 12.0),
    )


settings = get_settings()
