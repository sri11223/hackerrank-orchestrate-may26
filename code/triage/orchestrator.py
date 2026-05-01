"""Seven-stage ticket orchestration."""

from __future__ import annotations

import json
import re
from csv import DictReader
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .cache import SemanticCache
from .config import REPO_ROOT
from .generate import GroundedGenerationResult, generate_response
from .handlers import dispatch_trap_handler
from .retrieval import embed_query, load_chunks, retrieve
from .sanitize import sanitize_ticket
from .schema import RetrievedChunk, Ticket, TriageDecision, TrapResult, TrapTag
from .traps import classify_traps
from .verify import VerificationResult, verify_response


RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_TRACES_DIR = REPO_ROOT / "traces"
SAMPLE_LABELS_CSV = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
FALLBACK_PRODUCT_AREAS = (
    "screen",
    "community",
    "general_support",
    "privacy",
    "conversation_management",
    "travel_support",
)
SEMANTIC_CACHE = SemanticCache()


def process_ticket(
    row_dict: Mapping[str, Any],
    *,
    ticket_id: int | str = 0,
    traces_dir: Path | None = DEFAULT_TRACES_DIR,
) -> TriageDecision:
    """Run the full deterministic/LLM triage pipeline for one CSV row."""

    trace: dict[str, Any] = {
        "ticket_id": ticket_id,
        "input": dict(row_dict),
        "stages": {},
    }

    ticket = sanitize_ticket(
        issue=_get(row_dict, "issue"),
        subject=_get(row_dict, "subject"),
        company=_get(row_dict, "company"),
    )
    trace["stages"]["sanitize"] = ticket.model_dump()
    query_embedding = embed_query(ticket.issue or ticket.text)
    cached_decision = SEMANTIC_CACHE.check_cache(query_embedding)
    if cached_decision is not None:
        print("CACHE HIT: Bypassing LLM")
        trace["stages"]["semantic_cache"] = {
            "hit": True,
            "threshold": 0.95,
        }
        trace["final_decision"] = cached_decision.model_dump()
        _write_trace(trace, traces_dir, ticket_id)
        return cached_decision
    trace["stages"]["semantic_cache"] = {
        "hit": False,
        "threshold": 0.95,
    }

    trap_result = classify_traps(ticket.text or ticket.issue, ticket.company or "None")
    trace["stages"]["trap_classifier"] = _trap_dump(trap_result)

    chunks = retrieve(ticket.text, domain=ticket.company, k=5)
    chunks = _augment_self_service_chunks(ticket, trap_result, chunks)
    top_score = _top_retrieval_score(chunks)
    trace["stages"]["retrieval"] = {
        "top_score": top_score,
        "threshold": RETRIEVAL_CONFIDENCE_THRESHOLD,
        "chunks": [_chunk_trace(chunk) for chunk in chunks],
    }

    generation: GroundedGenerationResult | None = None
    verifier: VerificationResult | None = None
    handler_decision = dispatch_trap_handler(ticket, trap_result, chunks)

    if handler_decision is not None:
        final_decision = handler_decision
        trace["stages"]["handler"] = {
            "mode": "deterministic",
            "decision": final_decision.model_dump(),
        }
    else:
        generation, verifier, generation_trace = _generate_and_verify_with_self_healing(
            ticket=ticket,
            chunks=chunks,
            trap_result=trap_result,
        )
        trace["stages"]["generation"] = generation_trace
        generation_tag = _generation_tag(trap_result)

        final_decision = TriageDecision(
            status="replied",
            product_area=generation.product_area,
            response=generation.response,
            request_type=_generation_request_type(generation, trap_result),
            justification=(
                f"[{generation_tag.value}] grounded generation; retrieval_score={top_score:.2f}; "
                f"verifier={'pass' if verifier.safe else 'fail'}"
            ),
        )

    final_decision = _apply_confidence_gates(
        decision=final_decision,
        top_score=top_score,
        verifier=verifier,
    )
    final_decision = _snap_decision_product_area(final_decision, ticket, chunks)
    SEMANTIC_CACHE.add_to_cache(query_embedding, final_decision)
    trace["final_decision"] = final_decision.model_dump()
    _write_trace(trace, traces_dir, ticket_id)
    return final_decision


def error_decision(
    row_dict: Mapping[str, Any],
    exc: Exception,
    *,
    ticket_id: int | str,
    traces_dir: Path | None = DEFAULT_TRACES_DIR,
) -> TriageDecision:
    """Create a valid escalated row when a single ticket crashes."""

    decision = TriageDecision(
        status="escalated",
        product_area="general_support",
        response=(
            "I could not safely process this ticket automatically, so I am escalating it for "
            "human review."
        ),
        request_type="invalid",
        justification=f"[ERROR] row processing failed safely; error_type={type(exc).__name__}",
    )
    trace = {
        "ticket_id": ticket_id,
        "input": dict(row_dict),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "final_decision": decision.model_dump(),
    }
    _write_trace(trace, traces_dir, ticket_id)
    return decision


def _apply_confidence_gates(
    *,
    decision: TriageDecision,
    top_score: float,
    verifier: VerificationResult | None,
) -> TriageDecision:
    reasons: list[str] = []
    if top_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        reasons.append(f"retrieval_score={top_score:.2f}<0.35")
    if verifier is not None and not verifier.safe:
        reasons.append("verifier=fail")

    if not reasons:
        return decision

    return TriageDecision(
        status="escalated",
        product_area=decision.product_area,
        response=(
            "I am escalating this ticket because I could not verify a sufficiently grounded, "
            "safe automated response from the provided documentation."
        ),
        request_type=decision.request_type,
        justification=f"{decision.justification}; hard_override={'|'.join(reasons)}",
    )


def _snap_decision_product_area(
    decision: TriageDecision,
    ticket: Ticket,
    chunks: list[RetrievedChunk],
) -> TriageDecision:
    snapped = snap_product_area(decision.product_area, ticket, chunks)
    if snapped == decision.product_area:
        return decision
    return TriageDecision(
        status=decision.status,
        product_area=snapped,
        response=decision.response,
        request_type=decision.request_type,
        justification=f"{decision.justification}; product_area_snapped={decision.product_area}->{snapped}",
    )


def snap_product_area(
    product_area: str | None,
    ticket: Ticket,
    chunks: list[RetrievedChunk],
) -> str:
    """Force arbitrary model labels into the sample-observed product areas."""

    valid = known_product_areas()
    raw = normalize_product_area(product_area)
    if raw in valid:
        return raw

    heading_text = " ".join(chunk.heading_path for chunk in chunks[:3]).casefold()
    if "community" in heading_text:
        return "community"
    if "privacy" in heading_text or "sensitive data" in heading_text:
        return "privacy"
    if "conversation" in heading_text or "chat" in heading_text:
        return "conversation_management"
    ticket_text = ticket.text.casefold()
    company = (ticket.company or "").strip().casefold()
    if (
        company == "visa"
        and "card" in ticket_text
        and ("lost" in ticket_text or "stolen" in ticket_text)
        and "traveller" not in ticket_text
        and "traveler" not in ticket_text
        and "cheque" not in ticket_text
        and "check" not in ticket_text
    ):
        return "general_support"
    if "travel" in heading_text or "lost or stolen" in heading_text or "visa card" in heading_text:
        return "travel_support"
    if "screen" in heading_text or "test" in heading_text or "candidate" in heading_text:
        return "screen"

    close = get_close_matches(raw, valid, n=1, cutoff=0.55)
    if close:
        return close[0]

    if company == "hackerrank":
        return "screen"
    if company == "visa":
        return "travel_support"
    return "general_support"


@lru_cache(maxsize=1)
def known_product_areas() -> list[str]:
    if not SAMPLE_LABELS_CSV.exists():
        return list(FALLBACK_PRODUCT_AREAS)

    values: set[str] = set()
    with SAMPLE_LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = DictReader(handle)
        product_area_column = None
        for column in reader.fieldnames or []:
            if _normalize_column(column) == "product_area":
                product_area_column = column
                break
        if product_area_column is None:
            return list(FALLBACK_PRODUCT_AREAS)
        for row in reader:
            values.add(normalize_product_area(row.get(product_area_column)))
    return sorted(values or set(FALLBACK_PRODUCT_AREAS))


def normalize_product_area(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")
    return normalized or "general_support"


def _augment_self_service_chunks(
    ticket: Ticket,
    trap_result: TrapResult,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Append exact self-service chunks when RRF lands on an article index.

    Some help centers split article indexes and action steps into separate small
    chunks. For action requests, a high-ranked index title is useful, but the
    generator needs the neighboring step chunk to answer without hand-waving.
    """

    if TrapTag.ACTION_REQUEST not in trap_result.tags:
        return chunks

    text = ticket.text.casefold()
    wanted_heading: str | None = None
    if (
        (ticket.company or "").strip().casefold() == "claude"
        and "delete" in text
        and ("conversation" in text or "chat" in text)
    ):
        wanted_heading = "how can i delete or rename a conversation"

    if wanted_heading is None:
        return chunks

    existing = {chunk.chunk_id for chunk in chunks}
    augmented = list(chunks)
    for chunk in _augmentation_chunks():
        heading = chunk.heading_path.casefold()
        if chunk.domain != (ticket.company or "").strip().casefold():
            continue
        if wanted_heading not in heading:
            continue
        if chunk.chunk_id in existing:
            continue
        augmented.append(
            RetrievedChunk(
                **chunk.model_dump(),
                rrf_score=0.0,
                normalized_score=0.5,
                bm25_rank=None,
                dense_rank=None,
            )
        )
        existing.add(chunk.chunk_id)
        if len(augmented) >= 8:
            break
    return augmented


@lru_cache(maxsize=1)
def _augmentation_chunks() -> tuple[Any, ...]:
    return tuple(load_chunks())


def _normalize_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _get(row: Mapping[str, Any], key: str) -> str:
    for row_key, value in row.items():
        if str(row_key).strip().casefold() == key:
            if value is None:
                return ""
            text = str(value)
            return "" if text.casefold() == "nan" else text
    return ""


def _top_retrieval_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    return float(chunks[0].normalized_score)


def _chunk_trace(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "domain": chunk.domain,
        "heading_path": chunk.heading_path,
        "url_or_file": chunk.url_or_file,
        "rrf_score": chunk.rrf_score,
        "normalized_score": chunk.normalized_score,
        "bm25_rank": chunk.bm25_rank,
        "dense_rank": chunk.dense_rank,
    }


def _trap_dump(trap_result: TrapResult) -> dict[str, Any]:
    return {
        "tags": [tag.value if isinstance(tag, TrapTag) else str(tag) for tag in trap_result.tags],
        "reasoning": trap_result.reasoning,
    }


def _generate_and_verify_with_self_healing(
    *,
    ticket: Ticket,
    chunks: list[RetrievedChunk],
    trap_result: TrapResult,
) -> tuple[GroundedGenerationResult, VerificationResult, dict[str, Any]]:
    """Run actor-critic generation with one verifier-driven rewrite attempt."""

    drafts: list[dict[str, Any]] = []

    draft = generate_response(ticket, chunks, trap_tags=trap_result.tags)
    verifier = verify_response(draft, ticket, chunks)
    drafts.append(
        {
            "attempt": 1,
            "critique": [],
            "generation": draft.model_dump(),
            "verifier": verifier.model_dump(),
        }
    )

    if verifier.safe:
        return draft, verifier, {"mode": "actor_critic", "drafts": drafts}

    print(
        "[SELF-HEALING] Verifier flagged draft. "
        f"Issues: {verifier.issues}. Triggering rewrite..."
    )
    rewrite = generate_response(
        ticket,
        chunks,
        trap_tags=trap_result.tags,
        critique=verifier.issues,
    )
    rewrite_verifier = verify_response(rewrite, ticket, chunks)
    drafts.append(
        {
            "attempt": 2,
            "critique": verifier.issues,
            "generation": rewrite.model_dump(),
            "verifier": rewrite_verifier.model_dump(),
        }
    )
    return rewrite, rewrite_verifier, {"mode": "actor_critic", "drafts": drafts}


def _generation_tag(trap_result: TrapResult) -> TrapTag:
    for tag in trap_result.tags:
        if tag in {TrapTag.IDENTITY_FRAUD, TrapTag.ACTION_REQUEST, TrapTag.NORMAL_FAQ}:
            return tag
    return trap_result.tags[0] if trap_result.tags else TrapTag.NORMAL_FAQ


def _generation_request_type(
    generation: GroundedGenerationResult,
    trap_result: TrapResult,
) -> str:
    if generation.request_type == "invalid" and TrapTag.NORMAL_FAQ in trap_result.tags:
        return "product_issue"
    return generation.request_type


def _write_trace(trace: dict[str, Any], traces_dir: Path | None, ticket_id: int | str) -> None:
    if traces_dir is None:
        return
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"ticket_{ticket_id}.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
