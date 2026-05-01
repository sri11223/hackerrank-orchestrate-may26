"""Seven-stage ticket orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import REPO_ROOT
from .generate import GroundedGenerationResult, generate_response
from .handlers import dispatch_trap_handler
from .retrieval import retrieve
from .sanitize import sanitize_ticket
from .schema import RetrievedChunk, Ticket, TriageDecision, TrapResult, TrapTag
from .traps import classify_traps
from .verify import VerificationResult, verify_response


RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_TRACES_DIR = REPO_ROOT / "traces"


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

    trap_result = classify_traps(ticket.text or ticket.issue, ticket.company or "None")
    trace["stages"]["trap_classifier"] = _trap_dump(trap_result)

    chunks = retrieve(ticket.text, domain=ticket.company, k=5)
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
        generation = generate_response(ticket, chunks)
        trace["stages"]["generation"] = generation.model_dump()

        verifier = verify_response(generation, ticket, chunks)
        trace["stages"]["verifier"] = verifier.model_dump()

        final_decision = TriageDecision(
            status="replied",
            product_area=generation.product_area,
            response=generation.response,
            request_type=generation.request_type,
            justification=(
                f"[NORMAL_FAQ] grounded generation; retrieval_score={top_score:.2f}; "
                f"verifier={'pass' if verifier.safe else 'fail'}"
            ),
        )

    final_decision = _apply_confidence_gates(
        decision=final_decision,
        top_score=top_score,
        verifier=verifier,
    )
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
        product_area="general support",
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


def _write_trace(trace: dict[str, Any], traces_dir: Path | None, ticket_id: int | str) -> None:
    if traces_dir is None:
        return
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"ticket_{ticket_id}.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
