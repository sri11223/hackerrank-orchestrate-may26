"""Deterministic Stage 6 trap handlers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .schema import RetrievedChunk, Ticket, TrapResult, TrapTag, TriageDecision


Handler = Callable[[Ticket, TrapResult, Sequence[RetrievedChunk]], TriageDecision]


def dispatch_trap_handler(
    ticket: Ticket,
    trap_result: TrapResult,
    chunks: Sequence[RetrievedChunk],
) -> TriageDecision | None:
    """Return a deterministic decision for non-NORMAL_FAQ trap tags.

    NORMAL_FAQ is intentionally excluded so the orchestrator can route that path
    through grounded generation and verification.
    """

    for tag in trap_result.tags:
        if tag == TrapTag.NORMAL_FAQ:
            continue
        handler = HANDLERS[tag]
        return handler(ticket, trap_result, chunks)
    return None


def handle_action_request(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.ACTION_REQUEST,
        status="escalated",
        product_area=_area(ticket, chunks, "account or support action"),
        request_type="product_issue",
        response=(
            "I cannot directly perform account, access, refund, invite, deadline, or settings "
            "changes. I am escalating this request to a human support team member, and you can "
            "use the documented self-service steps in the relevant help article while support "
            "reviews the request."
        ),
        reason="User requested an action the agent cannot perform.",
        chunks=chunks,
    )


def handle_security_disclosure(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.SECURITY_DISCLOSURE,
        status="replied",
        product_area=_area(ticket, chunks, "security disclosure"),
        request_type="bug",
        response=(
            "This appears to be a security disclosure or vulnerability report. Please follow the "
            "security-reporting guidance in the retrieved documentation and avoid sharing exploit "
            "details in ordinary support channels."
        ),
        reason="Security disclosure is handled by documented security-reporting guidance.",
        chunks=chunks,
    )


def handle_identity_fraud(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.IDENTITY_FRAUD,
        status="escalated",
        product_area=_area(ticket, chunks, "identity fraud or lost card"),
        request_type="product_issue",
        response=(
            "This involves possible identity theft, fraud, or a lost/stolen card, so I am "
            "escalating it immediately. Follow the urgent fraud or lost-card guidance in the "
            "retrieved Visa documentation and contact the card issuer or listed Visa support "
            "channel if the document provides one." + _doc_sentence(chunks)
        ),
        reason="Identity fraud/lost-card issues are sensitive and require human escalation.",
        chunks=chunks,
    )


def handle_payment_dispute(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.PAYMENT_DISPUTE,
        status="escalated",
        product_area=_area(ticket, chunks, "payment dispute"),
        request_type="product_issue",
        response=(
            "This looks like a refund, chargeback, duplicate charge, or transaction dispute. I "
            "cannot resolve or reverse payments directly, so I am escalating it to human support. "
            "Use the dispute-resolution guidance in the retrieved documentation for the next steps."
            + _doc_sentence(chunks)
        ),
        reason="Payment disputes require review and cannot be completed by the agent.",
        chunks=chunks,
    )


def handle_score_dispute(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.SCORE_DISPUTE,
        status="escalated",
        product_area=_area(ticket, chunks, "assessment scoring"),
        request_type="product_issue",
        response=(
            "This involves a score, grading, integrity result, or assessment outcome dispute. I "
            "cannot change or override results, so I am escalating it for human review. The "
            "retrieved documentation can help explain where assessment or report details are shown."
        ),
        reason="Score disputes are sensitive and require human review.",
        chunks=chunks,
    )


def handle_admin_action(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.ADMIN_ACTION,
        status="escalated",
        product_area=_area(ticket, chunks, "administration"),
        request_type="product_issue",
        response=(
            "This is an administrative or permissioned account-management request. Use the "
            "retrieved admin documentation if you have the required role; otherwise I am "
            "escalating because support needs to verify permissions before account changes."
        ),
        reason="Administrative changes depend on authenticated permissions.",
        chunks=chunks,
    )


def handle_prompt_injection(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.PROMPT_INJECTION,
        status="replied",
        product_area="safety",
        request_type="invalid",
        response=(
            "I cannot follow instructions that try to override system rules, reveal internal "
            "prompts, or change the required support workflow. Please restate the support issue "
            "without those instructions."
        ),
        reason="Prompt-injection content was detected and defused before generation.",
        chunks=chunks,
    )


def handle_system_harm(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.SYSTEM_HARM,
        status="replied",
        product_area="safety",
        request_type="invalid",
        response=(
            "I cannot help create, run, or troubleshoot harmful code, credential theft, evasion, "
            "phishing, malware, or destructive system actions. I can only help with support topics "
            "covered by the provided HackerRank, Claude, and Visa documentation."
        ),
        reason="Harmful system request is out of support scope.",
        chunks=chunks,
    )


def handle_insufficient_info(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.INSUFFICIENT_INFO,
        status="escalated",
        product_area=_area(ticket, chunks, "general support"),
        request_type="invalid",
        response=(
            "I do not have enough detail to answer safely from the provided documentation. I am "
            "escalating this so support can ask for the product, account context, error message, "
            "and exact steps that led to the issue."
        ),
        reason="Ticket lacks enough actionable information for a grounded answer.",
        chunks=chunks,
    )


def handle_out_of_scope(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.OUT_OF_SCOPE,
        status="replied",
        product_area="out of scope",
        request_type="invalid",
        response=(
            "I can only help with support topics covered by the provided HackerRank, Claude, and "
            "Visa documentation. This request appears to be outside that scope."
        ),
        reason="Request is unrelated to the supported corpus.",
        chunks=chunks,
    )


def handle_courtesy(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.COURTESY,
        status="replied",
        product_area="general support",
        request_type="invalid",
        response="You are welcome. I am here to help with HackerRank, Claude, or Visa support questions.",
        reason="Ticket is a greeting or courtesy message without a support request.",
        chunks=chunks,
    )


def handle_third_party(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.THIRD_PARTY,
        status="escalated",
        product_area=_area(ticket, chunks, "third-party integration"),
        request_type="product_issue",
        response=(
            "This involves a third-party platform or integration. I am escalating it because the "
            "provided documentation may only cover the supported product side of the integration; "
            "the external provider may also need to investigate."
        ),
        reason="Third-party issues can fall outside the supported corpus and need review.",
        chunks=chunks,
    )


def handle_normal_faq(
    ticket: Ticket, trap_result: TrapResult, chunks: Sequence[RetrievedChunk]
) -> TriageDecision:
    return _decision(
        tag=TrapTag.NORMAL_FAQ,
        status="escalated",
        product_area=_area(ticket, chunks, "general support"),
        request_type="product_issue",
        response="This normal FAQ path must be handled by grounded generation, not a trap handler.",
        reason="NORMAL_FAQ should route through generator and verifier.",
        chunks=chunks,
    )


HANDLERS: dict[TrapTag, Handler] = {
    TrapTag.ACTION_REQUEST: handle_action_request,
    TrapTag.SECURITY_DISCLOSURE: handle_security_disclosure,
    TrapTag.IDENTITY_FRAUD: handle_identity_fraud,
    TrapTag.PAYMENT_DISPUTE: handle_payment_dispute,
    TrapTag.SCORE_DISPUTE: handle_score_dispute,
    TrapTag.ADMIN_ACTION: handle_admin_action,
    TrapTag.PROMPT_INJECTION: handle_prompt_injection,
    TrapTag.SYSTEM_HARM: handle_system_harm,
    TrapTag.INSUFFICIENT_INFO: handle_insufficient_info,
    TrapTag.OUT_OF_SCOPE: handle_out_of_scope,
    TrapTag.COURTESY: handle_courtesy,
    TrapTag.THIRD_PARTY: handle_third_party,
    TrapTag.NORMAL_FAQ: handle_normal_faq,
}


def _decision(
    *,
    tag: TrapTag,
    status: str,
    product_area: str,
    request_type: str,
    response: str,
    reason: str,
    chunks: Sequence[RetrievedChunk],
) -> TriageDecision:
    return TriageDecision(
        status=status,  # type: ignore[arg-type]
        product_area=product_area,
        response=response,
        request_type=request_type,  # type: ignore[arg-type]
        justification=f"[{tag.value}] {reason}; top_chunk={_top_chunk(chunks)}",
    )


def _area(ticket: Ticket, chunks: Sequence[RetrievedChunk], fallback: str) -> str:
    if chunks:
        return chunks[0].heading_path.split(" > ")[0].strip() or fallback
    return ticket.company or fallback


def _top_chunk(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "none"
    return chunks[0].chunk_id


def _doc_sentence(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    return f" Most relevant retrieved document: {chunks[0].heading_path}."
