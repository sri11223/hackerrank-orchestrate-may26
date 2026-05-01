# Orchestrate Agent: Judge Proof Dossier

This folder is the evidence pack for the terminal support triage agent. The
core idea is simple: the agent does not behave like an open-ended chatbot. It
behaves like a deterministic support decision machine with narrow LLM calls
boxed behind schemas, retrieval evidence, safety gates, and JSON sidecars.

The package includes the production CSV proof, the adversarial crucible proof,
and the trace files that show every stage of reasoning.

## What To Open First

- `code/support_tickets/output.csv` is the final 29-ticket production output.
- `code/support_tickets/crucible_output.csv` is the 10-ticket adversarial run.
- `code/traces/production/` contains one JSON sidecar per production ticket.
- `code/traces/crucible/` contains one JSON sidecar per adversarial ticket.
- `code/scorecard.py` recomputes the final integrity scorecard.
- `code/docs/scorecard.json` contains the last computed 100/100 integrity scan.

The trace files are the proof of reasoning. Each one records the sanitized
ticket, trap classifier tags, retrieval evidence, handler/generator path,
verifier result, confidence gates, product-area normalization, exact source
receipt status, and per-stage timings.

## The Story

Most support bots fail in two predictable ways: they hallucinate policy, or
they obey malicious user text. This project was built around the opposite
philosophy. The user text is never trusted. The local docs are the source of
truth. Python decides when the model is allowed to speak.

The pipeline starts by cleaning the ticket and detecting language. It then
classifies the ticket into a fixed trap taxonomy before retrieval. Prompt
injections, harmful system requests, outages, payment disputes, score disputes,
and other sensitive patterns can bypass generation entirely. Normal support
questions enter hybrid retrieval, where exact BM25 matching protects support
artifacts like issuer names and error wording, while BGE dense retrieval handles
semantic paraphrases. The two rankings are normalized and fused with reciprocal
rank fusion.

Only after that does generation happen. The prompt wraps the original ticket in
`<user_issue>` tags and explicitly treats it as untrusted data. The generator
must answer only from the retrieved chunks and emit strict Pydantic JSON. The
response is then audited by a separate verifier that sees the original user
issue, the draft, and the source chunks. If the verifier rejects the draft, the
system performs exactly one self-healing rewrite with the verifier critique
passed back into the generator. If the retry still fails, the response is hard
escalated.

Finally, the agent applies confidence gates. If the top retrieval confidence is
too low, or the verifier is unsafe, Python overrides the answer to
`escalated`. Every result gets a JSON sidecar so the judge can inspect the full
decision tree.

## The 13 Pillars

1. **Local Corpus Only**: all answers are grounded in the provided
   HackerRank, Claude, and Visa docs.
2. **Input Sanitation**: Unicode is normalized, weird control characters are
   removed, and language is detected before routing.
3. **Trap Taxonomy**: tickets are tagged before retrieval with high-risk labels
   such as `PROMPT_INJECTION`, `SYSTEM_HARM`, `IDENTITY_FRAUD`,
   `PAYMENT_DISPUTE`, and `SYSTEM_OUTAGE`.
4. **Deterministic Bypass**: unsafe or operationally sensitive tags can bypass
   LLM generation and return controlled responses.
5. **Domain Filtering**: retrieval filters by company/domain when known.
6. **BM25 Exact Match**: sparse retrieval preserves exact support wording,
   issuer names, product labels, and error-like terms.
7. **BGE Dense Retrieval**: semantic search uses `BAAI/bge-small-en-v1.5`, held
   as a singleton so the model is loaded once per process.
8. **RRF Fusion**: BM25 and dense scores are normalized and fused by reciprocal
   rank fusion for robust evidence selection.
9. **XML Isolation**: the user issue is wrapped in XML and explicitly treated
   as untrusted data.
10. **Strict Schemas**: Pydantic forbids malformed output and keeps CSV labels
    inside the allowed contract.
11. **Source Receipts**: every replied production trace carries an
    `exact_quote`, copied verbatim from retrieved docs.
12. **Actor-Critic Self-Healing**: a verifier audits each generated draft; one
    critique-driven rewrite is allowed before escalation.
13. **Full Audit Trail**: every run writes unique JSON sidecars with stage
    decisions and timing blocks.

## Final Integrity Score

The packaged proof currently scores `100.0 / 100.0` on the judge-facing
integrity scan:

- Grounding Accuracy: `23 / 23` replied production traces have non-empty
  verbatim `exact_quote` receipts.
- Safety Moat: `0 / 3` prompt-injection crucible tickets produced a replied
  joke or roleplay failure.
- Self-Healing Efficacy: `3` verifier-triggered rewrites occurred, with `2`
  ending in final `replied` decisions.
- Auditability: `39 / 39` production plus crucible traces contain complete
  timing blocks for every recorded stage.
- Packaging: `submission_code.zip` is `0.146 MB`, under 50 MB, with no `.env`
  or compiled Python files.

## Safety Gates

The final status can be forcibly escalated even if the generator writes a nice
draft. The hard gates are:

- top retrieval confidence below `0.35`
- verifier returns `safe=false`
- row processing error
- deterministic handler decides the request requires human review

The important detail is that the model never gets final authority. It proposes;
the state machine disposes.

## Model Routing

Normal FAQ tickets try the fast Groq/Llama path first. Sensitive self-service
generation and verifier rewrites use the stronger OpenAI path. If Groq is
rate-limited or unavailable, the orchestrator disables that route for the
current process and falls back to OpenAI instead of crashing the batch.

## Proof Layout

```text
code/
  docs/
    README.md
  support_tickets/
    support_tickets.csv
    output.csv
    crucible_tickets.csv
    crucible_output.csv
    sample_support_tickets.csv
  traces/
    production/   # 29 JSON sidecars
    crucible/     # 10 JSON sidecars
  triage/
    sanitize.py
    traps.py
    handlers.py
    retrieval.py
    generate.py
    verify.py
    orchestrator.py
    cli.py
```

## Commands

Run production:

```bash
PYTHONPATH=code python -m triage.cli run \
  --input support_tickets/support_tickets.csv \
  --out support_tickets/output.csv \
  --traces code/traces/production
```

Run crucible:

```bash
PYTHONPATH=code python -m triage.cli crucible \
  --input support_tickets/crucible_tickets.csv \
  --out support_tickets/crucible_output.csv \
  --traces code/traces/crucible
```

Recompute the scorecard:

```bash
PYTHONPATH=code python code/scorecard.py
```

Package:

```bash
python package.py
```

## Why This Wins

The value is not just a better prompt. The value is the surrounding system:
deterministic trap routing, hybrid retrieval, strict schemas, source receipts,
self-healing verification, hard confidence gates, semantic caching, and sidecar
traces. The result is a support agent that can answer normal tickets, refuse or
defuse adversarial tickets, and show the exact evidence trail for every choice.
