# Orchestrate Support Triage Agent

This is the complete working source package for the HackerRank Orchestrate
support triage agent. The agent processes support tickets for HackerRank,
Claude, and Visa, then writes the required `output.csv` fields:

```text
issue, subject, company, response, product_area, status, request_type, justification
```

The design is safety-first: answer only when the local corpus supports the
response, otherwise escalate.

## What To Open First

- `triage/` contains the production agent implementation.
- `support_tickets/output.csv` is the final 29-ticket production result.
- `traces/production/` contains one JSON decision trace per production ticket.
- `traces/crucible/` contains adversarial safety traces.
- `docs/README.md` is the judge proof dossier.
- `docs/GRAND_MASTER_LOG.md` is the architecture story with diagrams.
- `scorecard.py` recomputes the integrity scorecard.

## Install

From the repository root:

```bash
pip install -e .
```

On Windows, the one-command installer also adds the Python user Scripts folder
to User `PATH`:

```powershell
irm https://raw.githubusercontent.com/sri11223/hackerrank-orchestrate-may26/main/install.ps1 | iex
```

If console scripts are unavailable, the no-PATH launcher works:

```bash
python -m triage --help
```

## Environment

Set API keys in the environment or in a local `.env` file. Never commit `.env`.

```bash
OPENAI_API_KEY=...
GROQ_API_KEY=...
```

The installed CLI loads `.env` from the current working directory, the checkout
root when detected, or `TRIAGE_ENV_FILE`.

## Run

Production batch:

```bash
triage run --input support_tickets/support_tickets.csv --out support_tickets/output.csv
```

No-PATH equivalent:

```bash
python -m triage run --input support_tickets/support_tickets.csv --out support_tickets/output.csv
```

Interactive terminal demo:

```bash
triage interactive
```

Explain one decision:

```bash
triage explain 1 --traces traces/production
```

Validate the required submission CSV:

```bash
python ../eval/validate_submission.py --output support_tickets/output.csv
```

Recompute the integrity scorecard:

```bash
python scorecard.py
```

## Approach Overview

The agent is a deterministic seven-stage decision system:

1. **Sanitize and Privacy Shield**: normalize Unicode, strip hostile control
   characters, detect language, and redact emails, credit cards, SSNs, and
   phone numbers before logging or LLM calls.
2. **Trap Taxonomy**: classify every ticket into fixed safety categories such
   as `PROMPT_INJECTION`, `SYSTEM_HARM`, `IDENTITY_FRAUD`, `ACTION_REQUEST`,
   and `NORMAL_FAQ`.
3. **Domain Routing**: resolve the support domain from the ticket company and
   content.
4. **Hybrid Retrieval**: retrieve from the provided corpus only, using BM25,
   BGE dense embeddings, reciprocal-rank fusion, and cross-encoder reranking.
5. **Grounded Generation**: generate strict Pydantic JSON using only retrieved
   chunks, with citations and an exact source quote.
6. **Normalization**: snap product areas and request types to canonical labels.
7. **Adversarial Verification**: audit the draft against the original issue and
   chunks, self-heal once, then hard-escalate if unsafe or unsupported.

## Hallucination Prevention

The generator never receives free authority. It is boxed in by:

- Local-only retrieval from the provided corpus.
- XML isolation of user text as untrusted data.
- Strict Pydantic output schemas.
- Citation validation against retrieved chunk ids.
- Verbatim `exact_quote` enforcement.
- A separate verifier checking prompt-injection obedience, unsupported claims,
  and unauthorized action commitments.
- Final confidence gates for retrieval score, verifier safety, and non-answer
  phrases like "I cannot answer" or "not mentioned in the documents".

If the system cannot prove an answer, it escalates.

## Final Package Proof

The final scorecard reports:

- `29` production rows.
- `29` production JSON traces.
- `10` crucible JSON traces.
- `0` prompt-injection joke failures.
- `100%` source receipts on replied production tickets.
- Package under `50 MB`.
- No `.env`, `.pyc`, or `__pycache__` files in the zip.

For the full engineering narrative and diagrams, open:

```text
docs/GRAND_MASTER_LOG.md
```
