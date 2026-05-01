"""Package CLI for the support triage agent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import DATA_DIR, DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_CSV, LOG_PATH, get_settings
from .ingest import DEFAULT_OUTPUT_PATH, ingest_corpus
from .llm import ChatMessage, LLMClient, LLMConfigurationError, LLMResponseError, Provider
from .retrieval import RetrievalDependencyError, retrieve


app = typer.Typer(
    add_completion=False,
    help="Support triage agent for HackerRank, Claude, and Visa tickets.",
)


def _provider(value: str) -> Provider:
    normalized = value.strip().lower()
    if normalized not in {"openai", "groq"}:
        raise typer.BadParameter("provider must be either 'openai' or 'groq'")
    return normalized  # type: ignore[return-value]


@app.command()
def hello(
    message: str = typer.Argument("ping", help="Message to send to the skeleton."),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Optional provider to smoke-test with strict JSON output.",
    ),
) -> None:
    """Print environment status, optionally smoke-test an LLM provider."""

    settings = get_settings()
    typer.echo("Support triage CLI is ready.")
    typer.echo(f"OpenAI key loaded: {settings.has_openai}")
    typer.echo(f"Groq key loaded: {settings.has_groq}")
    typer.echo(f"Transcript log path: {LOG_PATH}")

    if provider is None:
        typer.echo(f"Echo: {message}")
        return

    selected = _provider(provider)
    client = LLMClient(settings=settings)
    messages = [
        ChatMessage(
            role="system",
            content='Return JSON only in the shape {"ok": true, "echo": "..."}',
        ),
        ChatMessage(role="user", content=message),
    ]
    try:
        result = client.chat_json(selected, messages, max_tokens=120)
    except (LLMConfigurationError, LLMResponseError) as exc:
        typer.echo(f"LLM smoke test failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(result.content)
    typer.echo(
        f"provider={result.provider} model={result.model} "
        f"input_tokens={result.input_tokens} output_tokens={result.output_tokens} "
        f"attempts={result.attempts}"
    )


@app.command()
def ingest(
    corpus_dir: Path = typer.Option(DATA_DIR, "--corpus-dir", help="Local support corpus root."),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--out",
        "-o",
        help="Chunk JSONL output path.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Accepted for future index rebuilds; chunks are always regenerated.",
    ),
) -> None:
    """Parse the raw support corpus into heading-aware JSONL chunks."""

    if rebuild:
        typer.echo("Rebuilding chunks; retrieval indexes are not implemented in this stage.")

    try:
        stats = ingest_corpus(corpus_dir=corpus_dir, output_path=output_path)
    except Exception as exc:
        typer.echo(f"ingest failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"corpus_root={stats.corpus_root}")
    typer.echo(f"files_read={stats.files_read}")
    typer.echo(f"chunks_written={stats.chunks_written}")
    typer.echo(f"output_path={stats.output_path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Support query to retrieve chunks for."),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Optional corpus domain/company: hackerrank, claude, or visa.",
    ),
    k: int = typer.Option(5, "--k", help="Number of fused results to print."),
) -> None:
    """Smoke-test hybrid retrieval from the terminal."""

    try:
        results = retrieve(query=query, domain=domain, k=k)
    except (FileNotFoundError, RetrievalDependencyError, RuntimeError, ValueError) as exc:
        typer.echo(f"search failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not results:
        typer.echo("No results.")
        return

    for index, result in enumerate(results, start=1):
        typer.echo(
            f"{index}. {result.chunk_id} "
            f"domain={result.domain} score={result.normalized_score:.3f} "
            f"bm25_rank={result.bm25_rank} dense_rank={result.dense_rank}"
        )
        typer.echo(f"   {result.heading_path}")
        typer.echo(f"   {result.text[:220].replace(chr(10), ' ')}")


@app.command()
def run(
    input_csv: Path = typer.Option(DEFAULT_INPUT_CSV, "--input", "-i", help="Input ticket CSV."),
    output_csv: Path = typer.Option(DEFAULT_OUTPUT_CSV, "--out", "-o", help="Output prediction CSV."),
    traces_dir: Optional[Path] = typer.Option(None, "--traces", help="Optional trace output dir."),
) -> None:
    """Placeholder for end-to-end triage run."""

    typer.echo(
        "run placeholder: "
        f"input_csv={input_csv} output_csv={output_csv} traces_dir={traces_dir}"
    )


if __name__ == "__main__":
    app()
