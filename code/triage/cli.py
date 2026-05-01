"""Package CLI for the support triage agent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .config import DATA_DIR, DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_CSV, LOG_PATH, get_settings
from .ingest import DEFAULT_OUTPUT_PATH, ingest_corpus
from .llm import ChatMessage, LLMClient, LLMConfigurationError, LLMResponseError, Provider
from .orchestrator import DEFAULT_TRACES_DIR, error_decision, process_ticket
from .retrieval import RetrievalDependencyError, retrieve


app = typer.Typer(
    add_completion=False,
    help="Support triage agent for HackerRank, Claude, and Visa tickets.",
)
console = Console()


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


def draw_banner() -> None:
    """Render the interactive terminal banner."""

    logo = Text()
    logo.append(
        r"""
   ____            __              __        __
  / __ \__________/ /_  ___  _____/ /_____ _/ /____
 / / / / ___/ ___/ __ \/ _ \/ ___/ __/ __ `/ __/ _ \
/ /_/ / /  / /__/ / / /  __(__  ) /_/ /_/ / /_/  __/
\____/_/   \___/_/ /_/\___/____/\__/\__,_/\__/\___/

    ___                    __
   /   | ____ ____  ____  / /_
  / /| |/ __ `/ _ \/ __ \/ __/
 / ___ / /_/ /  __/ / / / /_
/_/  |_\__, /\___/_/ /_/\__/
      /____/
""",
        style="bold bright_cyan",
    )
    subtitle = Text(
        "Deterministic support triage | traps -> retrieval -> generation -> verification",
        style="bold bright_green",
        justify="center",
    )
    panel_text = Text.assemble(logo, "\n", subtitle)
    console.print(
        Panel(
            panel_text,
            border_style="bright_green",
            title="[bold bright_cyan]HackerRank Orchestrate[/]",
            subtitle="[bright_black]type exit or quit to leave[/]",
            padding=(1, 2),
        )
    )


@app.command()
def interactive(
    traces_dir: Optional[Path] = typer.Option(
        DEFAULT_TRACES_DIR,
        "--traces",
        help="Decision trace output dir for interactive tickets.",
    ),
) -> None:
    """Launch a Rich-powered support-ticket REPL."""

    draw_banner()
    console.print("[bright_black]Paste a support issue and press Enter.[/]")
    ticket_id = 1

    try:
        while True:
            try:
                issue = _ticket_input().strip()
            except EOFError:
                console.print("\n[bright_black]Input closed. Goodbye.[/]")
                break

            if not issue:
                continue
            if issue.casefold() in {"exit", "quit"}:
                console.print("[bright_black]Session closed. Goodbye.[/]")
                break

            # Slash-command parser placeholder: /mode, /domain, /help, etc.
            if issue.startswith("/"):
                console.print(
                    Panel(
                        "Slash commands are reserved for the next phase.",
                        border_style="yellow",
                        title="[bold yellow]Command Parser[/]",
                    )
                )
                continue

            try:
                decision = process_ticket(
                    {"issue": issue, "subject": "", "company": None},
                    ticket_id=f"interactive_{ticket_id}",
                    traces_dir=traces_dir,
                )
            except Exception as exc:
                console.print(
                    Panel(
                        f"{type(exc).__name__}: {exc}",
                        title="[bold red]Processing Error[/]",
                        border_style="red",
                    )
                )
                ticket_id += 1
                continue

            status_style = "bold red" if decision.status == "escalated" else "bold bright_green"
            border_style = "red" if decision.status == "escalated" else "bright_green"
            body = Text()
            body.append("Status: ", style="bold")
            body.append(decision.status, style=status_style)
            body.append("\nProduct Area: ", style="bold")
            body.append(decision.product_area)
            body.append("\nRequest Type: ", style="bold")
            body.append(decision.request_type)
            body.append("\n\nResponse\n", style="bold bright_cyan")
            body.append(decision.response)
            body.append("\n\nJustification\n", style="bold bright_cyan")
            body.append(decision.justification, style="bright_black")
            console.print(
                Panel(
                    body,
                    title="[bold]Triage Result[/]",
                    border_style=border_style,
                    padding=(1, 2),
                )
            )
            ticket_id += 1
    except KeyboardInterrupt:
        console.print("\n[bright_black]Interrupted. Goodbye.[/]")


def _ticket_input() -> str:
    """Read one ticket line with a stable Rich prompt across Rich versions."""

    return console.input("[bold bright_cyan]Ticket > [/]")


@app.command()
def run(
    input_csv: Path = typer.Option(DEFAULT_INPUT_CSV, "--input", "-i", help="Input ticket CSV."),
    output_csv: Path = typer.Option(DEFAULT_OUTPUT_CSV, "--out", "-o", help="Output prediction CSV."),
    traces_dir: Optional[Path] = typer.Option(
        DEFAULT_TRACES_DIR,
        "--traces",
        help="Decision trace output dir. Pass an empty path only if traces are not needed.",
    ),
) -> None:
    """Run the full triage pipeline over a ticket CSV."""

    try:
        import pandas as pd
    except ImportError as exc:
        typer.echo("run failed: pandas is required. Install it with `pip install pandas`.", err=True)
        raise typer.Exit(1) from exc

    try:
        frame = pd.read_csv(input_csv, keep_default_na=False)
    except Exception as exc:
        typer.echo(f"run failed: could not read input CSV {input_csv}: {exc}", err=True)
        raise typer.Exit(1) from exc

    output_rows: list[dict[str, str]] = []
    for index, row in frame.iterrows():
        row_id = index + 1
        row_dict = row.to_dict()
        try:
            decision = process_ticket(row_dict, ticket_id=row_id, traces_dir=traces_dir)
        except Exception as exc:
            typer.echo(f"ticket {row_id}: escalated after processing error: {exc}", err=True)
            decision = error_decision(row_dict, exc, ticket_id=row_id, traces_dir=traces_dir)

        output_rows.append(
            {
                "issue": _row_value(row_dict, "issue"),
                "subject": _row_value(row_dict, "subject"),
                "company": _row_value(row_dict, "company"),
                "response": decision.response,
                "product_area": decision.product_area,
                "status": decision.status,
                "request_type": decision.request_type,
                "justification": decision.justification,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        output_rows,
        columns=[
            "issue",
            "subject",
            "company",
            "response",
            "product_area",
            "status",
            "request_type",
            "justification",
        ],
    ).to_csv(output_csv, index=False)
    typer.echo(f"wrote {len(output_rows)} rows to {output_csv}")
    if traces_dir is not None:
        typer.echo(f"wrote traces to {traces_dir}")


def _row_value(row: dict[str, object], key: str) -> str:
    for row_key, value in row.items():
        if str(row_key).strip().casefold() == key:
            return "" if value is None else str(value)
    return ""


if __name__ == "__main__":
    app()
