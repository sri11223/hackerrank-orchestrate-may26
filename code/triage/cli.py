"""Package CLI for the support triage agent."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .config import (
    DATA_DIR,
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    LOG_PATH,
    SUPPORT_TICKETS_DIR,
    get_settings,
)
from .ingest import DEFAULT_OUTPUT_PATH, ingest_corpus
from .llm import ChatMessage, LLMClient, LLMConfigurationError, LLMResponseError, Provider
from .orchestrator import DEFAULT_TRACES_DIR, error_decision, process_ticket
from .retrieval import RetrievalDependencyError, retrieve
from .schema import TriageDecision


app = typer.Typer(
    add_completion=False,
    help="Support triage agent for HackerRank, Claude, and Visa tickets.",
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()
COMPANY_MODES = {
    "auto": None,
    "none": None,
    "hackerrank": "HackerRank",
    "hacker rank": "HackerRank",
    "claude": "Claude",
    "anthropic": "Claude",
    "visa": "Visa",
}


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
    _print_home_commands()


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
    current_company: str | None = None

    try:
        while True:
            try:
                issue = _ticket_input(current_company).strip()
            except EOFError:
                console.print("\n[bright_black]Input closed. Goodbye.[/]")
                break

            if not issue:
                continue
            if issue.casefold() in {"exit", "quit"}:
                console.print("[bright_black]Session closed. Goodbye.[/]")
                break

            if issue.startswith("/"):
                should_continue, current_company = _handle_repl_command(issue, current_company)
                if not should_continue:
                    break
                continue

            try:
                with console.status(
                    "[bold cyan]Analyzing intent and retrieving hybrid vectors...",
                    spinner="dots",
                ):
                    decision = process_ticket(
                        {"issue": issue, "subject": "", "company": current_company},
                        ticket_id=f"interactive_{ticket_id}",
                        traces_dir=traces_dir,
                    )
            except Exception as exc:
                decision = error_decision(
                    {"issue": issue, "subject": "", "company": current_company},
                    exc,
                    ticket_id=f"interactive_{ticket_id}",
                    traces_dir=traces_dir,
                )
                console.print(
                    Panel(
                        f"{type(exc).__name__}: {exc}",
                        title="[bold red]Processing Error[/]",
                        border_style="red",
                    )
                )
                console.print(_decision_panel(decision))
                ticket_id += 1
                continue

            console.print(_decision_panel(decision))
            ticket_id += 1
    except KeyboardInterrupt:
        console.print("\n[bright_black]Interrupted. Goodbye.[/]")


def _ticket_input(current_company: str | None) -> str:
    """Read one ticket line with a stable Rich prompt across Rich versions."""

    return console.input(_shell_prompt(current_company))


def _shell_prompt(current_company: str | None) -> Text:
    mode = current_company or "Auto"
    mode_style = "bold bright_green" if current_company else "bold yellow"
    prompt = Text()
    prompt.append("[", style="bright_black")
    prompt.append("Orchestrate", style="bold bright_cyan")
    prompt.append(": ", style="bright_black")
    prompt.append(mode, style=mode_style)
    prompt.append("] > ", style="bright_black")
    return prompt


def _handle_repl_command(raw_command: str, current_company: str | None) -> tuple[bool, str | None]:
    """Handle slash commands and return ``(continue_loop, updated_company)``."""

    command_line = raw_command.strip()
    command, _, argument = command_line.partition(" ")
    command = command.casefold()
    argument = argument.strip()

    if command in {"/exit", "/quit"}:
        console.print("[bright_black]Session closed. Goodbye.[/]")
        return False, current_company

    if command == "/help":
        _print_help_table()
        return True, current_company

    if command == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        draw_banner()
        return True, current_company

    if command == "/mode":
        if not argument:
            console.print("[bold red]Usage:[/] /mode auto|Visa|HackerRank|Claude")
            return True, current_company

        normalized = " ".join(argument.casefold().split())
        if normalized not in COMPANY_MODES:
            console.print("[bold red]Unknown company.[/] Type [bold]/help[/] for options.")
            return True, current_company

        updated_company = COMPANY_MODES[normalized]
        label = updated_company or "Auto"
        console.print(f"[bold bright_green]Mode switched to {label}[/]")
        return True, updated_company

    console.print("[bold red]Unknown command. Type /help for options.[/]")
    return True, current_company


def _print_help_table() -> None:
    table = Table(
        title="Orchestrate Agent Commands",
        title_style="bold bright_cyan",
        border_style="bright_green",
        header_style="bold bright_green",
        show_lines=True,
    )
    table.add_column("Command", style="bold bright_cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Example", style="bright_black")
    table.add_row("/help", "Show this command menu.", "/help")
    table.add_row("/clear", "Clear the terminal and redraw the banner.", "/clear")
    table.add_row(
        "/mode <company>",
        "Force retrieval/generation domain, or return to automatic domain resolution.",
        "/mode Visa  |  /mode auto",
    )
    table.add_row("/exit", "Leave the interactive session.", "/exit")
    table.add_row("/quit", "Leave the interactive session.", "/quit")
    console.print(table)


def _print_home_commands() -> None:
    command_text = Text()
    command_text.append("Commands: ", style="bold bright_green")
    command_text.append("/help", style="bold bright_cyan")
    command_text.append("  ")
    command_text.append("/clear", style="bold bright_cyan")
    command_text.append("  ")
    command_text.append("/mode Visa|HackerRank|Claude|auto", style="bold bright_cyan")
    command_text.append("  ")
    command_text.append("/quit", style="bold bright_cyan")
    console.print(
        Panel(
            command_text,
            border_style="bright_black",
            padding=(0, 2),
        )
    )


def _decision_panel(decision: TriageDecision) -> Panel:
    status_style = "bold red" if decision.status == "escalated" else "bold bright_green"
    border_style = "red" if decision.status == "escalated" else "bright_green"

    metadata = Table.grid(padding=(0, 1), expand=True)
    metadata.add_column(justify="right", style="bold white", no_wrap=True)
    metadata.add_column(ratio=1)
    metadata.add_row("Status:", Text(decision.status.upper(), style=status_style))
    metadata.add_row("Product Area:", Text(decision.product_area, style="cyan"))
    metadata.add_row("Request Type:", Text(decision.request_type, style="yellow"))
    metadata.add_row("Justification:", Text(decision.justification, style="bright_black"))

    return Panel(
        Group(
            metadata,
            Text("-" * 72, style=border_style),
            Markdown(_response_with_receipt(decision)),
        ),
        title="[bold bright_cyan]ORCHESTRATE DECISION[/]",
        border_style=border_style,
        padding=(1, 2),
    )


def _response_with_receipt(decision: TriageDecision) -> str:
    exact_quote = " ".join(decision.exact_quote.split())
    if not exact_quote:
        return decision.response
    return f'{decision.response}\n\n> **Source Receipt:** *"{exact_quote}"*'


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

    _run_batch_csv(input_csv=input_csv, output_csv=output_csv, traces_dir=traces_dir)


@app.command()
def crucible(
    input_csv: Path = typer.Option(
        SUPPORT_TICKETS_DIR / "crucible_tickets.csv",
        "--input",
        "-i",
        help="Adversarial crucible input CSV.",
    ),
    output_csv: Path = typer.Option(
        SUPPORT_TICKETS_DIR / "crucible_output.csv",
        "--out",
        "-o",
        help="Adversarial crucible output CSV.",
    ),
    traces_dir: Optional[Path] = typer.Option(
        DEFAULT_TRACES_DIR / "crucible",
        "--traces",
        help="Decision trace output dir for crucible runs.",
    ),
) -> None:
    """Run the adversarial crucible dataset through the full batch pipeline."""

    _run_batch_csv(input_csv=input_csv, output_csv=output_csv, traces_dir=traces_dir)


def _run_batch_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    traces_dir: Optional[Path],
) -> None:
    """Run the full triage pipeline over a ticket CSV with Rich progress."""

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
    replied_count = 0
    escalated_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            _batch_progress_description(replied_count, escalated_count),
            total=len(frame),
        )
        for index, row in frame.iterrows():
            row_id = index + 1
            row_dict = row.to_dict()
            try:
                decision = process_ticket(row_dict, ticket_id=row_id, traces_dir=traces_dir)
            except Exception as exc:
                progress.console.print(
                    f"[bold red]ticket {row_id}: escalated after processing error:[/] {exc}"
                )
                decision = error_decision(row_dict, exc, ticket_id=row_id, traces_dir=traces_dir)

            if decision.status == "replied":
                replied_count += 1
            else:
                escalated_count += 1

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
            progress.update(
                task_id,
                advance=1,
                description=_batch_progress_description(replied_count, escalated_count),
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


def _batch_progress_description(replied_count: int, escalated_count: int) -> str:
    return f"Processing Tickets | Replied: {replied_count} | Escalated: {escalated_count}"


def _row_value(row: dict[str, object], key: str) -> str:
    for row_key, value in row.items():
        if str(row_key).strip().casefold() == key:
            return "" if value is None else str(value)
    return ""


if __name__ == "__main__":
    app()
