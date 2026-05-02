"""Package CLI for the support triage agent."""

from __future__ import annotations

import os
import sys
import time
import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .config import (
    CODE_DIR,
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


@app.command()
def explain(
    ticket_id: int = typer.Argument(..., help="Ticket id to explain from JSON trace sidecars."),
    traces_dir: Optional[Path] = typer.Option(
        None,
        "--traces",
        help=(
            "Trace directory to search. Defaults to packaged production traces, "
            "then crucible and root traces."
        ),
    ),
) -> None:
    """Render a human-auditable decision path for one ticket trace."""

    trace_path = _find_trace_file(ticket_id, traces_dir)
    if trace_path is None:
        console.print(
            Panel(
                f"Could not find a trace for ticket {ticket_id}. "
                "Try --traces code/traces/production or --traces code/traces/crucible.",
                title="[bold red]Trace Not Found[/]",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(
            Panel(
                f"{type(exc).__name__}: {exc}",
                title="[bold red]Trace Load Failed[/]",
                border_style="red",
            )
        )
        raise typer.Exit(1) from exc

    _render_explanation(trace, trace_path)


def _find_trace_file(ticket_id: int, traces_dir: Optional[Path]) -> Path | None:
    """Find the newest sidecar for a ticket id under the trace naming scheme."""

    if traces_dir is not None and traces_dir.is_file():
        return traces_dir

    for directory in _trace_search_dirs(traces_dir):
        if not directory.exists():
            continue
        matches: list[Path] = []
        for pattern in (f"ticket_{ticket_id}.json", f"ticket_{ticket_id}_*.json"):
            matches.extend(path for path in directory.rglob(pattern) if path.is_file())
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)
    return None


def _trace_search_dirs(traces_dir: Optional[Path]) -> list[Path]:
    if traces_dir is not None:
        return [traces_dir]

    candidates = [
        CODE_DIR / "traces" / "production",
        CODE_DIR / "traces" / "crucible",
        CODE_DIR / "traces",
        DEFAULT_TRACES_DIR / "production",
        DEFAULT_TRACES_DIR / "crucible",
        DEFAULT_TRACES_DIR,
        Path("traces"),
    ]
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(candidate)
    return deduped


def _render_explanation(trace: dict[str, Any], trace_path: Path) -> None:
    stages = trace.get("stages", {})
    if not isinstance(stages, dict):
        stages = {}

    final_decision = trace.get("final_decision", {})
    if not isinstance(final_decision, dict):
        final_decision = {}

    console.print(
        Panel(
            f"[bold]Ticket:[/] {trace.get('ticket_id', 'unknown')}\n"
            f"[bold]Trace:[/] {trace_path}\n"
            f"[bold]Trace ID:[/] {trace.get('trace_id', 'unknown')}",
            title="[bold bright_cyan]ORCHESTRATE AUDIT EXPLAINER[/]",
            border_style="bright_cyan",
        )
    )
    console.print(_grounding_panel(stages))
    console.print(_final_decision_table(final_decision))

    tree = Tree("[bold bright_cyan]Decision Path[/]")
    _add_sanitize_node(tree, stages)
    _add_trap_node(tree, stages)
    _add_retrieval_node(tree, stages)
    _add_verifier_node(tree, stages)
    console.print(tree)


def _grounding_panel(stages: dict[str, Any]) -> Panel:
    retrieval = stages.get("retrieval", {})
    chunks = retrieval.get("chunks", []) if isinstance(retrieval, dict) else []
    top_chunk = chunks[0] if chunks and isinstance(chunks[0], dict) else {}
    source = str(top_chunk.get("url_or_file") or top_chunk.get("chunk_id") or "the retrieved corpus")
    confidence = _score_text(retrieval.get("top_score") if isinstance(retrieval, dict) else None)
    message = (
        f"This decision was 100% grounded in [bold cyan]{source}[/] "
        f"with a confidence score of [bold bright_green]{confidence}[/]."
    )
    return Panel(message, title="[bold bright_green]Aha Moment[/]", border_style="bright_green")


def _final_decision_table(final_decision: dict[str, Any]) -> Table:
    status = str(final_decision.get("status") or "unknown")
    status_style = "bold red" if status == "escalated" else "bold bright_green"

    table = Table(
        title="Final Decision",
        title_style="bold bright_cyan",
        box=box.SIMPLE_HEAVY,
        border_style="bright_black",
        show_header=False,
    )
    table.add_column("Field", style="bold white", no_wrap=True)
    table.add_column("Value")
    table.add_row("Status", Text(status.upper(), style=status_style))
    table.add_row("Product Area", Text(str(final_decision.get("product_area") or ""), style="cyan"))
    table.add_row("Request Type", Text(str(final_decision.get("request_type") or ""), style="yellow"))
    table.add_row("Justification", str(final_decision.get("justification") or ""))
    quote = " ".join(str(final_decision.get("exact_quote") or "").split())
    if quote:
        table.add_row("Source Receipt", _truncate(quote, 220))
    return table


def _add_sanitize_node(tree: Tree, stages: dict[str, Any]) -> None:
    sanitize = stages.get("sanitize", {})
    if not isinstance(sanitize, dict):
        tree.add("[bold yellow]Stage 1 - Sanitize[/]: missing")
        return

    pii_detected = bool(sanitize.get("pii_detected"))
    pii_label = "[bold red]YES[/]" if pii_detected else "[bold bright_green]NO[/]"
    node = tree.add("[bold bright_cyan]Stage 1 - Sanitize[/]")
    node.add(f"PII redacted: {pii_label}")
    node.add(f"Language: [cyan]{sanitize.get('language', 'unknown')}[/]")
    node.add(f"Company: [cyan]{sanitize.get('company') or 'Auto'}[/]")


def _add_trap_node(tree: Tree, stages: dict[str, Any]) -> None:
    trap = stages.get("trap_classifier", {})
    if not isinstance(trap, dict):
        tree.add("[bold yellow]Stage 2 - Trap Classifier[/]: missing")
        return

    tags = trap.get("tags") or []
    tag_text = ", ".join(str(tag) for tag in tags) or "unknown"
    node = tree.add(f"[bold bright_cyan]Stage 2 - Trap Classifier[/]: [bold yellow]{tag_text}[/]")
    node.add(str(trap.get("reasoning") or "No reasoning recorded."))


def _add_retrieval_node(tree: Tree, stages: dict[str, Any]) -> None:
    retrieval = stages.get("retrieval", {})
    if not isinstance(retrieval, dict):
        tree.add("[bold yellow]Stage 4 - Hybrid Retrieval[/]: missing")
        return

    top_score = _score_text(retrieval.get("top_score"))
    node = tree.add(f"[bold bright_cyan]Stage 4 - Hybrid Retrieval[/]: top confidence [bold]{top_score}[/]")
    node.add(_retrieval_table(retrieval))


def _retrieval_table(retrieval: dict[str, Any]) -> Table:
    table = Table(
        title="Top 3 Retrieved Chunks",
        title_style="bold bright_green",
        box=box.SIMPLE,
        border_style="bright_black",
    )
    table.add_column("#", justify="right", style="bright_black", no_wrap=True)
    table.add_column("Chunk", style="bold cyan", no_wrap=True)
    table.add_column("Source / Heading", ratio=2)
    table.add_column("BM25", justify="right")
    table.add_column("Dense", justify="right")
    table.add_column("Fused", justify="right")

    chunks = retrieval.get("chunks") or []
    for index, chunk in enumerate(chunks[:3], start=1):
        if not isinstance(chunk, dict):
            continue
        source = chunk.get("url_or_file") or ""
        heading = chunk.get("heading_path") or ""
        table.add_row(
            str(index),
            str(chunk.get("chunk_id") or ""),
            f"{source}\n[bright_black]{_truncate(str(heading), 140)}[/]",
            _score_text(chunk.get("bm25_score")),
            _score_text(chunk.get("dense_score")),
            _score_text(chunk.get("normalized_score")),
        )
    if not table.rows:
        table.add_row("-", "none", "No retrieval chunks recorded.", "-", "-", "-")
    return table


def _add_verifier_node(tree: Tree, stages: dict[str, Any]) -> None:
    generation = stages.get("generation", {})
    if not isinstance(generation, dict) or not generation:
        handler = stages.get("handler", {})
        mode = handler.get("mode") if isinstance(handler, dict) else "unknown"
        tree.add(
            "[bold bright_cyan]Stage 7 - Verifier[/]: "
            f"bypassed by deterministic handler ([cyan]{mode}[/])"
        )
        return

    drafts = generation.get("drafts") or []
    if not drafts:
        tree.add("[bold yellow]Stage 7 - Verifier[/]: generation recorded without drafts")
        return

    final_verifier = drafts[-1].get("verifier", {}) if isinstance(drafts[-1], dict) else {}
    final_safe = final_verifier.get("safe") if isinstance(final_verifier, dict) else None
    status = "[bold bright_green]safe[/]" if final_safe else "[bold red]unsafe[/]"

    if any(isinstance(draft, dict) and draft.get("mode") == "self_healing_rewrite" for draft in drafts):
        node = tree.add(f"[bold bright_cyan]Stage 7 - Verifier[/]: self-healed, final {status}")
        node.add(_self_healing_table(drafts))
        return

    if any(isinstance(draft, dict) and draft.get("mode") == "second_opinion" for draft in drafts):
        node = tree.add(f"[bold bright_cyan]Stage 7 - Verifier[/]: second opinion, final {status}")
        node.add(_second_opinion_table(drafts))
        return

    verifier = drafts[0].get("verifier", {}) if isinstance(drafts[0], dict) else {}
    node = tree.add(f"[bold bright_cyan]Stage 7 - Verifier[/]: {status}")
    if isinstance(verifier, dict):
        issues = verifier.get("issues") or []
        node.add("Issues: " + (", ".join(str(issue) for issue in issues) if issues else "none"))


def _self_healing_table(drafts: list[Any]) -> Table:
    rewrite_index = next(
        (
            index
            for index, draft in enumerate(drafts)
            if isinstance(draft, dict) and draft.get("mode") == "self_healing_rewrite"
        ),
        1,
    )
    first = drafts[max(0, rewrite_index - 1)] if drafts and isinstance(drafts[max(0, rewrite_index - 1)], dict) else {}
    second = drafts[rewrite_index] if len(drafts) > rewrite_index and isinstance(drafts[rewrite_index], dict) else {}
    first_generation = first.get("generation", {}) if isinstance(first.get("generation"), dict) else {}
    first_verifier = first.get("verifier", {}) if isinstance(first.get("verifier"), dict) else {}
    second_generation = second.get("generation", {}) if isinstance(second.get("generation"), dict) else {}

    critique = second.get("critique") or first_verifier.get("issues") or []
    critique_text = "\n".join(f"- {item}" for item in critique) if critique else "No critique recorded."

    table = Table(
        title="Self-Healing Rewrite",
        title_style="bold yellow",
        box=box.SIMPLE_HEAVY,
        border_style="yellow",
    )
    table.add_column("Original Draft", ratio=1)
    table.add_column("Verifier Critique", ratio=1)
    table.add_column("Rewrite", ratio=1)
    table.add_row(
        _truncate(str(first_generation.get("response") or ""), 520),
        _truncate(critique_text, 520),
        _truncate(str(second_generation.get("response") or ""), 520),
    )
    return table


def _second_opinion_table(drafts: list[Any]) -> Table:
    cheap = next(
        (
            draft
            for draft in drafts
            if isinstance(draft, dict) and draft.get("mode") == "cheap_initial_second_opinion_requested"
        ),
        {},
    )
    second = next(
        (
            draft
            for draft in drafts
            if isinstance(draft, dict) and draft.get("mode") == "second_opinion"
        ),
        {},
    )
    cheap_generation = cheap.get("generation", {}) if isinstance(cheap.get("generation"), dict) else {}
    second_generation = second.get("generation", {}) if isinstance(second.get("generation"), dict) else {}
    second_verifier = second.get("verifier", {}) if isinstance(second.get("verifier"), dict) else {}

    table = Table(
        title="Second Opinion Gate",
        title_style="bold bright_cyan",
        box=box.SIMPLE_HEAVY,
        border_style="bright_cyan",
    )
    table.add_column("Cheap Draft", ratio=1)
    table.add_column("Strong Model Draft", ratio=1)
    table.add_column("Verifier", ratio=1)
    table.add_row(
        _truncate(str(cheap_generation.get("response") or ""), 420),
        _truncate(str(second_generation.get("response") or ""), 420),
        "safe=" + str(second_verifier.get("safe")) + "\nissues="
        + _truncate(", ".join(str(issue) for issue in second_verifier.get("issues", [])), 300),
    )
    return table


def _score_text(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _truncate(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


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
                    decision = asyncio.run(
                        process_ticket(
                            {"issue": issue, "subject": "", "company": current_company},
                            ticket_id=f"interactive_{ticket_id}",
                            traces_dir=traces_dir,
                        )
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

    started = time.perf_counter()
    try:
        metrics = asyncio.run(
            _run_batch_csv_async(frame=frame, output_csv=output_csv, traces_dir=traces_dir)
        )
    except KeyboardInterrupt:
        console.print("\n[bright_black]Batch interrupted.[/]")
        raise typer.Exit(130) from None

    elapsed = metrics.get("elapsed_seconds", time.perf_counter() - started)
    time_per_ticket = metrics.get("time_per_ticket_seconds", 0.0)
    typer.echo(f"wrote {metrics['row_count']} rows to {output_csv}")
    if traces_dir is not None:
        typer.echo(f"wrote traces to {traces_dir}")
    typer.echo(f"total time elapsed: {elapsed:.2f}s")
    typer.echo(f"time per ticket: {time_per_ticket:.2f}s")


async def _run_batch_csv_async(
    *,
    frame: Any,
    output_csv: Path,
    traces_dir: Optional[Path],
) -> dict[str, float | int]:
    """Process tickets concurrently while updating Rich progress on completion."""

    import pandas as pd

    output_rows: list[dict[str, str] | None] = [None] * len(frame)
    replied_count = 0
    escalated_count = 0
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(5)

    async def process_row(index: int, row: Any) -> tuple[int, dict[str, object], TriageDecision]:
        row_id = index + 1
        row_dict = row.to_dict()
        async with semaphore:
            try:
                decision = await process_ticket(row_dict, ticket_id=row_id, traces_dir=traces_dir)
            except Exception as exc:
                console.print(
                    f"[bold red]ticket {row_id}: escalated after processing error:[/] {exc}"
                )
                decision = error_decision(row_dict, exc, ticket_id=row_id, traces_dir=traces_dir)
            return index, row_dict, decision

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
        tasks = [
            asyncio.create_task(process_row(index, row))
            for index, row in frame.iterrows()
        ]
        for completed in asyncio.as_completed(tasks):
            index, row_dict, decision = await completed

            if decision.status == "replied":
                replied_count += 1
            else:
                escalated_count += 1

            output_rows[index] = {
                "issue": _row_value(row_dict, "issue"),
                "subject": _row_value(row_dict, "subject"),
                "company": _row_value(row_dict, "company"),
                "response": decision.response,
                "product_area": decision.product_area,
                "status": decision.status,
                "request_type": decision.request_type,
                "justification": decision.justification,
            }
            progress.update(
                task_id,
                advance=1,
                description=_batch_progress_description(replied_count, escalated_count),
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [row for row in output_rows if row is not None],
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
    elapsed = time.perf_counter() - started
    return {
        "row_count": len([row for row in output_rows if row is not None]),
        "elapsed_seconds": elapsed,
        "time_per_ticket_seconds": elapsed / len(frame) if len(frame) else 0.0,
    }


def _batch_progress_description(replied_count: int, escalated_count: int) -> str:
    return f"Processing Tickets | Replied: {replied_count} | Escalated: {escalated_count}"


def _row_value(row: dict[str, object], key: str) -> str:
    for row_key, value in row.items():
        if str(row_key).strip().casefold() == key:
            return "" if value is None else str(value)
    return ""


if __name__ == "__main__":
    app()
