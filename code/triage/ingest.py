"""Stage 4 corpus ingestion: raw support docs -> semantic JSONL chunks.

The retrieval indexes are intentionally out of scope for this step. This module
only builds the clean, deterministic chunk file that BM25 and dense retrieval
will consume later.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from .config import DATA_DIR


DOMAINS: tuple[str, ...] = ("hackerrank", "claude", "visa")
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".html", ".htm", ".txt"})
DEFAULT_OUTPUT_PATH = DATA_DIR / "processed" / "chunks.jsonl"
MAX_SECTION_TOKENS = 400
WINDOW_OVERLAP_TOKENS = 60

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOKEN_RE = re.compile(r"\S+")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)


@dataclass(frozen=True)
class Section:
    """A heading-aware unit before long-section token windowing."""

    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class IngestStats:
    """Summary returned to the CLI after writing chunks."""

    corpus_root: Path
    output_path: Path
    files_read: int
    chunks_written: int


class _HTMLToMarkdown(HTMLParser):
    """Lossy but deterministic HTML-to-heading-text converter.

    We do not need browser-perfect rendering for retrieval. We need stable text,
    real heading boundaries, and low noise. The parser keeps heading structure
    as Markdown-style ``#`` lines so the same splitter handles HTML and MD.
    """

    _block_tags = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "form",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n" + ("#" * int(tag[1])) + " ")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in self._block_tags or tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def ingest_corpus(
    corpus_dir: Path = DATA_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> IngestStats:
    """Read the local corpus and write deterministic JSONL chunks.

    ``corpus_dir`` may be either ``data/`` from the starter repo or ``data/raw/``
    from the architecture document. If ``data/raw`` exists below the supplied
    directory, it wins; otherwise we ingest the supplied directory directly.
    """

    source_root = resolve_corpus_root(corpus_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files_read = 0
    chunks_written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for domain in DOMAINS:
            domain_root = source_root / domain
            for file_path in iter_corpus_files(domain_root):
                files_read += 1
                for record in chunk_file(domain_root, file_path, domain, chunks_written):
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    chunks_written += 1

    return IngestStats(
        corpus_root=source_root,
        output_path=output_path,
        files_read=files_read,
        chunks_written=chunks_written,
    )


def resolve_corpus_root(corpus_dir: Path) -> Path:
    """Find the directory that directly contains hackerrank/claude/visa."""

    root = corpus_dir.resolve()
    raw = root / "raw"
    if _has_domain_dirs(raw):
        return raw
    if _has_domain_dirs(root):
        return root
    expected = ", ".join(DOMAINS)
    raise FileNotFoundError(
        f"Could not find corpus domain directories ({expected}) under {root} or {raw}"
    )


def _has_domain_dirs(path: Path) -> bool:
    return path.is_dir() and all((path / domain).is_dir() for domain in DOMAINS)


def iter_corpus_files(domain_root: Path) -> Iterable[Path]:
    """Yield supported corpus files in stable path order."""

    for path in sorted(domain_root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def chunk_file(
    domain_root: Path,
    file_path: Path,
    domain: str,
    chunk_offset: int = 0,
) -> list[dict[str, str]]:
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    metadata, body = split_frontmatter(raw)
    clean_text = normalize_document_text(body, file_path.suffix)
    sections = split_heading_sections(clean_text, fallback_title=metadata_title(metadata, file_path))
    base_path = base_heading_path(domain_root, file_path, metadata)
    url_or_file = metadata.get("source_url") or metadata.get("final_url") or metadata.get("url")
    if not url_or_file:
        url_or_file = file_path.relative_to(domain_root).as_posix()

    records: list[dict[str, str]] = []
    for section_index, section in enumerate(sections):
        heading_path = join_heading_path([*base_path, *section.heading_path])
        for window_index, window_text in enumerate(window_section_text(section.text)):
            chunk_id = make_chunk_id(domain, file_path, section_index, window_index, chunk_offset + len(records))
            records.append(
                {
                    "chunk_id": chunk_id,
                    "domain": domain,
                    "url_or_file": url_or_file,
                    "heading_path": heading_path,
                    "text": window_text,
                }
            )
    return records


def split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Parse the simple YAML frontmatter used by the corpus exports."""

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    frontmatter = match.group(1)
    metadata: dict[str, str] = {}
    current_list_key: str | None = None
    list_values: list[str] = []

    for line in frontmatter.splitlines():
        if line.startswith("  - ") and current_list_key:
            list_values.append(_strip_quotes(line[4:].strip()))
            metadata[current_list_key] = " > ".join(list_values)
            continue

        current_list_key = None
        list_values = []
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value == "":
            current_list_key = key
            metadata[key] = ""
        else:
            metadata[key] = _strip_quotes(value)

    return metadata, raw[match.end() :]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def normalize_document_text(body: str, suffix: str) -> str:
    """Remove markup noise while preserving headings and readable prose."""

    text = unescape(body.replace("\r\n", "\n").replace("\r", "\n"))
    if suffix.lower() in {".html", ".htm"} or re.search(r"<h[1-6]\b|<html\b|<body\b", text, re.I):
        parser = _HTMLToMarkdown()
        parser.feed(text)
        text = parser.text()

    # Markdown links/images often contain long signed asset URLs. Keep the
    # human-visible anchor/alt text and drop the retrieval-hostile URL payload.
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:p|div|section|article|li|ul|ol|table|tr|td|th)[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def split_heading_sections(text: str, fallback_title: str) -> list[Section]:
    """Split by Markdown/HTML headings before any token-window fallback."""

    sections: list[Section] = []
    heading_stack: list[str] = []
    current_path: tuple[str, ...] = (fallback_title,)
    buffer: list[str] = []

    def flush() -> None:
        content = normalize_section_block("\n".join(buffer))
        if content:
            sections.append(Section(heading_path=current_path, text=content))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if not match:
            buffer.append(line)
            continue

        flush()
        level = len(match.group(1))
        heading = normalize_inline_text(match.group(2))
        heading_stack[:] = heading_stack[: max(0, level - 1)]
        heading_stack.append(heading)
        current_path = tuple(heading_stack) or (fallback_title,)
        buffer = [heading]

    flush()
    if sections:
        return sections

    fallback_text = normalize_section_block(text)
    return [Section(heading_path=(fallback_title,), text=fallback_text)] if fallback_text else []


def normalize_section_block(text: str) -> str:
    lines = [normalize_inline_text(line) for line in text.splitlines()]
    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        compact_lines.append(line)
        previous_blank = blank
    return "\n".join(compact_lines).strip()


def normalize_inline_text(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" #\t")


def metadata_title(metadata: dict[str, str], file_path: Path) -> str:
    return metadata.get("title") or prettify_slug(file_path.stem)


def base_heading_path(domain_root: Path, file_path: Path, metadata: dict[str, str]) -> list[str]:
    breadcrumbs = metadata.get("breadcrumbs")
    if breadcrumbs:
        return [part.strip() for part in breadcrumbs.split(">") if part.strip()]

    relative_parent = file_path.relative_to(domain_root).parent
    if str(relative_parent) == ".":
        return []
    return [prettify_slug(part) for part in relative_parent.parts]


def prettify_slug(value: str) -> str:
    value = re.sub(r"^\d+[-_]+", "", value)
    value = value.replace("%2C", ",").replace("%27", "'")
    value = re.sub(r"[-_]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().title()


def join_heading_path(parts: Iterable[str]) -> str:
    """Join path pieces while removing adjacent duplicates."""

    joined: list[str] = []
    for part in parts:
        clean = normalize_inline_text(part)
        if not clean:
            continue
        if joined and joined[-1].casefold() == clean.casefold():
            continue
        joined.append(clean)
    return " > ".join(joined)


def window_section_text(text: str) -> list[str]:
    """Return the section as-is unless it exceeds the 400-token budget."""

    tokens = list(_TOKEN_RE.finditer(text))
    if len(tokens) <= MAX_SECTION_TOKENS:
        return [text]

    windows: list[str] = []
    start = 0
    step = MAX_SECTION_TOKENS - WINDOW_OVERLAP_TOKENS
    while start < len(tokens):
        end = min(len(tokens), start + MAX_SECTION_TOKENS)
        char_start = tokens[start].start()
        char_end = tokens[end - 1].end()
        windows.append(text[char_start:char_end].strip())
        if end == len(tokens):
            break
        start += step
    return windows


def make_chunk_id(
    domain: str,
    file_path: Path,
    section_index: int,
    window_index: int,
    ordinal: int,
) -> str:
    fingerprint_input = f"{domain}|{file_path.as_posix()}|{section_index}|{window_index}"
    fingerprint = hashlib.sha1(fingerprint_input.encode("utf-8")).hexdigest()[:10]
    return f"{domain}_{ordinal:05d}_{fingerprint}"
