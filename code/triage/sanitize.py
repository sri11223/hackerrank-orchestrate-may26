"""Stage 1 ticket sanitation and language detection."""

from __future__ import annotations

import re
import unicodedata

from .schema import Ticket


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_CC_RE = re.compile(r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PHONE_RE = re.compile(
    r"(?<![\d\w])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?![\d\w])"
)
_ZERO_WIDTH_AND_BIDI = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}


def scrub_pii(text: str) -> str:
    """Replace sensitive customer identifiers before any external LLM call."""

    scrubbed = str(text or "")
    scrubbed = _EMAIL_RE.sub("[EMAIL_REDACTED]", scrubbed)
    scrubbed = _CC_RE.sub("[CC_REDACTED]", scrubbed)
    scrubbed = _SSN_RE.sub("[SSN_REDACTED]", scrubbed)
    scrubbed = _PHONE_RE.sub("[PHONE_REDACTED]", scrubbed)
    return scrubbed


def normalize_text(value: str | None) -> str:
    """Normalize unicode, remove hostile controls, and compact whitespace.

    We keep ordinary newlines because ticket bodies often contain useful lists
    or copied error messages. Other control/format characters are converted to
    spaces so malicious invisible text cannot distort downstream prompts.
    """

    if value is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(value))
    cleaned_chars: list[str] = []
    for char in normalized:
        if char in _ZERO_WIDTH_AND_BIDI:
            continue
        if char in {"\n", "\t"}:
            cleaned_chars.append(char)
            continue
        category = unicodedata.category(char)
        if category.startswith("C"):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(char)

    cleaned = "".join(cleaned_chars).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_language(text: str) -> str:
    """Return a deterministic ISO language code, or ``unknown`` on weak input."""

    compact = " ".join(text.split())
    if len(compact) < 8:
        return "unknown"

    try:
        from langdetect import DetectorFactory, LangDetectException, detect  # type: ignore

        DetectorFactory.seed = 7
        return detect(compact)
    except ImportError:
        return "unknown"
    except LangDetectException:
        return "unknown"


def sanitize_ticket(issue: str | None, subject: str | None = "", company: str | None = None) -> Ticket:
    """Build the normalized ``Ticket`` schema used by the pipeline."""

    normalized_issue = normalize_text(issue)
    normalized_subject = normalize_text(subject)
    clean_issue = scrub_pii(normalized_issue)
    clean_subject = scrub_pii(normalized_subject)
    pii_detected = clean_issue != normalized_issue or clean_subject != normalized_subject
    clean_company = normalize_text(company)
    if clean_company.casefold() in {"", "none", "null", "nan", "n/a", "unknown"}:
        clean_company = ""

    language = detect_language("\n".join(part for part in (clean_subject, clean_issue) if part))
    return Ticket(
        issue=clean_issue,
        subject=clean_subject,
        company=clean_company or None,
        language=language,
        pii_detected=pii_detected,
    )
