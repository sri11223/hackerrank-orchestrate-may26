"""Repo-root import shim for ``python -m triage.cli``.

The submission code lives under ``code/triage``. This package extends its module
search path to that implementation directory so the hackathon-friendly command
works from the repository root without requiring PYTHONPATH setup.
"""

from pathlib import Path

_CODE_TRIAGE = Path(__file__).resolve().parents[1] / "code" / "triage"
if _CODE_TRIAGE.is_dir():
    __path__.append(str(_CODE_TRIAGE))  # type: ignore[name-defined]

__version__ = "0.1.0"
