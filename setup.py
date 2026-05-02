"""Installable CLI package for the Orchestrate support triage agent."""

from __future__ import annotations

import re
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
REQUIREMENTS = ROOT / "code" / "requirements.txt"


def _read_requirements() -> list[str]:
    requirements: list[str] = []
    if REQUIREMENTS.exists():
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                requirements.append(cleaned)

    # The batch CSV runner imports pandas lazily so lightweight commands can
    # start quickly, but global installs still need it for `orchestrate run`.
    if not any(_requirement_name(item) == "pandas" for item in requirements):
        requirements.append("pandas>=2.2,<3")
    return requirements


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().casefold()


setup(
    name="orchestrate-triage-agent",
    version="0.1.0",
    description="Deterministic, grounded support triage CLI for HackerRank Orchestrate.",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="HackerRank Orchestrate Team",
    python_requires=">=3.10",
    package_dir={"": "code"},
    packages=find_packages(where="code"),
    include_package_data=True,
    install_requires=_read_requirements(),
    entry_points={
        "console_scripts": [
            "orchestrate=triage.cli:main",
            "triage=triage.cli:main",
            "triage-ai=triage.cli:main",
        ],
    },
)
