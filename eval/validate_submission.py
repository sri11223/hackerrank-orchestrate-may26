"""Validate the generated HackerRank Orchestrate submission CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "support_tickets" / "output.csv"
EXPECTED_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]
EXPECTED_ROW_COUNT = 29
VALID_STATUSES = {"replied", "escalated"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate output.csv for final submission.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    failures = validate_output(args.output)
    if failures:
        print("submission validation FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("submission validation PASSED")
    print(f"rows={EXPECTED_ROW_COUNT}")
    print("columns=" + ",".join(EXPECTED_COLUMNS))
    print("statuses=replied,escalated")


def validate_output(output_path: Path) -> list[str]:
    failures: list[str] = []
    if not output_path.exists():
        return [f"output file does not exist: {output_path}"]

    frame = pd.read_csv(output_path, keep_default_na=False)
    columns = list(frame.columns)
    if columns != EXPECTED_COLUMNS:
        failures.append(f"columns mismatch: expected {EXPECTED_COLUMNS!r}, got {columns!r}")

    if len(frame) != EXPECTED_ROW_COUNT:
        failures.append(f"row count mismatch: expected {EXPECTED_ROW_COUNT}, got {len(frame)}")

    if "status" in frame.columns:
        invalid_statuses = sorted(set(frame["status"]) - VALID_STATUSES)
        if invalid_statuses:
            failures.append(
                "invalid status values: expected lowercase replied/escalated, got "
                + repr(invalid_statuses)
            )
    else:
        failures.append("missing required status column")

    if "response" in frame.columns:
        empty_rows = [
            str(index + 1)
            for index, value in enumerate(frame["response"])
            if not str(value).strip()
        ]
        if empty_rows:
            failures.append("empty/null responses at rows: " + ", ".join(empty_rows))
    else:
        failures.append("missing required response column")

    return failures


if __name__ == "__main__":
    main()
