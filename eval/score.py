"""Hard-field scorer for the 10-row sample support ticket set."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
DEFAULT_PREDICTIONS = REPO_ROOT / "support_tickets" / "output.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score hard CSV fields against sample labels.")
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--pred", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    expected = pd.read_csv(args.expected, keep_default_na=False)
    predictions = pd.read_csv(args.pred, keep_default_na=False)

    metrics = [
        ("status", normalize_status),
        ("request_type", normalize_request_type),
        ("product_area", normalize_area),
    ]

    print(f"expected={args.expected}")
    print(f"pred={args.pred}")
    print(f"expected_rows={len(expected)} pred_rows={len(predictions)}")
    if len(expected) != len(predictions):
        print("WARNING: row-count mismatch; missing/extra prediction rows are counted as incorrect.")

    denominator = len(expected)
    for metric, normalizer in metrics:
        correct = 0
        mismatches: list[str] = []
        expected_col = find_column(expected, metric)
        pred_col = find_column(predictions, metric)

        for index in range(denominator):
            expected_value = normalizer(expected.iloc[index][expected_col])
            pred_value = ""
            if index < len(predictions):
                pred_value = normalizer(predictions.iloc[index][pred_col])
            if expected_value == pred_value:
                correct += 1
            else:
                mismatches.append(
                    f"row {index + 1}: expected={expected_value!r} predicted={pred_value!r}"
                )

        pct = (correct / denominator * 100.0) if denominator else 0.0
        print(f"{metric}: {correct}/{denominator} = {pct:.1f}%")
        if mismatches:
            print(f"  first mismatches: {'; '.join(mismatches[:5])}")


def find_column(frame: pd.DataFrame, canonical: str) -> str:
    wanted = normalize_column(canonical)
    for column in frame.columns:
        if normalize_column(column) == wanted:
            return column
    raise KeyError(f"Could not find column {canonical!r} in {list(frame.columns)!r}")


def normalize_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def normalize_status(value: object) -> str:
    return str(value).strip().casefold()


def normalize_request_type(value: object) -> str:
    return normalize_column(value)


def normalize_area(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().casefold()).strip()


if __name__ == "__main__":
    main()
