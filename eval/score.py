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
    print(f"known_product_areas={extract_product_areas(args.expected)}")

    expected_issue_col = find_column(expected, "issue")
    pred_issue_col = find_column(predictions, "issue")
    pred_index_by_issue = {
        normalize_issue(predictions.iloc[index][pred_issue_col]): index
        for index in range(len(predictions))
    }

    denominator = len(expected)
    matched_pairs: list[tuple[int, int | None, str]] = []
    for index in range(denominator):
        issue_key = normalize_issue(expected.iloc[index][expected_issue_col])
        matched_pairs.append((index, pred_index_by_issue.get(issue_key), issue_key))

    unmatched = [index + 1 for index, pred_index, _ in matched_pairs if pred_index is None]
    if unmatched:
        print(f"WARNING: {len(unmatched)} expected rows have no matching prediction by issue text "
              f"(rows: {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}). "
              "These count as incorrect.")

    for metric, normalizer in metrics:
        correct = 0
        mismatches: list[str] = []
        expected_col = find_column(expected, metric)
        pred_col = find_column(predictions, metric)

        for expected_index, pred_index, _ in matched_pairs:
            expected_value = normalizer(expected.iloc[expected_index][expected_col])
            pred_value = ""
            if pred_index is not None:
                pred_value = normalizer(predictions.iloc[pred_index][pred_col])
            if expected_value == pred_value:
                correct += 1
            else:
                mismatches.append(
                    f"row {expected_index + 1}: expected={expected_value!r} predicted={pred_value!r}"
                )

        pct = (correct / denominator * 100.0) if denominator else 0.0
        print(f"{metric}: {correct}/{denominator} = {pct:.1f}%")
        if mismatches:
            print(f"  first mismatches: {'; '.join(mismatches[:5])}")

    print_status_debug_aligned(expected, predictions, matched_pairs)


def normalize_issue(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def print_status_debug_aligned(
    expected: pd.DataFrame,
    predictions: pd.DataFrame,
    matched_pairs: list[tuple[int, int | None, str]],
) -> None:
    expected_status_col = find_column(expected, "status")
    pred_status_col = find_column(predictions, "status")
    justification_col = find_column(predictions, "justification")

    print("status_mismatch_debug:")
    found = False
    for expected_index, pred_index, _ in matched_pairs:
        expected_status = normalize_status(expected.iloc[expected_index][expected_status_col])
        predicted_status = ""
        justification = ""
        if pred_index is not None:
            predicted_status = normalize_status(predictions.iloc[pred_index][pred_status_col])
            justification = str(predictions.iloc[pred_index][justification_col])
        if expected_status != predicted_status:
            found = True
            print(
                f"Row {expected_index + 1} - Expected: {expected_status}, Got: {predicted_status} "
                f"| Justification: {justification}"
            )
    if not found:
        print("  none")


def extract_product_areas(expected_csv: Path = DEFAULT_EXPECTED) -> list[str]:
    frame = pd.read_csv(expected_csv, keep_default_na=False)
    column = find_column(frame, "product_area")
    return sorted({normalize_area(value) for value in frame[column]})


def print_status_debug(expected: pd.DataFrame, predictions: pd.DataFrame) -> None:
    expected_status_col = find_column(expected, "status")
    pred_status_col = find_column(predictions, "status")
    justification_col = find_column(predictions, "justification")

    print("status_mismatch_debug:")
    found = False
    for index in range(len(expected)):
        expected_status = normalize_status(expected.iloc[index][expected_status_col])
        predicted_status = ""
        justification = ""
        if index < len(predictions):
            predicted_status = normalize_status(predictions.iloc[index][pred_status_col])
            justification = str(predictions.iloc[index][justification_col])
        if expected_status != predicted_status:
            found = True
            print(
                f"Row {index + 1} - Expected: {expected_status}, Got: {predicted_status} "
                f"| Justification: {justification}"
            )
    if not found:
        print("  none")


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
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    return normalized or "general_support"


if __name__ == "__main__":
    main()
