#!/usr/bin/env python3
"""Prepare a labelled-evaluation CSV from a message CSV without inventing labels.

By default, every output label is blank and must be independently reviewed.
Use --verified-label-column only when the source column contains human-verified
ground truth; never use a model's own predictions as ground truth.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from tools.evaluate_dataset import parse_label


OUTPUT_FIELDS = ["source_id", "message", "label", "language", "scenario", "sender"]


def prepare_csv(
    input_path: Path,
    output_path: Path,
    verified_label_column: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create the evaluator's CSV schema, leaving labels blank unless explicitly supplied."""
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different; the source file must not be overwritten.")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path.resolve()}. "
            "Choose another --output path or pass --force to replace it."
        )

    prepared: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "message" not in fieldnames:
            raise ValueError("Input CSV must have a column named exactly 'message'.")
        if verified_label_column and verified_label_column not in fieldnames:
            raise ValueError(
                f"Verified label column {verified_label_column!r} was not found. "
                f"Available columns: {fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            message = row.get("message") or ""
            if not message.strip():
                raise ValueError(f"Input row {row_number}: message cannot be blank.")

            source_id = (row.get("source_id") or "").strip() or f"row-{row_number - 1:06d}"
            if source_id in seen_ids:
                raise ValueError(f"Input row {row_number}: duplicate source_id {source_id!r}.")
            seen_ids.add(source_id)

            label = ""
            if verified_label_column:
                label = (row.get(verified_label_column) or "").strip()
                if not label:
                    raise ValueError(
                        f"Input row {row_number}: verified label column "
                        f"{verified_label_column!r} is blank."
                    )
                # Validate the label before writing any output. Keep the user's label text.
                parse_label(label, row_number)

            prepared.append(
                {
                    "source_id": source_id,
                    "message": message,
                    "label": label,
                    "language": (row.get("language") or "").strip(),
                    "scenario": (row.get("scenario") or "").strip(),
                    "sender": (row.get("sender") or "").strip(),
                }
            )

    if not prepared:
        raise ValueError("Input CSV has a header but no message rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(prepared)

    labelled_rows = sum(bool(row["label"]) for row in prepared)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(prepared),
        "labelled_rows": labelled_rows,
        "labels_left_blank": len(prepared) - labelled_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source CSV; must contain a message column")
    parser.add_argument("--output", type=Path, default=Path("data/evaluation.csv"), help="Output CSV (default: data/evaluation.csv)")
    parser.add_argument(
        "--verified-label-column",
        help="Optional source column containing independently human-verified labels. Never point this at model predictions.",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing the output file if it already exists")
    args = parser.parse_args()

    try:
        summary = prepare_csv(
            args.input,
            args.output,
            verified_label_column=args.verified_label_column,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Prepared {summary['rows']} rows: {summary['labelled_rows']} labelled, "
        f"{summary['labels_left_blank']} labels left blank."
    )
    print(f"Output: {summary['output']}")
    if summary["labels_left_blank"]:
        print(
            "NEXT: Open the output CSV and fill label with independently verified "
            "BENIGN/HAM or SCAM/SPAM/PHISHING labels. Do not copy SMSense predictions "
            "into the label column. The evaluation command intentionally rejects blank labels."
        )
    else:
        print("Verified label column copied and validated. Review it before evaluating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
