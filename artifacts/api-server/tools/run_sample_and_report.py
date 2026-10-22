#!/usr/bin/env python3
"""Run an input SMS CSV against SMSense row-by-row and write a detailed report.

Input CSV requires a message column. Optional columns: source_id, sender,
ground_truth, verified_label, actual_label, true_label, expected_label, label,
language, and scenario. This runner uses /api/adversarial-test so it does not
write each test message to ordinary SMSense history. Results and message text
stay local in the output CSV; treat output files as sensitive.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import httpx
from tools.review_predictions import analyze_csv, write_report

LABEL_COLUMNS = ("ground_truth", "verified_label", "actual_label", "true_label", "expected_label", "label")


def read_samples(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError("Sample CSV not found: " + str(path))
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "message" not in reader.fieldnames:
            raise ValueError("Input CSV must have a column named message")
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            message = row.get("message") or ""
            if not message.strip():
                raise ValueError("Row " + str(row_number) + ": message cannot be blank")
            label = ""
            for name in LABEL_COLUMNS:
                if row.get(name, "").strip():
                    label = row[name].strip()
                    break
            rows.append({
                "input_row": str(row.get("input_row") or row_number - 1),
                "source_id": str(row.get("source_id") or row.get("input_row") or "row-" + str(row_number - 1)),
                "message": message,
                "sender": str(row.get("sender") or ""),
                "ground_truth": label,
                "language": str(row.get("language") or ""),
                "scenario": str(row.get("scenario") or ""),
            })
    if not rows:
        raise ValueError("Input CSV has a header but no sample rows")
    return rows


def _error_text(body: Any, fallback: str) -> str:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("message") or detail.get("code") or detail)[:1000]
        if detail:
            return str(detail)[:1000]
        if body.get("error"):
            return str(body["error"])[:1000]
    return fallback[:1000]


def run_samples(samples: list[dict[str, str]], base_url: str, output_csv: Path, timeout: float = 300.0) -> dict[str, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [
        "input_row", "source_id", "message", "sender", "ground_truth", "language", "scenario",
        "status", "http_status", "predicted_label", "risk_score", "category", "error", "response_json",
    ]
    completed = 0
    failed = 0
    with httpx.Client(timeout=timeout) as client, output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for index, sample in enumerate(samples, start=1):
            payload_message: dict[str, Any] = {"message": sample["message"]}
            if sample["sender"]:
                payload_message["sender"] = sample["sender"]
            response_body: dict[str, Any] = {}
            status_code = ""
            result: dict[str, Any] = {}
            state = "ERROR"
            error = ""
            try:
                response = client.post(
                    base_url.rstrip("/") + "/api/adversarial-test",
                    json={"messages": [payload_message]},
                )
                status_code = str(response.status_code)
                try:
                    response_body = response.json()
                except ValueError:
                    response_body = {"error": response.text[:1000]}
                results = response_body.get("results") if isinstance(response_body, dict) else None
                if response.is_success and isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict):
                    result = results[0]
                    state = "OK"
                    completed += 1
                else:
                    state = "HTTP_ERROR"
                    error = _error_text(response_body, "API returned HTTP " + status_code)
                    failed += 1
            except (httpx.HTTPError, OSError) as exc:
                state = "REQUEST_ERROR"
                error = type(exc).__name__ + ": " + str(exc)
                response_body = {"error": error}
                failed += 1
            writer.writerow({
                **sample,
                "status": state,
                "http_status": status_code,
                "predicted_label": result.get("label", ""),
                "risk_score": result.get("risk_score", ""),
                "category": result.get("category", ""),
                "error": error,
                "response_json": json.dumps(response_body, ensure_ascii=False),
            })
            handle.flush()
            print("[" + str(index) + "/" + str(len(samples)) + "] " + state +
                  ((" HTTP " + status_code) if status_code else "") +
                  (" — " + error[:140] if error else ""))
    return {"total": len(samples), "successful": completed, "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Original test CSV; requires a message column")
    parser.add_argument("--base-url", default="http://127.0.0.1:4444")
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample-run"))
    parser.add_argument("--threshold", type=float, default=50.0, help="Threshold for optional labelled metrics")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--include-message-text-in-review", action="store_true", help="Include original messages in review_rows.csv; output is sensitive")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 100:
        parser.error("threshold must be between 0 and 100")
    if args.timeout <= 0:
        parser.error("timeout must be positive")

    samples = read_samples(args.csv)
    prediction_path = args.out_dir / "predictions.csv"
    run_stats = run_samples(samples, args.base_url, prediction_path, args.timeout)
    analysis = analyze_csv(prediction_path, args.threshold, args.include_message_text_in_review)
    write_report(analysis, args.out_dir, args.threshold, args.include_message_text_in_review)
    print(json.dumps({
        "run": run_stats,
        "review_cases": len(analysis["summary"]["review_cases"]),
        "ground_truth_rows_mapped": analysis["summary"]["ground_truth_rows_mapped"],
        "metrics": analysis["summary"]["metrics"],
        "output_directory": str(args.out_dir),
        "prediction_csv": str(prediction_path),
        "report": str(args.out_dir / "review_report.md"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
