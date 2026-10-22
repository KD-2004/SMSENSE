#!/usr/bin/env python3
"""Evaluate SMSense on a labelled CSV without writing messages to analysis history.

CSV columns: source_id,message,label,language,scenario,sender
Labels: HAM/LEGITIMATE/BENIGN or SCAM/SPAM/PHISHING/FRAUD.
The script calls /api/adversarial-test in batches (maximum 20), which does not
save each sample as a normal analysis-history entry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import httpx


POSITIVE_LABELS = {"SCAM", "SPAM", "PHISHING", "FRAUD", "SMISHING"}
NEGATIVE_LABELS = {"HAM", "LEGITIMATE", "BENIGN", "SAFE", "CLEAN", "NORMAL"}


def parse_label(value: str, row_number: int) -> int:
    label = (value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if label in POSITIVE_LABELS:
        return 1
    if label in NEGATIVE_LABELS:
        return 0
    raise ValueError(
        f"Row {row_number}: label must be one of {sorted(POSITIVE_LABELS | NEGATIVE_LABELS)}, got {value!r}"
    )


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"message", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            # Preserve the original SMS exactly; only use strip() to check
            # whether it contains meaningful characters.
            message = row.get("message") or ""
            if not message.strip():
                raise ValueError(f"Row {row_number}: message cannot be empty")
            rows.append(
                {
                    "row_number": row_number,
                    "source_id": (row.get("source_id") or f"row-{row_number:06d}").strip(),
                    "message": message,
                    "label": parse_label(row.get("label") or "", row_number),
                    "language": (row.get("language") or "unspecified").strip() or "unspecified",
                    "scenario": (row.get("scenario") or "unspecified").strip() or "unspecified",
                    "sender": (row.get("sender") or "").strip(),
                }
            )
    if not rows:
        raise ValueError("CSV has a header but no labelled examples")
    return rows


def roc_auc(rows: list[dict[str, Any]], score_key: str) -> float | None:
    positives = sum(1 for row in rows if row["label"] == 1)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return None

    ordered = sorted(rows, key=lambda row: float(row[score_key]))
    positive_rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        score = float(ordered[i][score_key])
        while j < len(ordered) and float(ordered[j][score_key]) == score:
            j += 1
        # Average 1-based rank for tied scores, for tie-aware Mann–Whitney AUC.
        average_rank = ((i + 1) + j) / 2.0
        positive_rank_sum += average_rank * sum(
            1 for row in ordered[i:j] if row["label"] == 1
        )
        i = j

    return round(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives),
        6,
    )


def metrics(rows: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        actual = int(row["label"])
        predicted = 1 if float(row[score_key]) >= threshold else 0
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    balanced_accuracy = (recall + specificity) / 2.0 if rows else 0.0
    return {
        "examples": len(rows),
        "scam_examples": sum(1 for row in rows if row["label"] == 1),
        "ham_examples": sum(1 for row in rows if row["label"] == 0),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "roc_auc": roc_auc(rows, score_key),
    }


def evaluate_groups(
    rows: list[dict[str, Any]], group_key: str, score_key: str, threshold: float
) -> dict[str, Any]:
    values = sorted({str(row.get(group_key) or "unspecified") for row in rows})
    return {
        value: metrics(
            [row for row in rows if str(row.get(group_key) or "unspecified") == value],
            score_key,
            threshold,
        )
        for value in values
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/evaluation.csv"))
    parser.add_argument("--base-url", default="http://127.0.0.1:4444")
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Fused risk-score threshold on the 0–100 scale (default: 50). Tune on validation data only.",
    )
    parser.add_argument("--batch-size", type=int, default=20, help="1–20 messages per API call (default: 20).")
    parser.add_argument("--report", type=Path, default=Path("data/evaluation_report.json"))
    parser.add_argument("--predictions", type=Path, default=Path("data/evaluation_predictions.csv"))
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 100.0:
        parser.error("--threshold must be between 0 and 100")
    if not 1 <= args.batch_size <= 20:
        parser.error("--batch-size must be between 1 and 20")

    rows = load_rows(args.csv)
    base_url = args.base_url.rstrip("/")
    predictions: list[dict[str, Any]] = []

    with httpx.Client(timeout=300.0) as client:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            payload = {
                "messages": [
                    {
                        "message": row["message"],
                        **({"sender": row["sender"]} if row["sender"] else {}),
                    }
                    for row in batch
                ]
            }
            response = client.post(f"{base_url}/api/adversarial-test", json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text[:500]
                raise RuntimeError(
                    f"API returned HTTP {response.status_code} for rows "
                    f"{start + 2}–{start + len(batch) + 1}: {detail}"
                ) from exc

            body = response.json()
            results = body.get("results")
            if not isinstance(results, list) or len(results) != len(batch):
                raise RuntimeError("API response did not contain the expected number of results")

            for source, result in zip(batch, results):
                try:
                    risk_score = float(result["risk_score"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"API response for {source['source_id']} has no numeric risk_score"
                    ) from exc
                if not math.isfinite(risk_score) or not 0.0 <= risk_score <= 100.0:
                    raise RuntimeError(
                        f"API returned an invalid risk_score for {source['source_id']}"
                    )

                model_scores: dict[str, float] = {}
                signals = (result.get("explanation") or {}).get("model_signals", [])
                if isinstance(signals, list):
                    for signal in signals:
                        if not isinstance(signal, dict) or not signal.get("name"):
                            continue
                        try:
                            scaled = float(signal["score"]) / 15.0 * 100.0
                        except (KeyError, TypeError, ValueError):
                            continue
                        if math.isfinite(scaled):
                            model_scores[str(signal["name"])] = max(0.0, min(100.0, scaled))

                predictions.append(
                    {
                        "source_id": source["source_id"],
                        "label": source["label"],
                        "language": source["language"],
                        "scenario": source["scenario"],
                        "fused_score": risk_score,
                        "fused_label": "SCAM" if risk_score >= args.threshold else "HAM",
                        "model_scores": model_scores,
                    }
                )

    metrics_by_model: dict[str, Any] = {
        "fused_risk_score": metrics(predictions, "fused_score", args.threshold)
    }
    model_names = sorted(
        {name for row in predictions for name in row["model_scores"]}
    )
    model_coverage: dict[str, int] = {}
    for name in model_names:
        available = [
            {**row, "model_percent": row["model_scores"][name]}
            for row in predictions
            if name in row["model_scores"]
        ]
        model_coverage[name] = len(available)
        metrics_by_model[name] = {
            "coverage": len(available),
            "metrics": metrics(available, "model_percent", args.threshold) if available else None,
        }

    breakdowns: dict[str, Any] = {
        "language": evaluate_groups(predictions, "language", "fused_score", args.threshold),
        "scenario": evaluate_groups(predictions, "scenario", "fused_score", args.threshold),
    }
    report = {
        "note": (
            "Measured only against the labels in the supplied CSV and the running local API. "
            "These metrics do not guarantee performance on future or unrepresented messages."
        ),
        "dataset": str(args.csv),
        "api_base_url": base_url,
        "threshold_percent": args.threshold,
        "model_signal_coverage": model_coverage,
        "metrics": metrics_by_model,
        "breakdowns": breakdowns,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "expected_label",
                "fused_label",
                "fused_score",
                "language",
                "scenario",
                "model_scores_json",
            ],
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "source_id": row["source_id"],
                    "expected_label": "SCAM" if row["label"] else "HAM",
                    "fused_label": row["fused_label"],
                    "fused_score": round(row["fused_score"], 4),
                    "language": row["language"],
                    "scenario": row["scenario"],
                    "model_scores_json": json.dumps(row["model_scores"], sort_keys=True),
                }
            )

    print(json.dumps(report["metrics"], indent=2))
    print(f"\nFull report: {args.report}")
    print(f"Predictions (message text omitted): {args.predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
