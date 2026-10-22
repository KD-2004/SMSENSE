#!/usr/bin/env python3
"""Create a local-only diagnostic report from SMSense prediction CSV files.

This tool does not call a model or upload data. It reports operational behavior
and review-worthy disagreements; it calculates accuracy only when a recognized
verified ground-truth column is present.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

POSITIVE = {"SCAM", "SPAM", "PHISHING", "FRAUD", "SMISHING", "MALICIOUS", "DANGEROUS"}
NEGATIVE = {"HAM", "LEGITIMATE", "BENIGN", "SAFE", "CLEAN", "NORMAL", "GENUINE"}
GROUND_TRUTH_COLUMNS = ("ground_truth", "verified_label", "actual_label", "true_label", "expected_label")
SUPPORTED_VERDICTS = {"safe", "suspicious", "dangerous"}


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        obj = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool_label(value: Any) -> int | None:
    key = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if key in POSITIVE or key in {"HIGH_RISK", "CRITICAL"}:
        return 1
    if key in NEGATIVE or key == "LOW":
        return 0
    return None


def _extract_result(row: dict[str, str]) -> dict[str, Any]:
    response = _json(row.get("response_json"))
    results = response.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        result = results[0]
    elif isinstance(response.get("result"), dict):
        result = response["result"]
    elif "risk_score" in response or "model_assessment" in response:
        result = response
    else:
        result = {}
    if "risk_score" not in result:
        score = _float(row.get("risk_score"))
        if score is not None:
            result["risk_score"] = score
    if "label" not in result:
        result["label"] = row.get("predicted_label", "")
    if "category" not in result:
        result["category"] = row.get("category", "")
    if "language" not in result:
        result["language"] = row.get("language", "")
    if not result.get("evidence") and row.get("evidence_json"):
        result["evidence"] = _json(row.get("evidence_json"))
    return result


def _metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in pairs)
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in pairs)
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in pairs)
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(pairs) if pairs else 0.0
    return {
        "examples": len(pairs), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
    }


def analyze_csv(csv_path: Path, threshold: float = 50.0, include_message_text: bool = False) -> dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError("Prediction CSV not found: " + str(csv_path))
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        rows = list(reader)
        fields = set(reader.fieldnames)
    label_column = next((name for name in GROUND_TRUTH_COLUMNS if name in fields), None)

    records: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        result = _extract_result(row)
        assessment = result.get("model_assessment") if isinstance(result.get("model_assessment"), dict) else {}
        diagnostics = result.get("decision_diagnostics") if isinstance(result.get("decision_diagnostics"), dict) else {}
        urls_data = result.get("url_analysis") if isinstance(result.get("url_analysis"), dict) else {}
        url_items = urls_data.get("urls") if isinstance(urls_data.get("urls"), list) else []
        similarity = result.get("similarity") if isinstance(result.get("similarity"), dict) else {}
        status = (row.get("status") or "").strip().upper()
        http_status = str(row.get("http_status") or "").strip()
        err = row.get("error") or ""
        body = _json(row.get("response_json"))
        if not result and isinstance(body.get("detail"), dict):
            err = err or str(body["detail"].get("message") or body["detail"])
        success = bool(result) and (not status or status in {"OK", "SUCCESS", "COMPLETED"}) and (not http_status or http_status == "200")

        score = _float(result.get("risk_score"))
        model_risk = _float(assessment.get("risk"))
        if model_risk is None:
            explanation = result.get("explanation") if isinstance(result.get("explanation"), dict) else {}
            signals = explanation.get("model_signals", [])
            for signal in signals if isinstance(signals, list) else []:
                if isinstance(signal, dict) and signal.get("name") == "SMS-Shield":
                    sig_score = _float(signal.get("score"))
                    if sig_score is not None:
                        model_risk = max(0.0, min(100.0, sig_score / 15.0 * 100.0))
                        break

        model_verdict = str(assessment.get("model_verdict") or assessment.get("verdict") or "").strip().lower()
        canonical_verdict = str(assessment.get("verdict") or "").strip().lower()
        verdict_supported = assessment.get("verdict_supported")
        if verdict_supported is None:
            verdict_supported = model_verdict in SUPPORTED_VERDICTS
        verdict_consistent = assessment.get("verdict_consistent")
        if verdict_consistent is None:
            verdict_consistent = model_verdict == canonical_verdict or not model_verdict

        final_label = str(result.get("label") or row.get("predicted_label") or "").strip()
        category = str(result.get("category") or row.get("category") or "unknown")
        message = str(row.get("message") or result.get("raw_message") or "")
        source_row = str(row.get("input_row") or row.get("source_id") or position)
        reasons: list[str] = []
        predicted = (1 if score is not None and score >= threshold else 0) if score is not None else None
        if success and model_risk is not None and model_risk >= 50 and score is not None and score < threshold:
            reasons.append("model_risk_at_least_50_but_final_score_below_threshold")
        if success and model_risk is not None and model_risk >= 70 and score is not None and score < 50:
            reasons.append("high_model_risk_but_final_score_below_50")
        if success and not bool(verdict_supported):
            reasons.append("unsupported_model_verdict")
        elif success and not bool(verdict_consistent):
            reasons.append("model_verdict_risk_band_mismatch")
        if success and url_items and any(isinstance(item, dict) and item.get("reputation") in {"not_checked", None, ""} for item in url_items):
            reasons.append("url_reputation_not_checked")
        if success and not similarity.get("available", False):
            reasons.append("similarity_unavailable")
        if success and not (row.get("sender") or result.get("sender")):
            reasons.append("sender_not_provided")

        truth_raw = row.get(label_column, "") if label_column else ""
        truth = _bool_label(truth_raw) if label_column else None
        records.append({
            "input_row": source_row, "status": "OK" if success else (status or "ERROR"),
            "http_status": http_status, "error": str(err)[:500],
            "final_score": score, "final_label": final_label, "category": category,
            "model_risk": model_risk, "model_verdict": model_verdict,
            "canonical_model_verdict": canonical_verdict, "verdict_supported": bool(verdict_supported),
            "verdict_consistent": bool(verdict_consistent),
            "url_count": int(urls_data.get("count", len(url_items)) or 0),
            "url_reputation_not_checked": any(isinstance(item, dict) and item.get("reputation") in {"not_checked", None, ""} for item in url_items),
            "similarity_available": bool(similarity.get("available", False)),
            "sender_provided": bool(row.get("sender") or result.get("sender")),
            "language": str(result.get("language") or row.get("language") or "unspecified"),
            "review_reasons": reasons, "ground_truth": truth_raw, "ground_truth_binary": truth,
            "predicted_scam_at_threshold": predicted,
            "message": message if include_message_text else "",
            "diagnostics_from_api": diagnostics,
        })

    success_records = [item for item in records if item["status"] == "OK" and item["final_score"] is not None]
    scores = [float(item["final_score"]) for item in success_records]
    model_scores = [float(item["model_risk"]) for item in success_records if item["model_risk"] is not None]
    labels = Counter(item["final_label"] or "unknown" for item in success_records)
    categories = Counter(item["category"] or "unknown" for item in success_records)
    languages = Counter(item["language"] or "unspecified" for item in success_records)
    verdicts = Counter(item["canonical_model_verdict"] or item["model_verdict"] or "missing" for item in success_records)
    reason_counts = Counter(reason for item in records for reason in item["review_reasons"])
    pairs = [
        (int(item["ground_truth_binary"]), int(item["predicted_scam_at_threshold"]))
        for item in records
        if item["ground_truth_binary"] is not None and item["predicted_scam_at_threshold"] is not None
    ]
    summary: dict[str, Any] = {
        "input_file": str(csv_path), "total_rows": len(records),
        "successful_predictions": len(success_records), "failed_predictions": len(records) - len(success_records),
        "success_rate": round(len(success_records) / len(records), 4) if records else 0.0,
        "final_label_distribution": dict(labels.most_common()),
        "category_distribution": dict(categories.most_common()),
        "language_distribution": dict(languages.most_common()),
        "canonical_model_verdict_distribution": dict(verdicts.most_common()),
        "final_score_stats": {
            "min": min(scores) if scores else None,
            "mean": round(statistics.mean(scores), 2) if scores else None,
            "median": round(statistics.median(scores), 2) if scores else None,
            "p90": round(sorted(scores)[max(0, math.ceil(0.90 * len(scores)) - 1)], 2) if scores else None,
            "max": max(scores) if scores else None,
        },
        "model_risk_stats": {
            "min": min(model_scores) if model_scores else None,
            "mean": round(statistics.mean(model_scores), 2) if model_scores else None,
            "median": round(statistics.median(model_scores), 2) if model_scores else None,
            "max": max(model_scores) if model_scores else None,
        },
        "model_risk_at_least_50_final_below_threshold": sum(1 for item in success_records if item["model_risk"] is not None and item["model_risk"] >= 50 and item["final_score"] is not None and item["final_score"] < threshold),
        "model_risk_at_least_70_final_below_50": sum(1 for item in success_records if item["model_risk"] is not None and item["model_risk"] >= 70 and item["final_score"] is not None and item["final_score"] < 50),
        "messages_with_urls": sum(1 for item in success_records if item["url_count"] > 0),
        "messages_without_sender": sum(1 for item in success_records if not item["sender_provided"]),
        "similarity_unavailable_count": sum(1 for item in success_records if not item["similarity_available"]),
        "review_reason_counts": dict(reason_counts.most_common()),
        "ground_truth_column": label_column,
        "ground_truth_rows_mapped": len(pairs),
        "threshold_used_for_optional_metrics": threshold,
        "metrics": _metrics(pairs) if pairs else None,
        "accuracy_note": (
            "Metrics are calculated only when verified ground-truth labels are present in a recognized column. "
            "Without labels, predictions are diagnostics only and are not accuracy measurements."
        ),
        "review_cases": [
            {key: item[key] for key in ("input_row", "final_score", "final_label", "model_risk", "canonical_model_verdict", "category", "review_reasons")}
            for item in records if item["review_reasons"]
        ],
        "failed_cases": [
            {key: item[key] for key in ("input_row", "http_status", "error")}
            for item in records if item["status"] != "OK"
        ],
    }
    return {"summary": summary, "rows": records}


def write_report(result: dict[str, Any], out_dir: Path, threshold: float, include_message_text: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    rows = result["rows"]
    (out_dir / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = [
        "input_row", "status", "http_status", "error", "final_score", "final_label", "category",
        "model_risk", "model_verdict", "canonical_model_verdict", "verdict_supported",
        "verdict_consistent", "url_count", "url_reputation_not_checked", "similarity_available",
        "sender_provided", "language", "review_reasons", "ground_truth", "predicted_scam_at_threshold",
    ]
    if include_message_text:
        columns.append("message")
    with (out_dir / "review_rows.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            exported = {key: row.get(key, "") for key in columns}
            exported["review_reasons"] = ";".join(row["review_reasons"])
            writer.writerow(exported)

    score_stats = summary["final_score_stats"]
    model_stats = summary["model_risk_stats"]
    url_unchecked = sum(1 for row in rows if row["status"] == "OK" and row["url_reputation_not_checked"])
    lines = [
        "# SMSense Sample Prediction Review", "",
        "Input file: " + summary["input_file"],
        "Rows: " + str(summary["total_rows"]),
        "Successful predictions: " + str(summary["successful_predictions"]) + " (" + format(summary["success_rate"], ".1%") + ")",
        "Failed predictions: " + str(summary["failed_predictions"]),
        "Final score range: " + str(score_stats["min"]) + "–" + str(score_stats["max"]) + "; median " + str(score_stats["median"]) + "; mean " + str(score_stats["mean"]),
        "SMS-Shield risk range: " + str(model_stats["min"]) + "–" + str(model_stats["max"]) + "; median " + str(model_stats["median"]) + "; mean " + str(model_stats["mean"]),
        "Threshold for optional labelled metrics: " + str(threshold), "",
        "## Accuracy", "",
    ]
    if summary["metrics"] is None:
        lines.extend([
            "Not measurable from this file: no recognized verified ground-truth column was supplied.",
            "This report describes prediction behavior and data gaps; it does not establish accuracy or improvement over an earlier model.",
        ])
    else:
        m = summary["metrics"]
        lines.extend([
            "Ground-truth rows scored: " + str(m["examples"]),
            "Accuracy: " + format(m["accuracy"], ".1%") + "; precision: " + format(m["precision"], ".1%") + "; recall: " + format(m["recall"], ".1%") + "; specificity: " + format(m["specificity"], ".1%") + "; F1: " + format(m["f1"], ".3f") + "; balanced accuracy: " + format(m["balanced_accuracy"], ".1%") + ".",
            "Confusion counts: TP=" + str(m["tp"]) + ", FP=" + str(m["fp"]) + ", TN=" + str(m["tn"]) + ", FN=" + str(m["fn"]) + ".",
        ])
    lines.extend([
        "", "## Main review findings", "",
        "Model risk at least 50 but final score below threshold: " + str(summary["model_risk_at_least_50_final_below_threshold"]),
        "Model risk at least 70 but final score below 50: " + str(summary["model_risk_at_least_70_final_below_50"]),
        "Messages with URLs: " + str(summary["messages_with_urls"]) + "; responses with unchecked URL reputation: " + str(url_unchecked),
        "Messages without sender metadata: " + str(summary["messages_without_sender"]),
        "Responses without semantic similarity: " + str(summary["similarity_unavailable_count"]),
        "", "### Final labels", "",
    ])
    for key, value in summary["final_label_distribution"].items():
        lines.append("- " + key + ": " + str(value))
    lines.extend(["", "### Review cases", ""])
    if summary["review_cases"]:
        lines.extend([
            "| Row | Final score/label | Model risk/verdict | Category | Reasons |",
            "|---:|---|---|---|---|",
        ])
        for item in summary["review_cases"]:
            lines.append(
                "| " + str(item["input_row"]) + " | " + str(item["final_score"]) + " / " + str(item["final_label"]) +
                " | " + str(item["model_risk"]) + " / " + str(item["canonical_model_verdict"]) +
                " | " + str(item["category"]) + " | " + ", ".join(item["review_reasons"]) + " |"
            )
    else:
        lines.append("No rows matched the configured review rules.")
    lines.extend(["", "### Failed rows", ""])
    if summary["failed_cases"]:
        for item in summary["failed_cases"]:
            lines.append("- Row " + str(item["input_row"]) + " (HTTP " + str(item["http_status"] or "unknown") + "): " + str(item["error"] or "no error detail"))
    else:
        lines.append("No failed rows were detected.")
    lines.extend([
        "", "## Recommended next steps", "",
        "1. Manually verify high model-risk / low final-score examples before changing fusion weights.",
        "2. Fix unsupported model labels/runtime failures and rerun the same inputs.",
        "3. Supply sender metadata where lawfully available; missing sender is not proof of fraud.",
        "4. Add validated URL reputation and a reviewed reference corpus before those signals influence production scoring.",
        "5. Use verified labels and an untouched held-out set before making accuracy claims.",
        "",
        "The row-level CSV omits SMS text by default. The include-message-text option writes original messages into the local triage CSV; keep that file private.",
        "",
    ])
    (out_dir / "review_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="CSV produced by the SMSense sample/adversarial runner")
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample-review"))
    parser.add_argument("--threshold", type=float, default=50.0, help="Threshold used only for optional labelled metrics")
    parser.add_argument("--include-message-text", action="store_true", help="Include message text in the local triage CSV")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 100:
        parser.error("threshold must be between 0 and 100")
    result = analyze_csv(args.csv, args.threshold, args.include_message_text)
    write_report(result, args.out_dir, args.threshold, args.include_message_text)
    summary = result["summary"]
    print(json.dumps({
        "total_rows": summary["total_rows"],
        "successful_predictions": summary["successful_predictions"],
        "failed_predictions": summary["failed_predictions"],
        "review_cases": len(summary["review_cases"]),
        "ground_truth_rows_mapped": summary["ground_truth_rows_mapped"],
        "metrics": summary["metrics"],
        "output_directory": str(args.out_dir),
    }, indent=2))
    print("Detailed report:", args.out_dir / "review_report.md")
    print("Row-by-row diagnostics:", args.out_dir / "review_rows.csv")
    print("Machine-readable summary:", args.out_dir / "review_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
