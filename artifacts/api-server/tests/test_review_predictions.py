from __future__ import annotations

import csv
import json

from tools.review_predictions import analyze_csv, write_report


def _write_csv(path, rows, fields):
    all_fields = list(dict.fromkeys(list(fields) + [key for row in rows for key in row.keys()]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def test_review_predictions_reports_gaps_without_claiming_accuracy(tmp_path):
    rows = [
        {
            "input_row": "10", "status": "OK", "http_status": "200",
            "message": "Please verify your account at https://example.invalid",
            "predicted_label": "LOW", "risk_score": "20",
            "response_json": json.dumps({"results": [{
                "risk_score": 20, "label": "LOW", "category": "URL_REVIEW", "language": "en",
                "sender": None,
                "model_assessment": {"risk": 65, "verdict": "suspicious", "model_verdict": "suspicious", "verdict_supported": True, "verdict_consistent": True},
                "url_analysis": {"count": 1, "urls": [{"reputation": "not_checked"}]},
                "similarity": {"available": False},
            }]}),
        },
        {
            "input_row": "11", "status": "HTTP_ERROR", "http_status": "503",
            "error": "", "message": "test", "predicted_label": "",
            "risk_score": "", "response_json": json.dumps({"detail": {"code": "MODEL_UNAVAILABLE", "message": "unsupported model verdict"}}),
        },
    ]
    source = tmp_path / "sample.csv"
    _write_csv(source, rows, list(rows[0].keys()))
    result = analyze_csv(source)
    summary = result["summary"]
    assert summary["total_rows"] == 2
    assert summary["successful_predictions"] == 1
    assert summary["failed_predictions"] == 1
    assert summary["metrics"] is None
    assert summary["model_risk_at_least_50_final_below_threshold"] == 1
    assert summary["messages_without_sender"] == 1
    assert summary["similarity_unavailable_count"] == 1
    assert summary["failed_cases"][0]["error"] == "unsupported model verdict"
    assert "url_reputation_not_checked" in result["rows"][0]["review_reasons"]
    out = tmp_path / "report"
    write_report(result, out, 50.0, False)
    assert (out / "review_report.md").exists()
    assert (out / "review_summary.json").exists()
    assert (out / "review_rows.csv").exists()


def test_review_predictions_metrics_only_use_verified_labels(tmp_path):
    source = tmp_path / "labelled.csv"
    rows = [
        {"input_row": "1", "status": "OK", "http_status": "200", "risk_score": "80", "predicted_label": "CRITICAL", "ground_truth": "SCAM"},
        {"input_row": "2", "status": "OK", "http_status": "200", "risk_score": "10", "predicted_label": "LOW", "ground_truth": "HAM"},
    ]
    _write_csv(source, rows, list(rows[0].keys()))
    summary = analyze_csv(source)["summary"]
    assert summary["ground_truth_rows_mapped"] == 2
    assert summary["metrics"]["accuracy"] == 1.0
    assert summary["metrics"]["tp"] == 1
    assert summary["metrics"]["tn"] == 1
