from tools.evaluate_dataset import metrics, parse_label, roc_auc


def test_label_mapping_accepts_common_ground_truth_terms():
    assert parse_label("HAM", 2) == 0
    assert parse_label("legitimate", 3) == 0
    assert parse_label("SCAM", 4) == 1
    assert parse_label("phishing", 5) == 1


def test_metrics_report_confusion_and_balanced_scores():
    rows = [
        {"label": 1, "score": 90},
        {"label": 1, "score": 70},
        {"label": 0, "score": 20},
        {"label": 0, "score": 80},
    ]
    result = metrics(rows, "score", 50)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["tn"] == 1
    assert result["fn"] == 0
    assert result["recall"] == 1.0
    assert result["precision"] == round(2 / 3, 6)
    assert result["specificity"] == 0.5


def test_roc_auc_handles_perfect_and_tied_rankings():
    perfect = [
        {"label": 0, "score": 10},
        {"label": 1, "score": 90},
    ]
    tied = [
        {"label": 0, "score": 50},
        {"label": 1, "score": 50},
    ]
    assert roc_auc(perfect, "score") == 1.0
    assert roc_auc(tied, "score") == 0.5
    assert roc_auc([{"label": 1, "score": 10}], "score") is None


def test_load_rows_preserves_message_whitespace(tmp_path):
    import csv
    from tools.evaluate_dataset import load_rows

    path = tmp_path / "messages.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "message", "label"])
        writer.writeheader()
        writer.writerow({"source_id": "case-1", "message": "  OTP text  ", "label": "HAM"})

    rows = load_rows(path)
    assert rows[0]["message"] == "  OTP text  "


def test_unknown_ground_truth_label_fails_clearly():
    try:
        parse_label("maybe", 9)
    except ValueError as exc:
        assert "Row 9" in str(exc)
    else:
        raise AssertionError("Unknown labels must not be silently treated as ham")
