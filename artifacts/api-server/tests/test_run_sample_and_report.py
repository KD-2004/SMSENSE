from __future__ import annotations

import csv
import pytest

from tools.run_sample_and_report import read_samples


def test_read_samples_preserves_message_sender_and_ground_truth(tmp_path):
    source = tmp_path / "input.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "message", "sender", "label", "language"])
        writer.writeheader()
        writer.writerow({
            "source_id": "sms-01",
            "message": "  Your account needs review at example.invalid  ",
            "sender": "BANK-XX",
            "label": "PHISHING",
            "language": "en",
        })
    rows = read_samples(source)
    assert rows[0]["message"] == "  Your account needs review at example.invalid  "
    assert rows[0]["sender"] == "BANK-XX"
    assert rows[0]["ground_truth"] == "PHISHING"
    assert rows[0]["source_id"] == "sms-01"


def test_read_samples_rejects_missing_message_column(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("text,label\nhello,HAM\n", encoding="utf-8")
    with pytest.raises(ValueError, match="message"):
        read_samples(source)


def test_read_samples_rejects_blank_messages(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("message,label\n ,HAM\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank"):
        read_samples(source)
