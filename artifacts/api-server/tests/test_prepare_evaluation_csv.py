import csv

import pytest

from tools.prepare_evaluation_csv import prepare_csv


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_prepare_csv_leaves_ground_truth_blank_and_preserves_message(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "evaluation.csv"
    write_csv(
        source,
        ["message", "sender", "language", "scenario", "predicted_label"],
        [
            {
                "message": "  Your parcel is delayed.  ",
                "sender": "VM-EXAMPLE",
                "language": "English",
                "scenario": "delivery",
                "predicted_label": "SCAM",
            },
            {
                "message": "Meeting at 5 pm",
                "sender": "",
                "language": "English",
                "scenario": "personal",
                "predicted_label": "BENIGN",
            },
        ],
    )

    summary = prepare_csv(source, output)
    rows = read_csv(output)

    assert summary["rows"] == 2
    assert summary["labelled_rows"] == 0
    assert summary["labels_left_blank"] == 2
    assert rows[0]["message"] == "  Your parcel is delayed.  "
    assert rows[0]["sender"] == "VM-EXAMPLE"
    assert rows[0]["label"] == ""
    assert rows[1]["label"] == ""
    # Model predictions must never be copied into the ground-truth label by default.


def test_prepare_csv_copies_only_an_explicit_verified_label_column(tmp_path):
    source = tmp_path / "verified.csv"
    output = tmp_path / "evaluation.csv"
    write_csv(
        source,
        ["message", "human_review", "sender"],
        [
            {"message": "A normal hello", "human_review": "BENIGN", "sender": ""},
            {"message": "Pay a fee to claim a prize", "human_review": "SCAM", "sender": "VM-TEST"},
        ],
    )

    summary = prepare_csv(source, output, verified_label_column="human_review")
    rows = read_csv(output)

    assert summary["labelled_rows"] == 2
    assert [row["label"] for row in rows] == ["BENIGN", "SCAM"]


def test_prepare_csv_rejects_unrecognized_verified_labels_without_writing_output(tmp_path):
    source = tmp_path / "bad.csv"
    output = tmp_path / "evaluation.csv"
    write_csv(source, ["message", "review"], [{"message": "Example", "review": "maybe"}])

    with pytest.raises(ValueError, match="unsupported label"):
        prepare_csv(source, output, verified_label_column="review")
    assert not output.exists()


def test_prepare_csv_refuses_to_overwrite_without_force(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "evaluation.csv"
    write_csv(source, ["message"], [{"message": "hello"}])
    output.write_text("keep this file", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --force"):
        prepare_csv(source, output)
    assert output.read_text(encoding="utf-8") == "keep this file"


def test_prepare_csv_rejects_duplicate_source_ids(tmp_path):
    source = tmp_path / "duplicate.csv"
    output = tmp_path / "evaluation.csv"
    write_csv(
        source,
        ["source_id", "message"],
        [
            {"source_id": "same-id", "message": "one"},
            {"source_id": "same-id", "message": "two"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate source_id"):
        prepare_csv(source, output)
    assert not output.exists()
