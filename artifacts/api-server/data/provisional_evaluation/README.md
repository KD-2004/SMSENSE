# Provisional SMS evaluation sample

## Purpose

This directory contains a 200-message sample with **heuristic, provisional labels**. It is intended only to exercise the SMSense evaluation command end to end and to help inspect the current workflow.

| Suggested label | Count |
|---|---:|
| BENIGN | 162 |
| SCAM | 38 |
| Total | 200 |

## Important limitation

These labels were inferred from message text and are **not independently verified ground truth**. For example, promotional messages with URLs may be legitimate or deceptive depending on the sender and destination; urgency appended to a normal-looking message is not proof by itself. Therefore:

- Do not cite metrics from this file as validated accuracy, precision, recall, F1, or thesis results.
- Do not treat the labels as a production-ready blocklist or as verified claims about the senders.
- For a research evaluation, use independently reviewed labels and retain ambiguous cases for adjudication.
- Use `tools/prepare_evaluation_csv.py` to prepare your actual dataset, then fill the label column using evidence external to SMSense's own predictions.

## Files in this directory

- `smsense_200_evaluation_provisional.csv`: evaluator-compatible `message,label` file, using the provisional suggestions.
- `smsense_200_provisional_labels.csv`: same suggestions with IDs and review metadata; `label_confidence` is `UNVERIFIED` and every row is marked for review.
- `smsense_200_label_review.csv`: review worksheet with a blank `label` field for independently adjudicated ground truth, alongside the provisional suggestion.

## Run a pipeline smoke test

From `artifacts/api-server/`, with SMSense running at your configured API base URL:

```powershell
python tools\evaluate_dataset.py `
  --csv data\provisional_evaluation\smsense_200_evaluation_provisional.csv `
  --base-url http://127.0.0.1:4444 `
  --threshold 50 `
  --report data\evaluation_report_provisional.json `
  --predictions data\evaluation_predictions_provisional.csv
```

Change the port if your backend is listening elsewhere. The report and predictions filenames include `provisional` intentionally so they are not confused with a verified benchmark.
