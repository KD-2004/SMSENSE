# SMSense API Backend

FastAPI backend for Indian SMS spam, scam and phishing analysis. It runs locally beside the static frontend.

## Local development (Windows PowerShell)

From the repository root, create and activate a virtual environment, install the backend dependencies, and copy the example configuration:

~~~powershell
cd artifacts/api-server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env

# Install/download the primary SMS-Shield model once (about 2.4 GB quantized).
ollama run hf.co/ganmoor-ai-labs/sms-shield:Q4_K_M
# Once the model loads, enter /bye to return to PowerShell.

python -m uvicorn app.main:app --host 127.0.0.1 --port 4444 --reload
~~~

The backend loads the .env file from this directory regardless of the process working directory. Keep the backend terminal open.

In a second terminal, from the repository root:

~~~powershell
python -m http.server 5500 --bind 127.0.0.1
~~~

Open http://127.0.0.1:5500. The frontend defaults to http://127.0.0.1:4444. To override before the frontend script executes, define window.SMSENSE_API_BASE; do not hard-code a tunnel URL in the HTML. CORS allows only http://127.0.0.1:5500 and http://localhost:5500 by default. It accepts browser-requested headers only for configured origins and restricts API methods to GET and POST. Set CORS_ALLOWED_ORIGINS to a comma-separated list of exact origins for a different local port; avoid a wildcard origin beyond local-only debugging. CORS is not authentication; do not expose this prototype directly to the internet without authentication, rate limiting, and network restrictions.

## Optional URL reputation

URL reputation uses the Google Safe Browsing Lookup API only when `GOOGLE_SAFE_BROWSING_API_KEY` is set in the backend `.env` file. The API key must be obtained and configured by the project owner; none is bundled with SMSense. The key is sent in the `x-goog-api-key` request header rather than a URL query parameter, to reduce accidental exposure in request logs.

~~~dotenv
GOOGLE_SAFE_BROWSING_API_KEY=your_google_safe_browsing_api_key
CORS_ALLOWED_ORIGINS=http://127.0.0.1:5500,http://localhost:5500
~~~

Only URL strings are sent to the provider, not the SMS body or sender. URLs can themselves contain tracking identifiers or sensitive query parameters, so enable this feature only when that disclosure is acceptable. The checker does not follow redirects or visit the destination. A reputation result of `malicious` adds bounded URL evidence; `no_match` means only that the provider returned no threat-list match at the time of lookup, not that the URL is guaranteed safe. Missing keys, provider errors and rate/limit conditions remain `not_checked` and do not fail the primary analysis.

The client checks at most 10 unique URLs per message, limits concurrent requests, caches successful answers for one hour, and uses a five-second per-request timeout. This optional enrichment does not replace safe browsing behavior or independently verified evaluation. The overall URL enrichment step is bounded by `SMSENSE_URL_REPUTATION_TIMEOUT_SECONDS` (4 seconds by default). The primary classifier has a 600-second overall deadline by default (`SMSENSE_CLASSIFIER_TIMEOUT_SECONDS`); raise it in `.env` for slow hardware or long pasted messages. Optional IndicBERT is bounded by `SMSENSE_CHALLENGER_TIMEOUT_SECONDS` (2 seconds) and remains diagnostic-only.

## API endpoints

- `GET /api/healthz`: quick reachability check.
- `GET /api/health`: component readiness.
- `POST /api/analyze`: analyse one message and save it to local SQLite history; returns the primary result without waiting for Qwen explanation.
- `POST /api/explain`: generate an optional post-hoc narrative separately; returns fallback safety guidance if Qwen is unavailable.
- `POST /api/adversarial-test`: bounded multi-message analysis without normal history writes.

OpenAPI is available at `/docs`.

## Models and evidence

The primary text detector is **SMS-Shield** (`hf.co/ganmoor-ai-labs/sms-shield:Q4_K_M`), an Indian SMS scam/inbox model fine-tuned from Qwen3-4B and available as a quantized Ollama model. Its structured verdict, risk band, scam type, and red flags are returned in `model_assessment`. For messages longer than 512 characters, the backend evaluates overlapping segments and uses the highest risk segment; this is intentionally recall-oriented and should be measured for false positives.

Qwen3 remains an optional post-hoc explainer and embedding provider. Both the primary SMS-Shield request and explainer explicitly disable Ollama thinking mode for structured results. The primary classifier retries once without `format=json` if constrained output is empty or invalid. The API returns the fused score first; the UI requests explanation from `/api/explain` separately, where failures return fallback guidance. Similarity and URL reputation enrichment have short configurable time budgets and cannot indefinitely delay the primary result. FAISS similarity is supporting evidence, not a phishing probability. The bundled index has only a few demonstration examples, so `SIMILARITY_SCORING_ENABLED=false` by default keeps its matches diagnostic-only. Enable scoring only after an independently labelled, campaign-grouped evaluation demonstrates incremental value. The optional IndicBERT challenger is disabled by default and never affects production scoring; install `requirements-optional-ml.txt` before enabling it. See [MODEL_EVALUATION.md](../../MODEL_EVALUATION.md) for model-card caveats and the labelled benchmark workflow.

## Optional multilingual challenger

The default installation does not install PyTorch/Transformers. To run the optional IndicBERT comparison, activate the virtual environment, run `python -m pip install -r requirements-optional-ml.txt`, set `INDICBERT_ENABLED=true` in `.env`, and restart the backend. IndicBERT adds substantial memory use and reports diagnostic scores only; it does not change the fused risk score.

## FAISS index

The bundled seed set is demonstration data, not a validated reference corpus. Rebuild after Ollama and the configured embedding model are available:

```powershell
python tools/build_faiss_index.py
```

Do not treat the small demo index as a validated detector. Before using similarity in production scoring, benchmark it on a larger labelled corpus.

## Labelled evaluation

The evaluation script requires a CSV with both a message column and a label column. The label must be independently verified ground truth. It is not inferred from SMSense predictions.

### 1. Prepare a starter CSV

From this backend directory, run the helper against your input messages CSV:

~~~powershell
python tools/prepare_evaluation_csv.py --input "C:\path\to\your_messages.csv" --output data\evaluation.csv
~~~

The helper preserves message text and copies source_id, sender, language, and scenario when those columns exist. It creates the label column blank by default, assigns source IDs when needed, and does not copy prediction labels. Open data\evaluation.csv in a spreadsheet editor and fill each label only after independent review.

Supported benign labels: HAM, LEGITIMATE, BENIGN, SAFE, CLEAN, or NORMAL. Supported scam labels: SCAM, SPAM, PHISHING, FRAUD, or SMISHING. The CSV must contain at least one example from each class. If you already have a column of human-verified labels, it can be copied explicitly with --verified-label-column human_review; use that option only for independently verified labels, never a model's predictions. The helper refuses to overwrite an existing output unless --force is supplied.

### 2. Run the evaluation

Start the backend and wait until the API is ready, then run:

~~~powershell
python tools/evaluate_dataset.py --csv data\evaluation.csv --base-url http://127.0.0.1:4444 --threshold 50 --report data\evaluation_report.json --predictions data\evaluation_predictions.csv
~~~

The evaluator rejects a missing CSV, blank/unsupported labels, duplicate source IDs, and a dataset containing only one class, with a clear error message. It uses /api/adversarial-test in batches of at most 20 and omits original message text from the exported prediction CSV.

For a research-quality result, tune the threshold on a validation set and report final metrics on an untouched test set. Review false positives and false negatives, plus language/scenario breakdowns. Do not present the 0–100 fused risk score as a probability. If ground-truth labels are not available yet, continue using tools/run_sample_and_report.py for operational diagnostics, but do not report accuracy.

## Tests

Run unit tests without downloading the large pretrained models:

```powershell
python -m pytest -q
```

Live-model checks are opt-in:

```powershell
$env:RUN_REAL_MODEL_TESTS = "true"
python -m pytest -m integration -q
```

No accuracy claims should be made until predictions are evaluated against independently verified labels.


## Review a completed sample-prediction run

After your sample/adversarial runner has created a prediction CSV, generate a detailed local report from the backend directory:

~~~powershell
python tools/review_predictions.py --csv "C:\path\to\SMSense_API_Predictions.csv" --out-dir data\sample-review
~~~

The tool writes review_report.md (human-readable score distributions, failure cases, and model/fusion disagreements), review_rows.csv (row-level diagnostics, with SMS text omitted by default), and review_summary.json (machine-readable summary). Use the include-message-text option only when you need original SMS text in the local triage CSV and can keep that output private.

Recognized verified-label columns are ground_truth, verified_label, actual_label, true_label, and expected_label. When one is present and contains recognized labels, precision, recall, specificity, F1, accuracy, and balanced accuracy are reported. Without verified labels, the tool explicitly states that it cannot measure accuracy from predictions alone.

The report flags failed API calls, model verdict/risk disagreement, model-risk/fusion-score gaps, missing sender metadata, unchecked URL reputation, and unavailable similarity. These findings are manual-review signals, not proof that an individual prediction is wrong.


## Run a sample dataset and generate a detailed report in one command

Start the API first, then from the backend folder run:

~~~powershell
python tools/run_sample_and_report.py --csv "C:\path\to\your_sample.csv" --out-dir data\sample-run
~~~

The input needs a column named message. Add sender when available. For measured accuracy, include a verified ground-truth column such as ground_truth or label, using HAM/LEGITIMATE/BENIGN or SCAM/SPAM/PHISHING/FRAUD labels. The runner calls the no-history adversarial endpoint one message at a time, records each success/failure and continues even if an item fails, then writes predictions.csv, review_report.md, review_rows.csv, and review_summary.json.

No ground-truth labels means no accuracy claim: the report instead lists score/model disagreements, failures, language/scenario distribution, and missing evidence. Keep the output folder private because predictions.csv contains original SMS text and model responses.
