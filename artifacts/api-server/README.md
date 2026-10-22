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

Open http://127.0.0.1:5500. The frontend defaults to http://127.0.0.1:4444. To override before the frontend script executes, define window.SMSENSE_API_BASE; do not hard-code a tunnel URL in the HTML. CORS is permissive for local prototyping; tighten allowed origins and add authentication before exposing the API to others.

## API endpoints

- `GET /api/healthz`: quick reachability check.
- `GET /api/health`: component readiness.
- `POST /api/analyze`: analyse one message and save it to local SQLite history.
- `POST /api/adversarial-test`: bounded multi-message analysis without normal history writes.

OpenAPI is available at `/docs`.

## Models and evidence

The primary text detector is **SMS-Shield** (`hf.co/ganmoor-ai-labs/sms-shield:Q4_K_M`), an Indian SMS scam/inbox model fine-tuned from Qwen3-4B and available as a quantized Ollama model. Its structured verdict, risk band, scam type, and red flags are returned in `model_assessment`. For messages longer than 512 characters, the backend evaluates overlapping segments and uses the highest risk segment; this is intentionally recall-oriented and should be measured for false positives.

Qwen3 remains an optional post-hoc explainer and embedding provider. FAISS similarity is supporting evidence, not a phishing probability. The optional IndicBERT challenger is disabled by default and never affects production scoring; install `requirements-optional-ml.txt` before enabling it. See [MODEL_EVALUATION.md](../../MODEL_EVALUATION.md) for model-card caveats and the labelled benchmark workflow.

## Optional multilingual challenger

The default installation does not install PyTorch/Transformers. To run the optional IndicBERT comparison, activate the virtual environment, run `python -m pip install -r requirements-optional-ml.txt`, set `INDICBERT_ENABLED=true` in `.env`, and restart the backend. IndicBERT adds substantial memory use and reports diagnostic scores only; it does not change the fused risk score.

## FAISS index

The bundled seed set is demonstration data, not a validated reference corpus. Rebuild after Ollama and the configured embedding model are available:

```powershell
python tools/build_faiss_index.py
```

Do not treat the small demo index as a validated detector. Before using similarity in production scoring, benchmark it on a larger labelled corpus.

## Labelled evaluation

To measure accuracy against independently reviewed ground-truth labels, fill in data/evaluation_template.csv and follow the [accuracy and robustness guide](../../MODEL_EVALUATION.md). With the backend running, from this directory run:

~~~powershell
python tools/evaluate_dataset.py --csv data/evaluation.csv --base-url http://127.0.0.1:4444 --threshold 50
~~~

The runner uses /api/adversarial-test in batches and omits message text from the exported prediction file. Tune any threshold on validation data only; use an untouched test set for the final report. Do not present the 0–100 fused risk score as a probability.

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
