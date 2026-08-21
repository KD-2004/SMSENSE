# BharatPhish Guard Backend

FastAPI REST backend for phishing and scam analysis of Indian SMS messages. It is designed to sit behind an existing React/Lovable frontend; it does not serve a frontend and never exposes Ollama or model credentials to the browser.

## Run locally

```bash
cd artifacts/api-server
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

The workspace workflow runs the same service on the configured `PORT` (8080 in the artifact preview). OpenAPI docs are available at `/api/docs`.

## Model behavior

The service uses pretrained HinSpam and SilverGuard adapters, Qwen3 via a private Ollama connection for post-hoc explanations, and Qwen3 embeddings plus FAISS for similarity evidence. Model adapters fail closed: if a required model is not available, `/api/analyze` returns HTTP 503 with a clear `MODEL_UNAVAILABLE` state instead of fabricating a security result. The Hugging Face adapters lazily download and cache the configured pretrained weights on first health/analyze use.

## API

- `GET /api/health` reports API component state for Ollama, Hugging Face, FAISS, and SQLite.
- `POST /api/analyze` accepts `{ "message": "...", "sender": "...", "timestamp": "..." }`.
- `POST /api/adversarial-test` evaluates a bounded batch and returns invariants.
- `GET /api/history`, `GET /api/history/{id}`, and `GET /api/stats` expose SQLite-backed history.
- `POST /api/feedback` stores `{ "analysis_id": 1, "helpful": true, "note": "..." }`.

## FAISS seed index

`data/seed_scam_examples.json` is explicitly labeled demo seed data. Build its local FAISS index only after Ollama and `qwen3-embedding:0.6b` are available:

```bash
python tools/build_faiss_index.py
```

If the index is missing or empty, health and analysis expose similarity as unavailable and do not assign similarity points.

## Live integration tests

The default test suite does not require model downloads. Run live validation after configuring both Ollama models and the Hugging Face cache:

```bash
RUN_REAL_MODEL_TESTS=true pytest -m integration -q
```

Rule-based fusion uses transparent caps: Content 30, URL 20, Behavior 20, Sender 15, Obfuscation 10, Similarity 5. KYC, OTP, a URL, or a bank name alone are not treated as proof of phishing. Thresholds (`RISK_LOW_THRESHOLD`, `RISK_SUSPICIOUS_THRESHOLD`, `RISK_HIGH_RISK_THRESHOLD`) and model endpoints are configuration values.