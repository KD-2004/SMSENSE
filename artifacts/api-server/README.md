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

The service uses pretrained HinSpam and SilverGuard adapters, Qwen3 via a private Ollama connection for post-hoc explanations, and Qwen3 embeddings plus FAISS for similarity evidence. Model adapters fail closed: if a required model is not available, `/api/analyze` returns HTTP 503 with a clear `MODEL_UNAVAILABLE` state instead of fabricating a security result. Optional ML dependencies are commented in `requirements.txt` because loading model weights is intentionally an explicit deployment decision.

## API

- `GET /api/health` reports API component state for Ollama, Hugging Face, FAISS, and SQLite.
- `POST /api/analyze` accepts `{ "message": "...", "sender": "...", "timestamp": "..." }`.
- `POST /api/adversarial-test` evaluates a bounded batch and returns invariants.
- `GET /api/history`, `GET /api/history/{id}`, and `GET /api/stats` expose SQLite-backed history.
- `POST /api/feedback` stores `{ "analysis_id": 1, "helpful": true, "note": "..." }`.

Rule-based fusion uses transparent caps: Content 30, URL 20, Behavior 20, Sender 15, Obfuscation 10, Similarity 5. KYC, OTP, a URL, or a bank name alone are not treated as proof of phishing. Thresholds (`RISK_LOW_THRESHOLD`, `RISK_SUSPICIOUS_THRESHOLD`, `RISK_HIGH_RISK_THRESHOLD`) and model endpoints are configuration values.