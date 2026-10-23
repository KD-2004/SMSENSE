# SMSense Developer Guide

SMSense is a local-first SMS phishing and scam analysis app. The repository contains a static frontend at the repository root and a FastAPI backend under `artifacts/api-server`.

## Run locally on Windows

Install Ollama and pull the primary model once:

```powershell
ollama run hf.co/ganmoor-ai-labs/sms-shield:Q4_K_M
```

When the model has downloaded and loaded, enter `/bye` to return to PowerShell.

In the first terminal, from the repository root:

```powershell
cd artifacts/api-server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 4444 --reload
```

In a second terminal, from the repository root:

```powershell
python -m http.server 5500 --bind 127.0.0.1
```

Open `http://127.0.0.1:5500`. The frontend uses `http://127.0.0.1:4444` by default. Confirm the API responds at `http://127.0.0.1:4444/api/healthz`; interactive API documentation is at `http://127.0.0.1:4444/docs`.

## Model roles

- **SMS-Shield 4B through Ollama** is the primary classifier. A missing or invalid primary model response is reported as an error; the backend must not invent a verdict.
- **Qwen3 chat** generates an optional explanation after the main score has been returned.
- **Qwen3 embeddings and FAISS** provide optional similarity diagnostics. The bundled demo index is excluded from scoring by default.
- **IndicBERT** is an optional challenger for diagnostics and evaluation. It is not required for ordinary SMS analysis.

See `LOCAL_DEVELOPMENT.md`, `artifacts/api-server/README.md`, and `MODEL_EVALUATION.md` for the complete setup and evaluation workflow.

## Tests and checks

From `artifacts/api-server`, run:

```powershell
python -m pytest -q
```

The normal test suite mocks AI inference and does not download models. To run opt-in live integration tests, first ensure the configured local Ollama models are available, then set `RUN_REAL_MODEL_TESTS=true` before running pytest. Qwen explanation and embedding checks are optional; the live SMS-Shield API test exercises the primary analysis response.

From the repository root, `pnpm run typecheck` checks the workspace's TypeScript packages. It does not replace the Python backend tests.

## Hosted frontend and Replit notes

GitHub Pages hosts the static HTML only; it cannot reach Ollama running on a visitor's computer through `127.0.0.1`. For local analysis, run the frontend and backend locally using the ports above. For a hosted frontend, configure `window.SMSENSE_API_BASE` to a deployed HTTPS backend and allow that exact origin in `CORS_ALLOWED_ORIGINS`.

The Replit API artifact has its own service-port configuration in `artifacts/api-server/.replit-artifact/artifact.toml` (currently port 8080 in its deployment configuration). Do not mix that hosted/service port with the local Windows port 4444; verify the runtime's `PORT` setting and frontend API base for the environment being used.

CORS is not authentication. Do not expose the prototype API publicly without appropriate authentication, rate limiting, and network restrictions.
