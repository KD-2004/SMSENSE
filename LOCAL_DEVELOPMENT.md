# SMSense Local Development

SMSense consists of a static frontend in the repository root and a FastAPI backend in `artifacts/api-server`.

## Install the primary model once

SMSense uses SMS-Shield 4B Q4_K_M through local Ollama. Install Ollama and then run:

~~~powershell
ollama run hf.co/ganmoor-ai-labs/sms-shield:Q4_K_M
~~~

Wait for the model to download and load; enter \`/bye\` to return to PowerShell. The quantized model is about 2.4 GB according to its model card. The backend will return a clear model-unavailable error if this model tag is not installed or Ollama is not reachable.

## Start the backend (Windows PowerShell)

From the repository root:

~~~powershell
cd artifacts/api-server
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 4444 --reload
~~~

If your existing virtual environment is located elsewhere, activate that environment instead. Qwen3 chat/embedding models remain optional providers for the explanation and FAISS similarity features; the main detector is SMS-Shield.

Confirm the server is live at [http://127.0.0.1:4444/api/healthz](http://127.0.0.1:4444/api/healthz). Interactive documentation is at [http://127.0.0.1:4444/docs](http://127.0.0.1:4444/docs).

## Start the frontend

Open a second PowerShell terminal at the repository root:

```powershell
python -m http.server 5500 --bind 127.0.0.1
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500). The frontend uses `http://127.0.0.1:4444` by default; this removes the dependency on a temporary ngrok URL. Change `window.SMSENSE_API_BASE` before the frontend script loads only when a different local API address is needed.

**Important:** use the local URL above when Ollama is running on this computer. The GitHub Pages copy cannot reach a visitor's `127.0.0.1` automatically, so it now displays a configuration warning instead of silently trying localhost. To use a hosted frontend, set `window.SMSENSE_API_BASE` to a deployed HTTPS backend and add the exact frontend origin to that backend's `CORS_ALLOWED_ORIGINS`; never expose an unauthenticated local API publicly.

The analysis result is shown as soon as the primary SMS-Shield inference and bounded core evidence finish. The optional Qwen narrative is fetched separately from `/api/explain`; missing/slow Qwen output must not prevent the score and verdict from appearing.

## Notes

- The local backend allows cross-origin requests for development. Restrict CORS and add authentication before exposing it beyond localhost.
- SMS-Shield must be installed in Ollama. The optional IndicBERT challenger requires `python -m pip install -r artifacts/api-server/requirements-optional-ml.txt` plus `INDICBERT_ENABLED=true`.
- Optional semantic similarity waits up to `SMSENSE_SIMILARITY_TIMEOUT_SECONDS` (3 seconds by default); after that, the core result is returned without similarity evidence.
- URL reputation enrichment waits up to `SMSENSE_URL_REPUTATION_TIMEOUT_SECONDS` (4 seconds by default); URL lookup failure does not discard the model result.
- FAISS similarity uses a three-example demo index by default. It should not be treated as a reliable classifier until a larger labelled corpus is tested.
- Run backend unit tests in the server environment with `python -m pytest -q`.
- No accuracy claims should be made until predictions are compared with independently verified labels.
