# SMSense Local Development

SMSense consists of a static frontend in the repository root and a FastAPI backend in `artifacts/api-server`.

## Start the backend (Windows PowerShell)

Run the following from the repository root (adjust the venv activation path if your environment is elsewhere):

```powershell
cd artifacts/api-server
..\..\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 4444 --reload
```

If your existing virtual environment is located at `artifacts/api-server/.venv`, activate `.venv\Scripts\Activate.ps1` instead.

Confirm the server is live at [http://127.0.0.1:4444/api/healthz](http://127.0.0.1:4444/api/healthz). Interactive documentation is at [http://127.0.0.1:4444/docs](http://127.0.0.1:4444/docs).

## Start the frontend

Open a second PowerShell terminal at the repository root:

```powershell
python -m http.server 5500 --bind 127.0.0.1
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500). The frontend uses `http://127.0.0.1:4444` by default; this removes the dependency on a temporary ngrok URL. Change `window.SMSENSE_API_BASE` before the frontend script loads only when a different local API address is needed.

## Notes

- The local backend allows cross-origin requests for development. Restrict CORS and add authentication before exposing it beyond localhost.
- Ollama and the required Hugging Face classifiers still need to be running/available.
- FAISS similarity uses a three-example demo index by default. It should not be treated as a reliable classifier until a larger labelled corpus is tested.
- Run backend unit tests in the server environment with `python -m pytest -q`.
- No accuracy claims should be made until predictions are compared with independently verified labels.
