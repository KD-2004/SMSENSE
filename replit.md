# BharatPhish Guard

FastAPI backend that analyzes suspicious SMS messages with transparent, capped evidence fusion and honest model availability states.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the FastAPI API server (port 8080)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Python dependencies are listed in `artifacts/api-server/requirements.txt`.

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Python 3.11+ and FastAPI
- DB: SQLite (standard library)
- Optional inference: Hugging Face pretrained models and Ollama Qwen3
- Optional similarity: FAISS with Qwen3 embeddings

## Where things live

- `artifacts/api-server/app/main.py` — FastAPI routes and health reporting
- `artifacts/api-server/app/service.py` — analysis orchestration
- `artifacts/api-server/app/fusion.py` — transparent rule-based scoring caps
- `artifacts/api-server/app/storage.py` — SQLite history and feedback
- `artifacts/api-server/README.md` — API and local setup documentation

## Architecture decisions

- Model failures are fail-closed: `/api/analyze` returns a clear 503 state rather than fabricating a score.
- Qwen3 is post-hoc only and cannot change the deterministic score or label.
- Raw SMS text is preserved exactly; normalization is a separate representation.

## Product

The service parses SMS messages, extracts sender and URL evidence, evaluates behavior and obfuscation signals, fuses capped evidence into LOW/SUSPICIOUS/HIGH RISK/CRITICAL labels, stores analysis history, and accepts feedback.

## User preferences

No frontend is included; the API is intended for an existing React/Lovable client.

## Gotchas

- Install Python dependencies from `artifacts/api-server/requirements.txt` before starting the workflow.
- Optional model packages and model weights are not loaded automatically; configure them explicitly.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
