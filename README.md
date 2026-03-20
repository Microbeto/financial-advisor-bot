# Financial Advisor Bot

This is how the project can run on a fresh computer with only Docker Desktop installed.

## Quick start (no local Python/Node/Mongo setup)

1. Optional: copy environment template to customize secrets/config:
   ```bash
   cp .env.example .env
   ```
   On PowerShell:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Start everything:
   ```bash
   docker compose up --build -d
   ```
3. Open:
   - Frontend: http://localhost:3000
   - Backend health: http://localhost:8000/health

## Stop and clean up

- Stop services:
  ```bash
  docker compose down
  ```
- Stop and remove Mongo volume data too:
  ```bash
  docker compose down -v
  ```

## Services included

- `mongo` (MongoDB 7)
- `backend` (FastAPI + Uvicorn)
- `frontend` (Next.js)

## LLM requirements

You do not need local Python/Node/Mongo installs.

LLM features are optional and need an Ollama runtime plus a pulled model. There are two ways:

1. Host-installed Ollama (manual):
    - Install Ollama on your computer
    - Pull a model (for example): `ollama pull llama3.2:1b`

2. Docker-only Ollama (recommended for clean machines):
    - Start with LLM profile:
       ```bash
       docker compose --profile llm up --build -d
       ```
    - Pull model into the Ollama container:
       ```bash
       docker exec -it fab-ollama ollama pull llama3.2:1b
       ```

If Ollama/model is unavailable, the app still runs and falls back to non-LLM paths.

