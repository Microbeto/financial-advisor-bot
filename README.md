# Financial Advisor Bot

This is how the project can run on a fresh computer with only Docker Desktop installed.

## Prerequisites

- Docker Desktop with Docker Compose v2 enabled
- Ports `3000`, `8000`, and (for LLM profile) `11434` available

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
   - Backend API docs: http://localhost:8000/docs

## First login

- Default built-in admin account:
  - Username: `admin`
  - Password: `1234`
- If you want to set `ADMIN_BOOTSTRAP_EMAIL` and `ADMIN_BOOTSTRAP_PASSWORD` in `.env`, that admin user is also created at startup.

## Environment variables used for Docker startup

From `.env.example`:

- `APP_SECRET`: JWT/app signing secret (set a strong value outside local dev)
- `ADMIN_BOOTSTRAP_EMAIL`: optional startup admin account email
- `ADMIN_BOOTSTRAP_PASSWORD`: optional startup admin account password
- `MARKET_CACHE_TTL_DAYS`: market cache retention
- `NEWS_CACHE_TTL_DAYS`: news cache retention
- `CORS_ORIGINS`: allowed frontend origins

## Verify services are actually healthy

```bash
docker compose ps
```

You should see `fab-mongo`, `fab-backend`, and `fab-frontend` as running/healthy.

If anything fails, check logs:

```bash
docker compose logs backend --tail=200
docker compose logs frontend --tail=200
docker compose logs mongo --tail=200
```

For live logs:

```bash
docker compose logs -f backend frontend mongo
```

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

To verify Ollama from host:

```bash
curl http://localhost:11434/api/tags
```

## Local development (without Docker)

Use this if you want hot reload and direct local debugging.

### Prerequisites

- Python 3.11
- Node.js 20+
- MongoDB running on `mongodb://127.0.0.1:27017`

### Setup

1. Backend dependencies:
    ```bash
    python -m venv venv
    # Windows PowerShell
    .\venv\Scripts\Activate.ps1
    pip install -r backend/requirements.txt
    ```
2. Frontend dependencies:
    ```bash
    cd frontend
    npm ci
    cd ..
    ```

### Start both apps together

From repository root:

```bash
node index.js
```

This starts:

- FastAPI on `http://127.0.0.1:8000`
- Next.js on `http://localhost:3000`

### Optional local env toggles

- `START_ML_ON_BOOT=0` to skip startup ML training job
- `BACKEND_PORT=8001` or `FRONTEND_PORT=3001` if default ports are busy
- `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000` to force frontend API target

## Common issues

- Backend health returns `{ "ok": true, "db": false }`:
   - MongoDB is not reachable. Start MongoDB and restart backend.
- Frontend loads but API calls fail:
   - Ensure `NEXT_PUBLIC_API_BASE_URL` points to backend (port `8000`), not frontend.
- LLM features not active:
   - Start with `--profile llm` and pull a model into Ollama (`llama3.2:1b` in examples).

