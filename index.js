// index.js
// Starts backend (FastAPI via Uvicorn) and frontend (Next.js).

const { spawn } = require("child_process");
const path = require("path");
const http = require("http");
const fs = require("fs");

const ROOT = __dirname;

const BACKEND_CWD = path.join(ROOT, "backend");
const FRONTEND_CWD = path.join(ROOT, "frontend");

const BACKEND_HOST = process.env.BACKEND_HOST || "127.0.0.1";
const BACKEND_PORT = Number(process.env.BACKEND_PORT || "8000");
const FRONTEND_PORT = Number(process.env.FRONTEND_PORT || "3000");

const HEALTH_PATH = process.env.BACKEND_HEALTH_PATH || "/health";

const STARTUP_TIMEOUT_MS = Number(process.env.STARTUP_TIMEOUT_MS || "30000");
const HEALTH_POLL_MS = Number(process.env.HEALTH_POLL_MS || "400");
const FORCE_KILL_WAIT_MS = Number(process.env.FORCE_KILL_WAIT_MS || "1500");

const START_ML_ON_BOOT = (process.env.START_ML_ON_BOOT ?? "1").trim().toLowerCase() !== "0";
const ML_BOOT_LOOKBACK_DAYS = Number(process.env.ML_BOOT_LOOKBACK_DAYS || "240");
const ML_BOOT_LABEL_HORIZON_DAYS = Number(process.env.ML_BOOT_LABEL_HORIZON_DAYS || "5");
const ML_BOOT_TEST_SIZE = Number(process.env.ML_BOOT_TEST_SIZE || "0.25");
const ML_BOOT_RANDOM_SEED = Number(process.env.ML_BOOT_RANDOM_SEED || "42");
const ML_BOOT_BASKET =
  process.env.ML_BOOT_BASKET ||
  "SPY,AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,UNH,XOM,TSLA,AVGO,AMD,NFLX,GS";

function isWin() {
  return process.platform === "win32";
}

function log(msg) {
  console.log(msg);
}
function warn(msg) {
  console.warn(msg);
}
function error(msg) {
  console.error(msg);
}

function fileExists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

function pickPythonExe() {
  // Prefer project venv python
  if (isWin()) {
    const venvPy = path.join(ROOT, "venv", "Scripts", "python.exe");
    if (fileExists(venvPy)) return { cmd: venvPy, isAbsolute: true };
    return { cmd: "python", isAbsolute: false };
  } else {
    const venvPy = path.join(ROOT, "venv", "bin", "python");
    if (fileExists(venvPy)) return { cmd: venvPy, isAbsolute: true };
    return { cmd: "python3", isAbsolute: false };
  }
}

function runProcess(name, command, args, cwd, extraEnv = {}, shellMode = false) {
  const child = spawn(command, args, {
    cwd,
    env: { ...process.env, ...extraEnv },
    stdio: "inherit",
    shell: shellMode,
    windowsHide: true,
  });

  child.on("error", (e) => {
    error(`[${name}] failed to start: ${e?.message || e}`);
  });

  child.on("exit", (code, signal) => {
    if (signal) log(`[${name}] exited via signal: ${signal}`);
    else log(`[${name}] exited with code: ${code}`);
  });

  return child;
}

function parseBasketCsv(raw) {
  return String(raw || "")
    .split(",")
    .map((x) => x.trim().toUpperCase())
    .filter(Boolean);
}

function triggerMlTraining(py, cwd, extraEnv = {}) {
  const basket = parseBasketCsv(ML_BOOT_BASKET);
  if (!basket.length) {
    warn("[ml] startup training skipped: empty ML_BOOT_BASKET");
    return null;
  }

  const pyCode = [
    "import asyncio",
    "from app import db",
    "from app.models import MLTrainingRequest",
    "from app.services.ml_workflow import train_models_async",
    `basket=${JSON.stringify(basket)}`,
    `lookback_days=${Number.isFinite(ML_BOOT_LOOKBACK_DAYS) ? ML_BOOT_LOOKBACK_DAYS : 240}`,
    `label_horizon_days=${Number.isFinite(ML_BOOT_LABEL_HORIZON_DAYS) ? ML_BOOT_LABEL_HORIZON_DAYS : 5}`,
    `test_size=${Number.isFinite(ML_BOOT_TEST_SIZE) ? ML_BOOT_TEST_SIZE : 0.25}`,
    `random_seed=${Number.isFinite(ML_BOOT_RANDOM_SEED) ? ML_BOOT_RANDOM_SEED : 42}`,
    "db.init_db()",
    "req=MLTrainingRequest(stock_basket=basket,lookback_days=int(lookback_days),label_horizon_days=int(label_horizon_days),test_size=float(test_size),random_seed=int(random_seed),run_async=True)",
    "res=asyncio.run(train_models_async(req))",
    "print(f'[ml] startup training completed: run_id={res.run_id} selected={res.selected_model_id} trained={len(res.trained_models)}')",
    "db.close_db()",
  ].join("; ");

  const shellMode = py.isAbsolute ? false : isWin();
  log("[ml] starting startup training job");
  const trainer = runProcess("ml-startup-train", py.cmd, ["-c", pyCode], cwd, extraEnv, shellMode);
  trainer.on("exit", (code) => {
    if (code === 0) log("[ml] startup training job finished");
    else warn(`[ml] startup training job exited with code ${code}`);
  });
  return trainer;
}

function killProcessTree(child) {
  if (!child || child.killed || child.exitCode !== null) return;

  if (isWin()) {
    // taskkill is a shell command; keep shell=true here.
    spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
      stdio: "ignore",
      shell: true,
      windowsHide: true,
    });
  } else {
    try {
      child.kill("SIGTERM");
    } catch {}
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function httpGet(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      const chunks = [];
      res.on("data", (d) => chunks.push(d));
      res.on("end", () => {
        resolve({
          status: res.statusCode || 0,
          body: Buffer.concat(chunks).toString("utf-8"),
        });
      });
    });
    req.on("error", reject);
    req.setTimeout(5000, () => req.destroy(new Error("timeout")));
  });
}

async function isBackendHealthy() {
  const url = `http://${BACKEND_HOST}:${BACKEND_PORT}${HEALTH_PATH}`;
  try {
    const r = await httpGet(url);
    return r.status >= 200 && r.status < 300;
  } catch {
    return false;
  }
}

async function getBackendHealth() {
  const url = `http://${BACKEND_HOST}:${BACKEND_PORT}${HEALTH_PATH}`;
  try {
    const r = await httpGet(url);
    if (!(r.status >= 200 && r.status < 300)) {
      return { ok: false, db: false };
    }
    const payload = JSON.parse(String(r.body || "{}"));
    return {
      ok: Boolean(payload?.ok),
      db: Boolean(payload?.db),
    };
  } catch {
    return { ok: false, db: false };
  }
}

async function waitForBackendHealthy(timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await isBackendHealthy()) return true;
    await sleep(HEALTH_POLL_MS);
  }
  return false;
}

async function checkPortInUse(port) {
  // If /health responds, backend is up.
  // If it doesn't, port may still be in use by some other process.
  if (await isBackendHealthy()) return { inUse: true, healthy: true };

  // Try a bare TCP connect via HTTP request to root; if it connects, something is listening.
  const url = `http://${BACKEND_HOST}:${port}/`;
  try {
    const r = await httpGet(url);
    // Any status means something answered.
    return { inUse: true, healthy: false, status: r.status };
  } catch {
    return { inUse: false, healthy: false };
  }
}

async function checkFrontendPortInUse(port) {
  const url = `http://127.0.0.1:${port}/`;
  try {
    const r = await httpGet(url);
    return { inUse: true, status: r.status };
  } catch {
    return { inUse: false, status: 0 };
  }
}

let shuttingDown = false;

async function shutdown(children, exitCode = 0) {
  if (shuttingDown) return;
  shuttingDown = true;

  for (const c of children) killProcessTree(c);

  const start = Date.now();
  while (Date.now() - start < FORCE_KILL_WAIT_MS) {
    const alive = children.some((c) => c && c.exitCode === null);
    if (!alive) break;
    await sleep(50);
  }

  process.exit(exitCode);
}

async function main() {
  // Guard: if port is in use but /health is not ours, tell you clearly.
  const portCheck = await checkPortInUse(BACKEND_PORT);
  if (portCheck.inUse && !portCheck.healthy) {
    warn(
      `[backend] port ${BACKEND_PORT} is already in use, but ${HEALTH_PATH} is not healthy. ` +
        `Stop the other process or set BACKEND_PORT to a free port.`
    );
  }

  const backendAlreadyUp = await isBackendHealthy();

  const backendEnv = {
    HOST: BACKEND_HOST,
    PORT: String(BACKEND_PORT),
    RELOAD: process.env.RELOAD ?? "true",
    // Pass through your existing mongo env names (do not introduce new ones)
    MONGODB_URI: process.env.MONGODB_URI,
    MONGODB_DB: process.env.MONGODB_DB,
    CORS_ORIGINS:
      process.env.CORS_ORIGINS ??
      `http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}`,
  };

  const children = [];

  if (!backendAlreadyUp) {
    const py = pickPythonExe();

    // If python path is absolute (venv python.exe), DO NOT use shell.
    const backendShell = py.isAbsolute ? false : isWin();

    const uvicornArgs = [
      "-m",
      "uvicorn",
      "app.main:app",
      "--host",
      BACKEND_HOST,
      "--port",
      String(BACKEND_PORT),
      "--reload",
      "--reload-dir",
      BACKEND_CWD,
    ];

    log(`[backend] starting: ${py.cmd} ${uvicornArgs.join(" ")}`);

    const backend = runProcess("backend", py.cmd, uvicornArgs, BACKEND_CWD, backendEnv, backendShell);
    children.push(backend);

    backend.on("exit", (code) => {
      if (shuttingDown) return;
      shutdown(children, code ?? 1);
    });

    const ok = await waitForBackendHealthy(STARTUP_TIMEOUT_MS);
    if (!ok) {
      error(`[backend] did not become healthy within ${STARTUP_TIMEOUT_MS}ms`);
      await shutdown(children, 1);
      return;
    }

    log("[backend] healthy");
  } else {
    log(`[backend] already running on http://${BACKEND_HOST}:${BACKEND_PORT}`);
  }

  if (START_ML_ON_BOOT) {
    const health = await getBackendHealth();
    if (!health.db) {
      warn("[ml] startup training skipped: backend running in degraded mode (db=false)");
    } else {
      const py = pickPythonExe();
      const trainer = triggerMlTraining(py, BACKEND_CWD, backendEnv);
      if (trainer) children.push(trainer);
    }
  } else {
    log("[ml] startup training disabled (START_ML_ON_BOOT=0)");
  }

  // Frontend env: set NEXT_PUBLIC_API_BASE_URL so the browser hits the right backend.
  const frontendEnv = {
    NEXT_PUBLIC_API_BASE_URL:
      process.env.NEXT_PUBLIC_API_BASE_URL ?? `http://${BACKEND_HOST}:${BACKEND_PORT}`,
    PORT: String(FRONTEND_PORT),
  };

  const frontendPortState = await checkFrontendPortInUse(FRONTEND_PORT);
  if (frontendPortState.inUse) {
    log(`[frontend] already running on http://localhost:${FRONTEND_PORT}`);
  } else {
    log("[frontend] starting: npm run dev (cwd=frontend)");

    // Use shell on Windows so "npm" resolves in PATH.
    const frontend = runProcess("frontend", "npm", ["run", "dev"], FRONTEND_CWD, frontendEnv, isWin());
    children.push(frontend);

    frontend.on("exit", (code) => {
      if (shuttingDown) return;
      shutdown(children, code ?? 1);
    });
  }

  process.on("SIGINT", () => shutdown(children, 0));
  process.on("SIGTERM", () => shutdown(children, 0));
}

main().catch((e) => {
  error(`[main] fatal: ${e?.stack || e}`);
  process.exit(1);
});
