// frontend/src/lib/api-client.ts
import type {
  RiskProfile,
  PortfolioSnapshot,
  Goal,
  Recommendation,
  AuditLogEntry,
  UserPublic,
  AuthResponse,
  UserCreate,
  UserLogin,
  DailySignals,
  AnnotatedHistoryResponse,
  DashboardResponse,
  MLModelListResponse,
  MLRuntimeSettings,
  MLRuntimeSettingsUpdate,
} from "./types";
import { API_BASE_URL } from "./config";

const USER_KEY = "advisor_user";
const TOKEN_KEY = "advisor_token";
const PORTFOLIO_KEY = "advisor_portfolio";
const GOALS_KEY = "advisor_goals";
const AUDIT_LOG_KEY = "advisor_audit_log";

type RequestOptions = {
  method?: string;
  headers?: Record<string, string>;
  body?: unknown;
  cache?: RequestCache;
  timeoutMs?: number;
  userId?: string;
  retry?: {
    attempts?: number; // total attempts (including first)
    baseDelayMs?: number;
    maxDelayMs?: number;
    retryOnStatuses?: number[];
  };
};

function stripTrailingSlash(s: string) {
  return s.replace(/\/+$/, "");
}

function joinUrl(base: string, path: string) {
  const b = stripTrailingSlash(base);
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

function getStoredUserId(): string | undefined {
  if (typeof window === "undefined") return undefined;
  try {
    const raw = window.localStorage.getItem(USER_KEY);
    if (!raw) return undefined;
    const parsed = JSON.parse(raw) as { user_id?: string; id?: string };
    return parsed?.user_id || parsed?.id || undefined;
  } catch {
    return undefined;
  }
}

function getStoredToken(): string | undefined {
  if (typeof window === "undefined") return undefined;
  try {
    const token = window.localStorage.getItem(TOKEN_KEY);
    return token || undefined;
  } catch {
    return undefined;
  }
}

function isNotFoundError(err: unknown): boolean {
  return typeof (err as { status?: unknown })?.status === "number" && (err as { status: number }).status === 404;
}

function loadLocalJson<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function saveLocalJson<T>(key: string, value: T): T {
  if (typeof window === "undefined") return value;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore storage failures
  }
  return value;
}

function userScopedKey(baseKey: string, userId?: string): string {
  const uid = userId ?? getStoredUserId() ?? "public";
  return `${baseKey}:${uid}`;
}

function buildHeaders(extra?: Record<string, string>, userId?: string) {
  const h: Record<string, string> = {
    "Content-Type": "application/json",
    ...(extra || {}),
  };

  const token = getStoredToken();
  if (token) h["Authorization"] = `Bearer ${token}`;

  const effectiveUserId = userId ?? getStoredUserId();
  if (effectiveUserId) h["X-User-Id"] = effectiveUserId;

  return h;
}

async function readError(res: Response): Promise<string> {
  const contentType = res.headers.get("content-type") || "";
  try {
    if (contentType.includes("application/json")) {
      const j = (await res.json()) as any;
      if (typeof j?.detail === "string") return j.detail;
      if (typeof j?.message === "string") return j.message;
      return JSON.stringify(j);
    }
  } catch {
    // fall through
  }

  try {
    const text = await res.text();
    return text || `Request failed: ${res.status}`;
  } catch {
    return `Request failed: ${res.status}`;
  }
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function isRetryableStatus(status: number, allowed: number[]) {
  return allowed.includes(status);
}

function computeBackoff(attemptIndex: number, baseDelayMs: number, maxDelayMs: number) {
  // attemptIndex: 0 for first retry delay, 1 for second, ...
  const raw = baseDelayMs * Math.pow(2, attemptIndex);
  const jitter = Math.random() * 0.25 + 0.875; // 0.875..1.125
  return Math.min(maxDelayMs, Math.floor(raw * jitter));
}

async function requestOnce<T>(url: string, opts: RequestOptions): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = opts.timeoutMs ?? 30000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      method: opts.method ?? "GET",
      headers: buildHeaders(opts.headers, opts.userId),
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
      cache: opts.cache ?? "no-store",
      signal: controller.signal,
    });

    if (!res.ok) {
      const msg = await readError(res);
      const err: any = new Error(msg);
      err.status = res.status;
      throw err;
    }

    if (res.status === 204) return undefined as unknown as T;

    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      return (await res.json()) as T;
    }

    return (await res.text()) as unknown as T;
  } catch (err: any) {
    if (err?.name === "AbortError") {
      const e: any = new Error(`Request timed out after ${timeoutMs}ms`);
      e.status = 0;
      throw e;
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const url = joinUrl(API_BASE_URL, path);

  const retryCfg = opts.retry ?? {};
  const attempts = Math.max(1, retryCfg.attempts ?? 1);
  const baseDelayMs = Math.max(50, retryCfg.baseDelayMs ?? 300);
  const maxDelayMs = Math.max(baseDelayMs, retryCfg.maxDelayMs ?? 2500);
  const retryOnStatuses = retryCfg.retryOnStatuses ?? [408, 429, 500, 502, 503, 504];

  let lastErr: any = null;

  for (let i = 0; i < attempts; i++) {
    try {
      return await requestOnce<T>(url, opts);
    } catch (e: any) {
      lastErr = e;

      // Do not retry on client errors except explicit allowed.
      const status = typeof e?.status === "number" ? e.status : 0;

      const canRetry =
        i < attempts - 1 &&
        (status === 0 || isRetryableStatus(status, retryOnStatuses) || /timed out/i.test(String(e?.message || "")));

      if (!canRetry) throw e;

      const delay = computeBackoff(i, baseDelayMs, maxDelayMs);
      await sleep(delay);
    }
  }

  throw lastErr ?? new Error("Request failed");
}

/** Auth (kept for future; backend may not implement register/users yet) */
export async function register(payload: UserCreate) {
  return request<AuthResponse>("/auth/register", {
    method: "POST",
    body: payload,
    retry: { attempts: 1 },
  });
}

export async function login(payload: UserLogin) {
  return request<AuthResponse>("/auth/login", {
    method: "POST",
    body: payload,
    retry: { attempts: 1 },
  });
}

export async function me() {
  return request<UserPublic>("/auth/me", {
    retry: { attempts: 1 },
  });
}

export async function listUsers() {
  return request<{ items: UserPublic[] }>("/auth/users", {
    retry: { attempts: 1 },
  });
}

/** Risk Profile */
export async function getRiskProfile(userId?: string) {
  return request<RiskProfile>("/risk-profile", {
    userId,
    retry: { attempts: 2 },
  });
}

export async function saveRiskProfile(profile: RiskProfile, userId?: string) {
  return request<RiskProfile>("/risk-profile", {
    method: "PUT",
    body: profile,
    userId,
    retry: { attempts: 2 },
  });
}

/** Portfolio (not implemented in backend yet) */
export async function getPortfolio(userId?: string) {
  try {
    return await request<PortfolioSnapshot>("/portfolio", { userId });
  } catch (err) {
    if (!isNotFoundError(err)) throw err;
    return loadLocalJson<PortfolioSnapshot>(userScopedKey(PORTFOLIO_KEY, userId), {
      cash: 0,
      holdings: [],
    });
  }
}

export async function savePortfolio(portfolio: PortfolioSnapshot, userId?: string) {
  try {
    return await request<PortfolioSnapshot>("/portfolio", {
      method: "POST",
      body: portfolio,
      userId,
    });
  } catch (err) {
    if (!isNotFoundError(err)) throw err;
    return saveLocalJson(userScopedKey(PORTFOLIO_KEY, userId), portfolio);
  }
}

/** Goals (not implemented in backend yet) */
export async function getGoals(userId?: string) {
  try {
    return await request<Goal[]>("/goals", { userId });
  } catch (err) {
    if (!isNotFoundError(err)) throw err;
    return loadLocalJson<Goal[]>(userScopedKey(GOALS_KEY, userId), []);
  }
}

export async function saveGoals(goals: Goal[], userId?: string) {
  try {
    return await request<Goal[]>("/goals", {
      method: "POST",
      body: goals,
      userId,
    });
  } catch (err) {
    if (!isNotFoundError(err)) throw err;
    return saveLocalJson(userScopedKey(GOALS_KEY, userId), goals);
  }
}

/** Audit Log (not implemented in backend yet) */
export async function getAuditLog(userId?: string) {
  try {
    return await request<AuditLogEntry[]>("/audit-log", { userId });
  } catch (err) {
    if (!isNotFoundError(err)) throw err;
    return loadLocalJson<AuditLogEntry[]>(userScopedKey(AUDIT_LOG_KEY, userId), []);
  }
}

export async function clearAuditLog(userId?: string) {
  try {
    return await request<{ status: string }>("/audit-log/clear", {
      method: "POST",
      userId,
    });
  } catch (err) {
    if (!isNotFoundError(err)) throw err;
    saveLocalJson<AuditLogEntry[]>(userScopedKey(AUDIT_LOG_KEY, userId), []);
    return { status: "ok" };
  }
}

/** Recommendations (not implemented in backend yet) */
export async function getRecommendations(portfolio: PortfolioSnapshot, userId?: string) {
  return request<Recommendation[]>("/recommendations", {
    method: "POST",
    body: portfolio,
    userId,
  });
}

/** Daily Signals */
export async function getDailySignals(userId?: string, force = false) {
  const qs = force ? "?force=true" : "";
  return request<DailySignals>(`/signals/today${qs}`, {
    userId,
    // This call can be slow if backend is fetching prices/news.
    timeoutMs: 90000,
    retry: {
      attempts: 3,
      baseDelayMs: 400,
      maxDelayMs: 3000,
      retryOnStatuses: [408, 429, 500, 502, 503, 504],
    },
  });
}

/** Dashboard */
export async function getDashboardPublic() {
  return request<DashboardResponse>("/dashboard/public", {
    timeoutMs: 90000,
    retry: { attempts: 3, baseDelayMs: 400, maxDelayMs: 3000 },
  });
}

export async function getDashboard(userId?: string) {
  return request<DashboardResponse>("/dashboard", {
    userId,
    timeoutMs: 90000,
    retry: { attempts: 3, baseDelayMs: 400, maxDelayMs: 3000 },
  });
}

/** Market annotated history */
export async function getMarketHistory(symbol: string, days = 180) {
  const enc = encodeURIComponent(symbol);
  return request<AnnotatedHistoryResponse>(`/market/annotated/${enc}?days=${days}`, {
    timeoutMs: 60000,
    retry: { attempts: 2, baseDelayMs: 300, maxDelayMs: 2000 },
  });
}

/** Glossary */
export async function getGlossary() {
  return request<{ terms: Record<string, string> }>("/glossary", {
    retry: { attempts: 2, baseDelayMs: 250, maxDelayMs: 1500 },
  });
}

/** ML model registry (premium+) */
export async function getMlModels() {
  const res = await request<MLModelListResponse>("/ml/models", {
    retry: { attempts: 1 },
  });
  return res.items ?? [];
}

/** ML runtime settings (admin) */
export async function getMlRuntimeSettings() {
  return request<MLRuntimeSettings>("/ml/settings", {
    retry: { attempts: 1 },
  });
}

export async function updateMlRuntimeSettings(payload: MLRuntimeSettingsUpdate) {
  return request<MLRuntimeSettings>("/ml/settings", {
    method: "PUT",
    body: payload,
    retry: { attempts: 1 },
  });
}
