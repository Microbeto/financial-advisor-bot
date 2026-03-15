const CACHE_PREFIX = "chart-cache:v1:";

type CacheEnvelope<T> = {
  savedAt: number;
  data: T;
};

function readEnvelope<T>(key: string): CacheEnvelope<T> | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(CACHE_PREFIX + key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CacheEnvelope<T>;
    if (!parsed || typeof parsed.savedAt !== "number") return null;
    return parsed;
  } catch {
    return null;
  }
}

export function getCachedChartData<T>(key: string, maxAgeMs = 1000 * 60 * 30): T | null {
  const env = readEnvelope<T>(key);
  if (!env) return null;
  const age = Date.now() - env.savedAt;
  if (age > maxAgeMs) return null;
  return env.data;
}

export function setCachedChartData<T>(key: string, data: T): void {
  if (typeof window === "undefined") return;
  try {
    const payload: CacheEnvelope<T> = { savedAt: Date.now(), data };
    window.localStorage.setItem(CACHE_PREFIX + key, JSON.stringify(payload));
  } catch {
    // Ignore storage failures (quota/private mode)
  }
}
