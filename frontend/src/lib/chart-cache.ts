// Chart data client-side caching with localStorage using time-based expiry.
const CACHE_PREFIX = "chart-cache:v1:";

// Wrapper structure to track when cached data was saved.
type CacheEnvelope<T> = {
  savedAt: number;
  data: T;
};

// Retrieve cached data envelope from localStorage, returning null if not found or invalid.
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

// Get cached chart data if it exists and is not older than maxAgeMs (default 30 minutes).
export function getCachedChartData<T>(key: string, maxAgeMs = 1000 * 60 * 30): T | null {
  const env = readEnvelope<T>(key);
  if (!env) return null;
  const age = Date.now() - env.savedAt;
  if (age > maxAgeMs) return null;
  return env.data;
}

// Store chart data in localStorage with current timestamp for expiry tracking.
export function setCachedChartData<T>(key: string, data: T): void {
  if (typeof window === "undefined") return;
  try {
    const payload: CacheEnvelope<T> = { savedAt: Date.now(), data };
    window.localStorage.setItem(CACHE_PREFIX + key, JSON.stringify(payload));
  } catch {
    // Ignore storage failures (quota/private mode)
  }
}
