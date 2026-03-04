// frontend/src/lib/config.ts

export function normalizeBaseUrl(input?: string): string {
  const s = (input || "").trim();
  if (!s) return "http://127.0.0.1:8000";
  return s.endsWith("/") ? s.slice(0, -1) : s;
}

/**
 * Single source of truth for backend URL.
 * Must point to FastAPI (port 8000), never Next.js (3000).
 */
export const API_BASE_URL = normalizeBaseUrl(
  process.env.NEXT_PUBLIC_API_BASE_URL ||
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    "http://127.0.0.1:8000"
);
