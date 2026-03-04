"use client";

import { useState } from "react";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

export default function HomePage() {
  const { isSignedIn, login, register } = useAuth();
  const router = useRouter();

  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const m = (params.get("mode") || "").toLowerCase();
    if (m === "register") {
      setMode("register");
    } else if (m === "login") {
      setMode("login");
    }
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      if (mode === "register") {
        await register({ email, password });
      } else {
        await login({ email, password });
      }
      router.push("/dashboard");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Authentication failed";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  }

  if (isSignedIn) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 px-6 py-5 text-sm">
          <p className="mb-3 text-slate-200">
            You are already signed in. The dashboard shows your top ideas and news.
          </p>
          <button
            type="button"
            onClick={() => router.push("/dashboard")}
            className="rounded-md border border-sky-600 bg-sky-600 px-3 py-1.5 text-xs font-medium text-slate-50 hover:bg-sky-500"
          >
            Go to dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="w-full max-w-md rounded-xl border border-slate-800 bg-slate-900/80 p-6 shadow-lg">
        <h1 className="mb-2 text-lg font-semibold">Welcome to Advisor Bot</h1>
        <p className="mb-4 text-sm text-slate-300">
          Sign in or create an account to see your personalised market radar and
          policy-based recommendations.
        </p>

        <div className="mb-4 flex justify-center">
          <div className="inline-flex rounded-md border border-slate-700 bg-slate-900 p-1 text-xs">
            <button
              type="button"
              onClick={() => setMode("login")}
              className={[
                "rounded px-3 py-1",
                mode === "login" ? "bg-sky-600 text-slate-50" : "text-slate-300 hover:bg-slate-800",
              ].join(" ")}
            >
              Login
            </button>
            <button
              type="button"
              onClick={() => setMode("register")}
              className={[
                "rounded px-3 py-1",
                mode === "register" ? "bg-sky-600 text-slate-50" : "text-slate-300 hover:bg-slate-800",
              ].join(" ")}
            >
              Register
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && <p className="text-xs text-red-400">{error}</p>}

          <div className="space-y-1">
            <label className="text-xs text-slate-300">Email or username</label>
            <input
              type="text"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
              placeholder="admin or you@example.com"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-slate-300">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
              placeholder="••••••••"
            />
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-md border border-sky-600 bg-sky-600 px-3 py-1.5 text-sm font-medium text-slate-50 hover:bg-sky-500"
          >
            {submitting
              ? mode === "register"
                ? "Creating account…"
                : "Signing in…"
              : mode === "register"
                ? "Create account"
                : "Continue"}
          </button>

          <p className="pt-1 text-center text-[11px] text-slate-400">
            Authentication is connected to your backend. Accounts are created with
            the default user role unless promoted by admin.
          </p>
        </form>
      </div>
    </div>
  );
}
