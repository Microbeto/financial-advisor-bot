"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  getMlRuntimeSettings,
  updateMlRuntimeSettings,
} from "@/lib/api-client";
import type { MLRuntimeSettings } from "@/lib/types";

type Holding = {
  symbol: string;
  quantity: number;
  avg_price: number;
};

type PortfolioSnapshot = {
  cash: number;
  holdings: Holding[];
};

type Recommendation = {
  symbol: string;
  action: "BUY" | "SELL" | "HOLD";
  size: number;
  confidence: number;
  rationale: string;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function AdminPage() {
  const { isSignedIn, role } = useAuth();
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<Recommendation[]>([]);
  const [basket, setBasket] = useState<Set<string>>(new Set());
  const [mlSettings, setMlSettings] = useState<MLRuntimeSettings | null>(null);
  const [mlDraft, setMlDraft] = useState<MLRuntimeSettings | null>(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlSaving, setMlSaving] = useState(false);
  const [mlMessage, setMlMessage] = useState<string | null>(null);
  const [mlError, setMlError] = useState<string | null>(null);

  // Guard: only admins allowed
  useEffect(() => {
    if (!isSignedIn) {
      router.replace("/");
      return;
    }
    if (role !== "admin") {
      router.replace("/dashboard");
      return;
    }
  }, [isSignedIn, role, router]);

  useEffect(() => {
    if (!isSignedIn || role !== "admin") return;

    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setError(null);

        const pfRes = await fetch(`${API_BASE}/portfolio`, {
          method: "GET",
          credentials: "include",
        });
        if (!pfRes.ok) throw new Error("Failed to load portfolio");
        const portfolio: PortfolioSnapshot = await pfRes.json();

        const recRes = await fetch(`${API_BASE}/recommendations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(portfolio),
        });
        if (!recRes.ok) throw new Error("Failed to load recommendations");
        const recs: Recommendation[] = await recRes.json();

        if (cancelled) return;

        // Candidates: all BUYs plus any HOLDs above some confidence threshold
        const filtered = recs.filter(
          (r) => r.action === "BUY" || (r.action === "HOLD" && r.confidence >= 0.7),
        );

        setCandidates(filtered);
        setBasket(new Set(filtered.filter((r) => r.action === "BUY").map((r) => r.symbol)));
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError("Failed to load admin basket view.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [isSignedIn, role]);

  useEffect(() => {
    if (!isSignedIn || role !== "admin") return;

    let cancelled = false;

    async function loadMlSettings() {
      try {
        setMlLoading(true);
        setMlError(null);
        const data = await getMlRuntimeSettings();
        if (cancelled) return;
        setMlSettings(data);
        setMlDraft(data);
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setMlError("Failed to load ML runtime settings.");
        }
      } finally {
        if (!cancelled) setMlLoading(false);
      }
    }

    loadMlSettings();
    return () => {
      cancelled = true;
    };
  }, [isSignedIn, role]);

  function toggleInBasket(symbol: string) {
    setBasket((prev) => {
      const next = new Set(prev);
      if (next.has(symbol)) {
        next.delete(symbol);
      } else {
        next.add(symbol);
      }
      return next;
    });
  }

  function updateDraft<K extends keyof MLRuntimeSettings>(
    key: K,
    value: MLRuntimeSettings[K],
  ) {
    setMlDraft((prev) => {
      if (!prev) return prev;
      return { ...prev, [key]: value };
    });
  }

  async function saveMlSettings() {
    if (!mlDraft) return;
    try {
      setMlSaving(true);
      setMlMessage(null);
      setMlError(null);
      const saved = await updateMlRuntimeSettings(mlDraft);
      setMlSettings(saved);
      setMlDraft(saved);
      setMlMessage("ML runtime settings saved.");
    } catch (err) {
      console.error(err);
      setMlError("Failed to save ML runtime settings.");
    } finally {
      setMlSaving(false);
    }
  }

  function resetMlSettings() {
    if (!mlSettings) return;
    setMlDraft(mlSettings);
    setMlMessage(null);
    setMlError(null);
  }

  const selectedCount = basket.size;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Admin: Trading basket</h1>
        <p className="max-w-2xl text-sm text-slate-300">
          This view shows the stocks the regime filter has surfaced. Select which
          ones belong in the live trading basket. Later, this selection can be
          pushed to your execution engine.
        </p>
      </div>

      {error && (
        <div className="rounded-md border border-red-500/40 bg-red-950/40 px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-xs">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-[11px] text-slate-300">
            Candidates from current policy engine
          </div>
          <div className="text-[11px] text-slate-400">
            Selected in basket:{" "}
            <span className="font-mono text-sky-300">{selectedCount}</span>
          </div>
        </div>

        {loading ? (
          <p className="text-[11px] text-slate-300">Loading…</p>
        ) : candidates.length === 0 ? (
          <p className="text-[11px] text-slate-300">
            No candidates available. Check portfolio and recommendations.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto text-left text-[11px]">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-900/80 text-slate-300">
                  <th className="px-2 py-1">In basket</th>
                  <th className="px-2 py-1">Symbol</th>
                  <th className="px-2 py-1">Action</th>
                  <th className="px-2 py-1">Confidence</th>
                  <th className="px-2 py-1 w-1/2">Rationale</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((r) => {
                  const checked = basket.has(r.symbol);
                  return (
                    <tr
                      key={r.symbol}
                      className="border-b border-slate-800/80 hover:bg-slate-900"
                    >
                      <td className="px-2 py-1 align-top">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleInBasket(r.symbol)}
                          aria-label={`Toggle ${r.symbol} in basket`}
                          title={`Toggle ${r.symbol} in basket`}
                          className="h-3 w-3 rounded border-slate-500 bg-slate-900 accent-sky-500"
                        />
                      </td>
                      <td className="px-2 py-1 align-top font-mono text-slate-100">
                        {r.symbol}
                      </td>
                      <td className="px-2 py-1 align-top">
                        <span
                          className={[
                            "inline-flex rounded-full px-2 py-[1px] text-[10px] font-medium",
                            r.action === "BUY"
                              ? "bg-emerald-900/60 text-emerald-200 border border-emerald-500/50"
                              : r.action === "SELL"
                              ? "bg-rose-900/60 text-rose-200 border border-rose-500/50"
                              : "bg-slate-800 text-slate-200 border border-slate-600",
                          ].join(" ")}
                        >
                          {r.action}
                        </span>
                      </td>
                      <td className="px-2 py-1 align-top text-slate-200">
                        {(r.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="px-2 py-1 align-top text-slate-300">
                        {r.rationale}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400">
          <p>
            Current selection is kept locally in this prototype. In production,
            send it to an API like <code>/admin/basket</code>.
          </p>
          <button
            type="button"
            className="rounded-md border border-sky-600 bg-sky-600 px-3 py-1 text-[11px] font-medium text-slate-50 hover:bg-sky-500"
          >
            Save basket (wire later)
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-xs">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-[11px] text-slate-300">ML Pipeline Settings</div>
          <div className="text-[11px] text-slate-400">
            Runtime tuning for labeling and walk-forward validation
          </div>
        </div>

        {mlLoading ? (
          <p className="text-[11px] text-slate-300">Loading settings…</p>
        ) : !mlDraft ? (
          <p className="text-[11px] text-slate-300">No settings available.</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1">
              <span className="text-[11px] text-slate-300">Take-profit barrier</span>
              <input
                type="number"
                min={0.001}
                step={0.001}
                value={mlDraft.tp_barrier}
                onChange={(e) => updateDraft("tp_barrier", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>

            <label className="space-y-1">
              <span className="text-[11px] text-slate-300">Stop-loss barrier</span>
              <input
                type="number"
                min={0.001}
                step={0.001}
                value={mlDraft.sl_barrier}
                onChange={(e) => updateDraft("sl_barrier", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>

            <label className="space-y-1">
              <span className="text-[11px] text-slate-300">Time barrier (days)</span>
              <input
                type="number"
                min={1}
                step={1}
                value={mlDraft.time_barrier_days}
                onChange={(e) => updateDraft("time_barrier_days", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>

            <label className="space-y-1">
              <span className="text-[11px] text-slate-300">Walk-forward splits</span>
              <input
                type="number"
                min={2}
                step={1}
                value={mlDraft.walk_forward_splits}
                onChange={(e) => updateDraft("walk_forward_splits", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>

            <label className="space-y-1 md:col-span-2">
              <span className="text-[11px] text-slate-300">Walk-forward minimum training rows</span>
              <input
                type="number"
                min={40}
                step={1}
                value={mlDraft.walk_forward_min_train}
                onChange={(e) => updateDraft("walk_forward_min_train", Number(e.target.value))}
                className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-100"
              />
            </label>
          </div>
        )}

        {mlError && (
          <div className="mt-3 rounded-md border border-red-500/40 bg-red-950/40 px-3 py-2 text-[11px] text-red-200">
            {mlError}
          </div>
        )}
        {mlMessage && (
          <div className="mt-3 rounded-md border border-emerald-500/40 bg-emerald-950/40 px-3 py-2 text-[11px] text-emerald-200">
            {mlMessage}
          </div>
        )}

        <div className="mt-3 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={resetMlSettings}
            disabled={mlLoading || mlSaving || !mlSettings}
            className="rounded-md border border-slate-600 bg-slate-800 px-3 py-1 text-[11px] font-medium text-slate-100 disabled:opacity-50"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={saveMlSettings}
            disabled={mlLoading || mlSaving || !mlDraft}
            className="rounded-md border border-sky-600 bg-sky-600 px-3 py-1 text-[11px] font-medium text-slate-50 hover:bg-sky-500 disabled:opacity-50"
          >
            {mlSaving ? "Saving…" : "Save ML settings"}
          </button>
        </div>
      </div>
    </div>
  );
}
