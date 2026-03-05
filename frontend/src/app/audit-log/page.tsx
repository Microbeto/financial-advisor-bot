"use client";

import { useEffect, useState } from "react";
import type {
  AuditLogEntry,
  MLModelInfo,
  MLRuntimeSettings,
  MLTournamentStatsResponse,
} from "@/lib/types";
import {
  getAuditLog,
  getMlModels,
  getMlRuntimeSettings,
  getMlTournamentStats,
} from "@/lib/api-client";
import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/lib/auth-context";

function formatTime(ts: string) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

function asPct(n: number, digits = 2) {
  if (!Number.isFinite(n)) return "0.00%";
  return `${(n * 100).toFixed(digits)}%`;
}

function asNum(n: number, digits = 3) {
  if (!Number.isFinite(n)) return "0.000";
  return n.toFixed(digits);
}

export default function AuditLogPage() {
  const { role } = useAuth();
  const [items, setItems] = useState<AuditLogEntry[]>([]);
  const [models, setModels] = useState<MLModelInfo[]>([]);
  const [tournament, setTournament] = useState<MLTournamentStatsResponse | null>(null);
  const [runtimeSettings, setRuntimeSettings] = useState<MLRuntimeSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const canExport = role === "premium" || role === "admin" || role === "manager";
  const canViewMlAdmin = canExport;
  const canViewSettings = role === "admin";

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setError(null);

        const [auditRes, modelsRes] = await Promise.allSettled([
          getAuditLog(),
          canViewMlAdmin ? getMlModels() : Promise.resolve([]),
        ]);

        if (cancelled) return;

        if (auditRes.status === "fulfilled") {
          setItems(auditRes.value);
        }

        let modelList: MLModelInfo[] = [];
        if (modelsRes.status === "fulfilled") {
          modelList = (modelsRes.value || []) as MLModelInfo[];
          setModels(modelList);
        }

        const selected =
          modelList.find((m) => Boolean(m.is_selected ?? m.selected)) ||
          modelList.find((m) => Boolean(m.selected)) ||
          null;

        if (canViewMlAdmin) {
          try {
            const t = await getMlTournamentStats(selected?.model_id);
            if (!cancelled) setTournament(t);
          } catch {
            if (!cancelled) setTournament(null);
          }
        }

        if (canViewSettings) {
          try {
            const s = await getMlRuntimeSettings();
            if (!cancelled) setRuntimeSettings(s);
          } catch {
            if (!cancelled) setRuntimeSettings(null);
          }
        }

        if (auditRes.status === "rejected" && modelsRes.status === "rejected") {
          setError("Failed to load backend audit and ML lifecycle details.");
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError("Failed to load backend audit details.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = items.filter((entry) => {
    const q = query.trim().toLowerCase();
    if (!q) return true;
    return (
      String(entry.id).includes(q) ||
      (entry.description || "").toLowerCase().includes(q) ||
      (entry.regime || "").toLowerCase().includes(q) ||
      String(entry.model_id || "").toLowerCase().includes(q)
    );
  });

  const trainingEntries = items.filter((x) => (x.regime || "").toLowerCase() === "ml_training");
  const modelEntries = items.filter((x) => (x.regime || "").toLowerCase() === "ml_model");
  const selectedModels = models.filter((m) => Boolean(m.is_selected ?? m.selected));
  const deployedModels = models.filter((m) => Boolean(m.is_deployed ?? m.deployed));

  function inferredRegime(entry: AuditLogEntry): string {
    if (entry.regime) return entry.regime;
    const text = (entry.description || "").toLowerCase();
    if (text.includes("aggressive")) return "aggressive";
    if (text.includes("conservative")) return "conservative";
    if (text.includes("balanced") || text.includes("moderate")) return "balanced";
    return "n/a";
  }

  function exportCsv() {
    if (!canExport || typeof window === "undefined") return;
    const header = ["id", "timestamp", "regime", "num_recommendations", "description"];
    const rows = filtered.map((entry) => [
      String(entry.id),
      entry.timestamp,
      inferredRegime(entry),
      String(entry.num_recommendations),
      (entry.description || "").replace(/"/g, '""'),
    ]);
    const csv = [header.join(","), ...rows.map((r) => `${r[0]},${r[1]},${r[2]},${r[3]},"${r[4]}"`)].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "audit-log.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <RequireAuth>
      <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Audit Log</h1>
        <p className="text-sm text-slate-300 max-w-2xl">
          End-to-end backend trace for ML training runs, model creation, selection,
          deployment state, and prediction routing context.
        </p>
      </div>

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 text-xs">
          <p className="text-slate-400">Audit entries</p>
          <p className="mt-1 text-lg font-semibold text-slate-100">{items.length}</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 text-xs">
          <p className="text-slate-400">ML training runs</p>
          <p className="mt-1 text-lg font-semibold text-slate-100">{trainingEntries.length}</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 text-xs">
          <p className="text-slate-400">Models in registry</p>
          <p className="mt-1 text-lg font-semibold text-slate-100">{models.length}</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 text-xs">
          <p className="text-slate-400">Selected / Deployed</p>
          <p className="mt-1 text-lg font-semibold text-slate-100">
            {selectedModels.length} / {deployedModels.length}
          </p>
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by id, model_id, description, or regime"
            className="w-full max-w-md rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
          />
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-slate-700 bg-slate-950 px-2 py-0.5 text-[10px] text-slate-300">
              Role: {role}
            </span>
            {canExport && (
              <button
                type="button"
                onClick={exportCsv}
                className="rounded-md border border-slate-600 bg-slate-800 px-3 py-1 text-[11px] text-slate-100 hover:bg-slate-700"
              >
                Export CSV
              </button>
            )}
          </div>
        </div>

        {loading && (
          <p className="text-xs text-slate-400">
            Loading audit entries…
          </p>
        )}

        {error && (
          <p className="text-xs text-red-400">
            {error}
          </p>
        )}

        {!loading && !error && items.length === 0 && (
          <p className="text-xs text-slate-400">
            No audit entries yet. Trigger a recommendation from the dashboard.
          </p>
        )}

        {!loading && !error && filtered.length > 0 && (
          <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950/60">
            <table className="min-w-full text-left text-xs text-slate-200">
              <thead className="border-b border-slate-800 bg-slate-900 text-[11px] uppercase text-slate-400">
                <tr>
                  <th className="px-3 py-2">ID</th>
                  <th className="px-3 py-2">Timestamp</th>
                  <th className="px-3 py-2">Event Type</th>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2">Recommendations</th>
                  <th className="px-3 py-2">Description</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((entry) => {
                  const regime = inferredRegime(entry);
                  return (
                    <tr key={entry.id} className="border-b border-slate-900 hover:bg-slate-900/70">
                      <td className="px-3 py-2 font-mono text-slate-100">#{entry.id}</td>
                      <td className="px-3 py-2 text-slate-300">{formatTime(entry.timestamp)}</td>
                      <td className="px-3 py-2">
                        <span className={[
                          "inline-flex rounded-full border px-2 py-0.5 text-[10px]",
                          regime === "ml_training"
                            ? "border-violet-500/50 bg-violet-900/30 text-violet-200"
                            : regime === "ml_model"
                              ? "border-fuchsia-500/50 bg-fuchsia-900/30 text-fuchsia-200"
                              : regime === "aggressive"
                            ? "border-amber-500/50 bg-amber-900/30 text-amber-200"
                            : regime === "conservative"
                              ? "border-emerald-500/50 bg-emerald-900/30 text-emerald-200"
                              : regime === "balanced"
                                ? "border-sky-500/50 bg-sky-900/30 text-sky-200"
                                : "border-slate-600 bg-slate-800 text-slate-300",
                        ].join(" ")}>
                          {regime}
                        </span>
                      </td>
                      <td className="px-3 py-2 font-mono text-[11px] text-slate-300">{entry.model_id || "-"}</td>
                      <td className="px-3 py-2 text-slate-300">{entry.num_recommendations}</td>
                      <td className="px-3 py-2 text-slate-300">{entry.description}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {canViewMlAdmin && (
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-100">Model Selection Process</h2>
            <p className="text-xs text-slate-400">
              Registry snapshot from backend model ranking, selected routing set, and tournament combiner outcome.
            </p>
          </div>

          {models.length === 0 ? (
            <p className="text-xs text-slate-400">No ML models in registry yet.</p>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950/60">
              <table className="min-w-full text-left text-xs text-slate-200">
                <thead className="border-b border-slate-800 bg-slate-900 text-[11px] uppercase text-slate-400">
                  <tr>
                    <th className="px-3 py-2">Model ID</th>
                    <th className="px-3 py-2">Algorithm</th>
                    <th className="px-3 py-2">Rank</th>
                    <th className="px-3 py-2">Score</th>
                    <th className="px-3 py-2">F1</th>
                    <th className="px-3 py-2">Selected</th>
                    <th className="px-3 py-2">Deployed</th>
                    <th className="px-3 py-2">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.model_id} className="border-b border-slate-900 hover:bg-slate-900/70">
                      <td className="px-3 py-2 font-mono text-[11px] text-slate-100">{m.model_id}</td>
                      <td className="px-3 py-2 text-slate-300">{m.algorithm || "-"}</td>
                      <td className="px-3 py-2 text-slate-300">{m.rank ?? "-"}</td>
                      <td className="px-3 py-2 text-slate-300">{asNum(Number(m.score ?? 0), 3)}</td>
                      <td className="px-3 py-2 text-slate-300">{asNum(Number(m.metrics?.f1 ?? 0), 3)}</td>
                      <td className="px-3 py-2 text-slate-300">{(m.is_selected ?? m.selected) ? "yes" : "no"}</td>
                      <td className="px-3 py-2 text-slate-300">{(m.is_deployed ?? m.deployed) ? "yes" : "no"}</td>
                      <td className="px-3 py-2 text-slate-400">{formatTime(String(m.created_at || ""))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <h3 className="text-xs font-semibold text-slate-100">Tournament Evaluator (Walk-Forward)</h3>
            {!tournament || !tournament.competitor_stats || Object.keys(tournament.competitor_stats).length === 0 ? (
              <p className="mt-2 text-xs text-slate-400">
                No tournament stats found for current selected model. Train/select a `multi_armed_tournament` model to populate this.
              </p>
            ) : (
              <div className="mt-2 space-y-2">
                <p className="text-xs text-slate-300">
                  Winner: <span className="font-semibold text-slate-100">{tournament.winner_name || "n/a"}</span>
                  {" · "}
                  Model: <span className="font-mono text-slate-200">{tournament.model_id || "n/a"}</span>
                </p>
                <div className="overflow-x-auto rounded border border-slate-800 bg-slate-950/60">
                  <table className="min-w-full text-left text-xs text-slate-200">
                    <thead className="border-b border-slate-800 bg-slate-900 text-[11px] uppercase text-slate-400">
                      <tr>
                        <th className="px-3 py-2">Competitor</th>
                        <th className="px-3 py-2">Sharpe</th>
                        <th className="px-3 py-2">Max Drawdown</th>
                        <th className="px-3 py-2">Mean Return</th>
                        <th className="px-3 py-2">Volatility</th>
                        <th className="px-3 py-2">Samples</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(tournament.competitor_stats).map(([name, s]) => (
                        <tr key={name} className="border-b border-slate-900 hover:bg-slate-900/70">
                          <td className="px-3 py-2 text-slate-200">{name}</td>
                          <td className="px-3 py-2 text-slate-300">{asNum(Number(s.sharpe ?? 0), 3)}</td>
                          <td className="px-3 py-2 text-slate-300">{asPct(Number(s.max_drawdown ?? 0), 2)}</td>
                          <td className="px-3 py-2 text-slate-300">{asPct(Number(s.mean_return ?? 0), 2)}</td>
                          <td className="px-3 py-2 text-slate-300">{asPct(Number(s.volatility ?? 0), 2)}</td>
                          <td className="px-3 py-2 text-slate-300">{Math.round(Number(s.sample_count ?? 0))}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {canViewSettings && runtimeSettings && (
            <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
              <h3 className="text-xs font-semibold text-slate-100">ML Runtime Settings (Admin)</h3>
              <div className="mt-2 grid gap-2 text-xs text-slate-300 sm:grid-cols-2 xl:grid-cols-3">
                <div>TP barrier: <span className="text-slate-100">{runtimeSettings.tp_barrier}</span></div>
                <div>SL barrier: <span className="text-slate-100">{runtimeSettings.sl_barrier}</span></div>
                <div>Time barrier days: <span className="text-slate-100">{runtimeSettings.time_barrier_days}</span></div>
                <div>Walk-forward splits: <span className="text-slate-100">{runtimeSettings.walk_forward_splits}</span></div>
                <div>Min train samples: <span className="text-slate-100">{runtimeSettings.walk_forward_min_train}</span></div>
                <div>Model entries in audit: <span className="text-slate-100">{modelEntries.length}</span></div>
              </div>
            </div>
          )}
        </section>
      )}
      </div>
    </RequireAuth>
  );
}
