"use client";

import { useEffect, useState } from "react";
import type { AuditLogEntry } from "@/lib/types";
import { getAuditLog } from "@/lib/api-client";
import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/lib/auth-context";

function formatTime(ts: string) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

export default function AuditLogPage() {
  const { role } = useAuth();
  const [items, setItems] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const canExport = role === "premium" || role === "admin" || role === "manager";

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const data = await getAuditLog();
        if (!cancelled) {
          setItems(data);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError("Failed to load audit log.");
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
      (entry.regime || "").toLowerCase().includes(q)
    );
  });

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
          Each recommendation call is recorded here with a timestamp and a
          short description.
        </p>
      </div>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by id, description, or regime"
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
                  <th className="px-3 py-2">Regime</th>
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
                          regime === "aggressive"
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
      </div>
    </RequireAuth>
  );
}
