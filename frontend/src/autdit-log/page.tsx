"use client";

import { useEffect, useState } from "react";
import type { AuditLogEntry } from "@/lib/types";
import { getAuditLog } from "@/lib/api-client";

function formatTime(ts: string) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

export default function AuditLogPage() {
  const [items, setItems] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Audit Log</h1>
        <p className="text-sm text-slate-300 max-w-2xl">
          Each recommendation call is recorded here with a timestamp and a
          short description.
        </p>
      </div>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm">
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

        <ul className="space-y-3">
          {items.map((entry) => (
            <li
              key={entry.id}
              className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-xs"
            >
              <div className="flex items-center justify-between gap-4">
                <div className="font-medium text-slate-100">
                  #{entry.id} · {entry.num_recommendations} recs
                </div>
                <div className="text-[11px] text-slate-400">
                  {formatTime(entry.timestamp)}
                </div>
              </div>
              <p className="mt-1 text-slate-300">
                {entry.description}
              </p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
