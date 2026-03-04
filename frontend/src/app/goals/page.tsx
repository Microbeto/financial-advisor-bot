"use client";

import { useEffect, useState } from "react";
import type { Goal } from "@/lib/types";
import { getGoals, saveGoals } from "@/lib/api-client";

function makeEmptyGoal(nextId: number): Goal {
  return {
    id: nextId,
    name: "",
    target_amount: 0,
    target_date: null,
    priority: "medium",
    risk_bucket: "balanced",
  };
}

function toDateInputValue(value: string | null): string {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "";
  return d.toISOString().slice(0, 10);
}

export default function GoalsPage() {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const data = await getGoals();
        if (!cancelled) {
          setGoals(data);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError("Failed to load goals from backend.");
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

  function nextId() {
    return goals.length ? Math.max(...goals.map((g) => g.id)) + 1 : 1;
  }

  function updateGoal(
    id: number,
    field: keyof Goal,
    value: string
  ) {
    setGoals((prev) =>
      prev.map((g) => {
        if (g.id !== id) return g;
        if (field === "name") return { ...g, name: value };
        if (field === "target_amount") {
          return { ...g, target_amount: value === "" ? 0 : Number(value) };
        }
        if (field === "priority") {
          return { ...g, priority: value as Goal["priority"] };
        }
        if (field === "risk_bucket") {
          return { ...g, risk_bucket: value as Goal["risk_bucket"] };
        }
        if (field === "target_date") {
          return { ...g, target_date: value === "" ? null : value };
        }
        return g;
      })
    );
  }

  function addGoal() {
    setGoals((prev) => [...prev, makeEmptyGoal(nextId())]);
  }

  function removeGoal(id: number) {
    setGoals((prev) => prev.filter((g) => g.id !== id));
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setMessage(null);

    try {
      const cleaned = goals
        .filter((g) => g.name.trim() !== "")
        .map((g) => ({
          ...g,
          target_amount: Number(g.target_amount) || 0,
        }));
      const saved = await saveGoals(cleaned);
      setGoals(saved);
      setMessage("Goals saved.");
    } catch (err) {
      console.error(err);
      setError("Failed to save goals.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Goals</h1>
        <p className="text-sm text-slate-300 max-w-2xl">
          Define one or more investment goals. Later, the recommendation engine
          will map each goal to a risk bucket and time horizon.
        </p>
      </div>

      <form
        onSubmit={handleSave}
        className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm"
      >
        {loading && (
          <p className="text-xs text-slate-400">
            Loading goals from backend…
          </p>
        )}

        {error && (
          <p className="text-xs text-red-400">
            {error}
          </p>
        )}

        {message && (
          <p className="text-xs text-emerald-400">
            {message}
          </p>
        )}

        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-medium text-slate-100">
            Goal list
          </h2>
          <button
            type="button"
            onClick={addGoal}
            className="inline-flex items-center rounded-md border border-slate-600 bg-slate-800 px-2 py-1 text-[11px] font-medium text-slate-50 hover:bg-slate-700"
          >
            Add goal
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-xs text-slate-200">
            <thead className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
              <tr>
                <th className="px-2 py-1">Name</th>
                <th className="px-2 py-1">Target amount</th>
                <th className="px-2 py-1">Target date</th>
                <th className="px-2 py-1">Priority</th>
                <th className="px-2 py-1">Risk bucket</th>
                <th className="px-2 py-1"></th>
              </tr>
            </thead>
            <tbody>
              {goals.map((g) => (
                <tr key={g.id} className="border-b border-slate-900">
                  <td className="px-2 py-1">
                    <input
                      type="text"
                      value={g.name}
                      onChange={(e) =>
                        updateGoal(g.id, "name", e.target.value)
                      }
                      className="w-32 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                      placeholder="Retirement"
                    />
                  </td>
                  <td className="px-2 py-1">
                    <input
                      type="number"
                      value={g.target_amount}
                      onChange={(e) =>
                        updateGoal(g.id, "target_amount", e.target.value)
                      }
                      className="w-28 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                      min={0}
                      step="1000"
                    />
                  </td>
                  <td className="px-2 py-1">
                    <input
                      type="date"
                      value={toDateInputValue(g.target_date)}
                      onChange={(e) =>
                        updateGoal(g.id, "target_date", e.target.value)
                      }
                      className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                    />
                  </td>
                  <td className="px-2 py-1">
                    <select
                      value={g.priority}
                      onChange={(e) =>
                        updateGoal(g.id, "priority", e.target.value)
                      }
                      className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </td>
                  <td className="px-2 py-1">
                    <select
                      value={g.risk_bucket}
                      onChange={(e) =>
                        updateGoal(g.id, "risk_bucket", e.target.value)
                      }
                      className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                    >
                      <option value="capital_preservation">
                        Capital preservation
                      </option>
                      <option value="balanced">Balanced</option>
                      <option value="growth">Growth</option>
                    </select>
                  </td>
                  <td className="px-2 py-1 text-right">
                    <button
                      type="button"
                      onClick={() => removeGoal(g.id)}
                      className="text-[11px] text-slate-400 hover:text-red-400"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}

              {goals.length === 0 && (
                <tr>
                  <td
                    colSpan={6}
                    className="px-2 py-2 text-center text-[11px] text-slate-500"
                  >
                    No goals defined yet. Add at least one.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="mt-4 inline-flex items-center rounded-md border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-50 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {saving ? "Saving…" : "Save goals"}
        </button>
      </form>
    </div>
  );
}
