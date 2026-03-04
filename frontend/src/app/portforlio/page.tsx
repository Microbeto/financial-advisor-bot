"use client";

import { useEffect, useState } from "react";
import type { Holding, PortfolioSnapshot } from "@/lib/types";
import { getPortfolio, savePortfolio } from "@/lib/api-client";
import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/lib/auth-context";

function makeEmptyHolding(): Holding {
  return {
    symbol: "",
    quantity: 0,
    avg_price: 0,
  };
}

export default function PortfolioPage() {
  const { role } = useAuth();
  const [portfolio, setPortfolio] = useState<PortfolioSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const data = await getPortfolio();
        if (!cancelled) {
          setPortfolio(data);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError("Failed to load portfolio from backend.");
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

  function updateHolding(index: number, field: keyof Holding, value: string) {
    if (!portfolio) return;

    const updated = portfolio.holdings.map((h, i) => {
      if (i !== index) return h;

      if (field === "symbol") {
        return { ...h, symbol: value.toUpperCase() };
      }
      if (field === "quantity") {
        return { ...h, quantity: value === "" ? 0 : Number(value) };
      }
      if (field === "avg_price") {
        return { ...h, avg_price: value === "" ? 0 : Number(value) };
      }
      return h;
    });

    setPortfolio({ ...portfolio, holdings: updated });
  }

  function addHolding() {
    if (!portfolio) return;
    setPortfolio({
      ...portfolio,
      holdings: [...portfolio.holdings, makeEmptyHolding()],
    });
  }

  function removeHolding(index: number) {
    if (!portfolio) return;
    const updated = portfolio.holdings.filter((_, i) => i !== index);
    setPortfolio({ ...portfolio, holdings: updated });
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    if (!portfolio) return;

    setSaving(true);
    setError(null);
    setMessage(null);

    try {
      const cleaned: PortfolioSnapshot = {
        cash: Number(portfolio.cash) || 0,
        holdings: portfolio.holdings.filter(
          (h) => h.symbol.trim() !== "" && h.quantity > 0
        ),
      };

      const saved = await savePortfolio(cleaned);
      setPortfolio(saved);
      setMessage("Portfolio saved.");
    } catch (err) {
      console.error(err);
      setError("Failed to save portfolio.");
    } finally {
      setSaving(false);
    }
  }

  const totalInvested = (portfolio?.holdings ?? []).reduce(
    (acc, h) => acc + (Number(h.quantity) || 0) * (Number(h.avg_price) || 0),
    0,
  );
  const cash = Number(portfolio?.cash ?? 0);
  const totalValue = totalInvested + cash;
  const canViewAdvanced = role === "premium" || role === "admin" || role === "manager";

  return (
    <RequireAuth>
      <div className="space-y-6">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Portfolio</h1>
          <p className="max-w-2xl text-sm text-slate-300">
            Edit your current holdings and cash. This snapshot is used by the
            recommendation engine.
          </p>
          <div className="mt-2 inline-flex items-center rounded-full border border-slate-700 bg-slate-900 px-2.5 py-0.5 text-[10px] text-slate-300">
            Access tier: {role}
          </div>
        </div>

        {portfolio && (
          <section className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">Total portfolio value</div>
              <div className="mt-1 text-lg font-semibold text-slate-100">${totalValue.toFixed(2)}</div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">Invested in user stocks</div>
              <div className="mt-1 text-lg font-semibold text-sky-300">${totalInvested.toFixed(2)}</div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">Cash reserve</div>
              <div className="mt-1 text-lg font-semibold text-emerald-300">${cash.toFixed(2)}</div>
            </div>
          </section>
        )}

        <form
          onSubmit={handleSave}
          className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm"
        >
          {loading && (
            <p className="text-xs text-slate-400">
              Loading portfolio…
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

          {portfolio && (
            <>
              <div className="space-y-1">
                <label className="text-xs font-medium text-slate-200">
                  Cash balance
                </label>
                <input
                  type="number"
                  value={portfolio.cash}
                  onChange={(e) =>
                    setPortfolio({
                      ...portfolio,
                      cash: e.target.value === "" ? 0 : Number(e.target.value),
                    })
                  }
                  className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                  min={0}
                  step="0.01"
                />
              </div>

              <div className="mt-4">
                <div className="mb-2 flex items-center justify-between">
                  <h2 className="text-sm font-medium text-slate-100">
                    Holdings
                  </h2>

                  <button
                    type="button"
                    onClick={addHolding}
                    className="rounded-md border border-slate-600 bg-slate-800 px-2 py-1 text-[11px] text-slate-50 hover:bg-slate-700"
                  >
                    Add row
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="min-w-full text-left text-xs text-slate-200">
                    <thead className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                      <tr>
                        <th className="px-2 py-1">Symbol</th>
                        <th className="px-2 py-1">Quantity</th>
                        <th className="px-2 py-1">Avg Price</th>
                        <th className="px-2 py-1">Market Value</th>
                        {canViewAdvanced && <th className="px-2 py-1">Allocation</th>}
                        <th className="px-2 py-1"></th>
                      </tr>
                    </thead>

                    <tbody>
                      {portfolio.holdings.map((h, idx) => {
                        const value = (Number(h.quantity) || 0) * (Number(h.avg_price) || 0);
                        const allocationPct = totalValue > 0 ? (value / totalValue) * 100 : 0;
                        return (
                        <tr key={idx} className="border-b border-slate-900 hover:bg-slate-950/60">
                          <td className="px-2 py-1">
                            <input
                              type="text"
                              value={h.symbol}
                              onChange={(e) =>
                                updateHolding(idx, "symbol", e.target.value)
                              }
                              className="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                              placeholder="AAPL"
                            />
                          </td>

                          <td className="px-2 py-1">
                            <input
                              type="number"
                              value={h.quantity}
                              onChange={(e) =>
                                updateHolding(idx, "quantity", e.target.value)
                              }
                              className="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                              min={0}
                            />
                          </td>

                          <td className="px-2 py-1">
                            <input
                              type="number"
                              value={h.avg_price}
                              onChange={(e) =>
                                updateHolding(idx, "avg_price", e.target.value)
                              }
                              className="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-50 focus:outline-none focus:ring-1 focus:ring-sky-500"
                              min={0}
                              step="0.01"
                            />
                          </td>

                          <td className="px-2 py-1 font-medium text-slate-200">
                            ${value.toFixed(2)}
                          </td>

                          {canViewAdvanced && (
                            <td className="px-2 py-1">
                              <div className="w-32">
                                <div className="mb-1 text-[10px] text-slate-400">
                                  {allocationPct.toFixed(1)}%
                                </div>
                                <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                                  <div
                                    className="h-full bg-sky-500"
                                    style={{ width: `${Math.max(0, Math.min(100, allocationPct))}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                          )}

                          <td className="px-2 py-1 text-right">
                            <button
                              type="button"
                              onClick={() => removeHolding(idx)}
                              className="text-[11px] text-slate-400 hover:text-red-400"
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      )})}

                      {portfolio.holdings.length === 0 && (
                        <tr>
                          <td
                            colSpan={canViewAdvanced ? 6 : 5}
                            className="px-2 py-2 text-center text-[11px] text-slate-500"
                          >
                            No holdings. Add at least one row or keep only cash.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                <div className="mt-3 rounded-lg border border-slate-700 bg-slate-950/70 p-3 text-[11px] text-slate-300">
                  {canViewAdvanced
                    ? "Advanced role enabled: allocation bars and richer stock exposure view are active for simulation and policy checks."
                    : "Standard role enabled: you can fully maintain your stocks and cash; advanced simulation overlays are available on premium tier."}
                </div>
              </div>
            </>
          )}

          <button
            type="submit"
            disabled={saving || !portfolio}
            className="mt-4 inline-flex items-center rounded-md border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-50 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {saving ? "Saving…" : "Save portfolio"}
          </button>
        </form>
      </div>
    </RequireAuth>
  );
}