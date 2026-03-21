// Portfolio management page for viewing holdings, market prices, and ML-based recommendations.
"use client";

import { useEffect, useState } from "react";
import type { Holding, PortfolioSnapshot, Recommendation } from "@/lib/types";
import { getMarketHistory, getPortfolio, getRecommendations, mlPredict, savePortfolio } from "@/lib/api-client";
import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/lib/auth-context";

function makeEmptyHolding(): Holding {
  return {
    symbol: "",
    quantity: 0,
    avg_price: 0,
  };
}

  function allocationWidthClass(pct: number): string {
    if (pct <= 0) return "w-0";
    if (pct <= 10) return "w-1/12";
    if (pct <= 20) return "w-2/12";
    if (pct <= 30) return "w-3/12";
    if (pct <= 40) return "w-4/12";
    if (pct <= 50) return "w-5/12";
    if (pct <= 60) return "w-6/12";
    if (pct <= 70) return "w-7/12";
    if (pct <= 80) return "w-8/12";
    if (pct <= 90) return "w-9/12";
    return "w-full";
  }

function buyPriceFromHistory(raw: unknown, buyDate: string): number {
  const parsed = raw as { points?: Array<{ t?: string; c?: number }> };
  const points = Array.isArray(parsed?.points) ? parsed.points : [];
  if (!points.length) return 0;

  let exact = 0;
  let prior = 0;
  for (const p of points) {
    const d = String(p?.t || "").slice(0, 10);
    const c = Number(p?.c || 0);
    if (!Number.isFinite(c) || c <= 0) continue;
    if (d === buyDate) {
      exact = c;
      break;
    }
    if (d && d < buyDate) {
      prior = c;
    }
  }

  if (exact > 0) return exact;
  if (prior > 0) return prior;

  const firstValid = points.find((p) => Number.isFinite(Number(p?.c || 0)) && Number(p?.c || 0) > 0);
  return Number(firstValid?.c || 0);
}

export default function PortfolioPage() {
  const { role } = useAuth();
  const [portfolio, setPortfolio] = useState<PortfolioSnapshot | null>(null);
  const [latestPrices, setLatestPrices] = useState<Record<string, number>>({});
  const [advice, setAdvice] = useState<Recommendation[]>([]);
  const [adviceLoading, setAdviceLoading] = useState(false);
  const [adviceError, setAdviceError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [autofilling, setAutofilling] = useState(false);
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

  useEffect(() => {
    let cancelled = false;

    async function fillAdminPortfolio() {
      if (role !== "admin" || !portfolio) return;
      if ((portfolio.holdings ?? []).length > 0) return;
      const buyDate = "2025-03-13";

      const basket = [
        "SPY",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "JPM",
        "UNH",
        "XOM",
        "TSLA",
        "AVGO",
        "AMD",
        "NFLX",
        "GS",
        "COST",
        "LLY",
        "V",
        "WMT",
        "HD",
      ];

      try {
        setAutofilling(true);
        const pred = await mlPredict({ stock_basket: basket, lookback_days: 30 });
        const top = [...(pred.items ?? [])]
          .sort((a, b) => b.probability_up - a.probability_up)
          .slice(0, 10);

        const prices = new Map<string, number>();
        await Promise.all(
          top.map(async (item) => {
            try {
              const hist = await getMarketHistory(item.symbol, 520);
              const px = buyPriceFromHistory(hist, buyDate);
              prices.set(item.symbol, px > 0 ? px : 100);
            } catch {
              prices.set(item.symbol, 100);
            }
          })
        );

        const autoHoldings: Holding[] = top.map((item) => ({
          symbol: item.symbol,
          quantity: 100,
          avg_price: prices.get(item.symbol) ?? 100,
        }));

        const next: PortfolioSnapshot = { ...portfolio, holdings: autoHoldings };
        const saved = await savePortfolio(next);
        if (!cancelled) {
          setPortfolio(saved);
          setMessage("Admin portfolio auto-filled: quantity 100 and buy-date (2025-03-13) average prices.");
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (!cancelled) setAutofilling(false);
      }
    }

    fillAdminPortfolio();
    return () => {
      cancelled = true;
    };
  }, [role, portfolio]);

  useEffect(() => {
    let cancelled = false;

    async function normalizeLegacyAdminPortfolio() {
      if (role !== "admin" || !portfolio) return;
      const hs = portfolio.holdings ?? [];
      if (!hs.length) return;

      const isLegacySeed = hs.every((h) => {
        const q = Number(h.quantity || 0);
        const p = Number(h.avg_price || 0);
        return Number.isFinite(q) && Number.isFinite(p) && q === 1 && p === 100;
      });

      if (!isLegacySeed) return;

      const buyDate = "2025-03-13";

      try {
        setAutofilling(true);
        const priceEntries = await Promise.all(
          hs.map(async (h) => {
            const sym = String(h.symbol || "").trim().toUpperCase();
            if (!sym) return [sym, 100] as const;
            try {
              const hist = await getMarketHistory(sym, 520);
              const px = buyPriceFromHistory(hist, buyDate);
              return [sym, px > 0 ? px : 100] as const;
            } catch {
              return [sym, 100] as const;
            }
          })
        );

        const bySymbol = new Map<string, number>(priceEntries);
        const normalized: PortfolioSnapshot = {
          ...portfolio,
          holdings: hs.map((h) => {
            const sym = String(h.symbol || "").trim().toUpperCase();
            return {
              ...h,
              symbol: sym,
              quantity: 100,
              avg_price: bySymbol.get(sym) ?? 100,
            };
          }),
        };

        const saved = await savePortfolio(normalized);
        if (!cancelled) {
          setPortfolio(saved);
          setMessage("Admin portfolio normalized: quantity 100 and per-stock buy-date average price applied.");
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (!cancelled) setAutofilling(false);
      }
    }

    normalizeLegacyAdminPortfolio();
    return () => {
      cancelled = true;
    };
  }, [role, portfolio]);

  useEffect(() => {
    let cancelled = false;

    async function loadLatestPrices() {
      const symbols = Array.from(
        new Set(
          (portfolio?.holdings ?? [])
            .map((h) => String(h.symbol || "").trim().toUpperCase())
            .filter((s) => s.length > 0)
        )
      );

      if (!symbols.length) {
        if (!cancelled) setLatestPrices({});
        return;
      }

      const entries = await Promise.all(
        symbols.map(async (sym) => {
          try {
            const hist = await getMarketHistory(sym, 10);
            const points = Array.isArray((hist as any)?.points) ? ((hist as any).points as Array<{ c?: number }>) : [];
            let px = 0;
            for (let i = points.length - 1; i >= 0; i--) {
              const c = Number(points[i]?.c || 0);
              if (Number.isFinite(c) && c > 0) {
                px = c;
                break;
              }
            }
            return [sym, px] as const;
          } catch {
            return [sym, 0] as const;
          }
        })
      );

      if (!cancelled) {
        const out: Record<string, number> = {};
        for (const [sym, px] of entries) {
          if (px > 0) out[sym] = px;
        }
        setLatestPrices(out);
      }
    }

    loadLatestPrices();
    return () => {
      cancelled = true;
    };
  }, [portfolio?.holdings]);

  useEffect(() => {
    let cancelled = false;

    async function loadAdvice() {
      if (!portfolio || !(portfolio.holdings ?? []).length) {
        if (!cancelled) {
          setAdvice([]);
          setAdviceError(null);
        }
        return;
      }

      try {
        setAdviceLoading(true);
        setAdviceError(null);
        const recs = await getRecommendations(portfolio);
        if (!cancelled) setAdvice(recs || []);
      } catch {
        if (!cancelled) {
          setAdvice([]);
          setAdviceError("Live ML advice is unavailable for this account role right now.");
        }
      } finally {
        if (!cancelled) setAdviceLoading(false);
      }
    }

    loadAdvice();
    return () => {
      cancelled = true;
    };
  }, [portfolio]);

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

  const totalInvested = (portfolio?.holdings ?? []).reduce((acc, h) => {
    const qty = Number(h.quantity) || 0;
    const avg = Number(h.avg_price) || 0;
    const sym = String(h.symbol || "").trim().toUpperCase();
    const live = Number(latestPrices[sym] || 0);
    const px = live > 0 ? live : avg;
    return acc + qty * px;
  }, 0);
  const totalCostBasis = (portfolio?.holdings ?? []).reduce((acc, h) => {
    const qty = Number(h.quantity) || 0;
    const avg = Number(h.avg_price) || 0;
    return acc + qty * avg;
  }, 0);
  const unrealizedPnl = totalInvested - totalCostBasis;
  const unrealizedReturnPct = totalCostBasis > 0 ? (unrealizedPnl / totalCostBasis) * 100 : 0;
  const cash = Number(portfolio?.cash ?? 0);
  const totalValue = totalInvested + cash;
  const canViewAdvanced = role === "premium" || role === "admin" || role === "manager";

  const positionRows = (portfolio?.holdings ?? []).map((h) => {
    const qty = Number(h.quantity) || 0;
    const avg = Number(h.avg_price) || 0;
    const sym = String(h.symbol || "").trim().toUpperCase();
    const live = Number(latestPrices[sym] || 0);
    const px = live > 0 ? live : avg;
    const marketValue = qty * px;
    const costValue = qty * avg;
    const pnl = marketValue - costValue;
    return {
      symbol: sym,
      marketValue,
      costValue,
      pnl,
      weight: totalInvested > 0 ? marketValue / totalInvested : 0,
    };
  });

  const maxWeight = positionRows.length ? Math.max(...positionRows.map((x) => x.weight)) : 0;
  const cashRatio = totalValue > 0 ? cash / totalValue : 0;
  const effectiveN = (() => {
    const sumSq = positionRows.reduce((acc, x) => acc + x.weight * x.weight, 0);
    if (sumSq <= 1e-9) return 0;
    return 1 / sumSq;
  })();

  const gainers = positionRows.filter((x) => x.pnl > 0).length;
  const losers = positionRows.filter((x) => x.pnl < 0).length;

  const adviceMap = new Map(advice.map((r) => [String(r.symbol || "").toUpperCase(), r]));
  const sellSignals = advice.filter((x) => x.action === "SELL").length;
  const holdSignals = advice.filter((x) => x.action === "HOLD").length;
  const buySignals = advice.filter((x) => x.action === "BUY").length;

  const riskFlags: string[] = [];
  if (maxWeight > 0.25) {
    riskFlags.push("Single-name concentration is high (>25%). Trim the largest position.");
  }
  if (cashRatio < 0.05) {
    riskFlags.push("Cash buffer is low (<5%). Keep some dry powder for volatility spikes.");
  }
  if ((portfolio?.holdings ?? []).length < 5) {
    riskFlags.push("Diversification is limited (<5 holdings). Add uncorrelated names.");
  }
  if (sellSignals >= 3) {
    riskFlags.push("ML engine is flagging multiple SELL signals. Consider de-risking weakest names.");
  }

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

        {portfolio && (
          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">Unrealized P/L</div>
              <div className={["mt-1 text-lg font-semibold", unrealizedPnl >= 0 ? "text-emerald-300" : "text-rose-300"].join(" ")}>
                {unrealizedPnl >= 0 ? "+" : ""}${unrealizedPnl.toFixed(2)}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">Return on advice baseline</div>
              <div className={["mt-1 text-lg font-semibold", unrealizedReturnPct >= 0 ? "text-emerald-300" : "text-rose-300"].join(" ")}>
                {unrealizedReturnPct >= 0 ? "+" : ""}{unrealizedReturnPct.toFixed(2)}%
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">Winners / Losers</div>
              <div className="mt-1 text-lg font-semibold text-slate-100">{gainers} / {losers}</div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-[10px] uppercase tracking-wide text-slate-400">Diversification score</div>
              <div className="mt-1 text-lg font-semibold text-slate-100">{effectiveN.toFixed(1)}</div>
            </div>
          </section>
        )}

        {portfolio && (
          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm">
            <h2 className="text-sm font-semibold text-slate-100">Risk management outlook</h2>
            <p className="mt-1 text-xs text-slate-300">
              This combines concentration, liquidity buffer, diversification, and live ML advice to help manage oncoming risk.
            </p>

            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                <div className="text-[10px] uppercase tracking-wide text-slate-400">Largest position weight</div>
                <div className="mt-1 text-sm font-semibold text-slate-100">{(maxWeight * 100).toFixed(1)}%</div>
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                <div className="text-[10px] uppercase tracking-wide text-slate-400">Cash buffer</div>
                <div className="mt-1 text-sm font-semibold text-slate-100">{(cashRatio * 100).toFixed(1)}%</div>
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                <div className="text-[10px] uppercase tracking-wide text-slate-400">ML advice mix</div>
                <div className="mt-1 text-sm font-semibold text-slate-100">BUY {buySignals} / HOLD {holdSignals} / SELL {sellSignals}</div>
              </div>
            </div>

            <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-xs">
              {adviceLoading ? (
                <div className="text-slate-300">Loading live ML advice…</div>
              ) : adviceError ? (
                <div className="text-amber-300">{adviceError}</div>
              ) : riskFlags.length ? (
                <ul className="space-y-1 text-slate-200">
                  {riskFlags.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              ) : (
                <div className="text-emerald-300">No immediate structural risk flags from current portfolio metrics.</div>
              )}
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

          {autofilling && (
            <p className="text-xs text-slate-400">
              Building admin holdings from top ML opportunities with buy-date (2025-03-13) baseline…
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
                  title="Cash balance"
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
                        <th className="px-2 py-1">Advice</th>
                        <th className="px-2 py-1"></th>
                      </tr>
                    </thead>

                    <tbody>
                      {portfolio.holdings.map((h, idx) => {
                        const qty = Number(h.quantity) || 0;
                        const avg = Number(h.avg_price) || 0;
                        const sym = String(h.symbol || "").trim().toUpperCase();
                        const live = Number(latestPrices[sym] || 0);
                        const priceForValue = live > 0 ? live : avg;
                        const value = qty * priceForValue;
                        const allocationPct = totalValue > 0 ? (value / totalValue) * 100 : 0;
                        const adviceItem = adviceMap.get(sym);
                        const adviceAction = adviceItem?.action ?? "HOLD";
                        return (
                        <tr key={idx} className="border-b border-slate-900 hover:bg-slate-950/60">
                          <td className="px-2 py-1">
                            <input
                              type="text"
                              title="Holding symbol"
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
                              title="Holding quantity"
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
                              title="Holding average price"
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
                                    className={["h-full bg-sky-500", allocationWidthClass(allocationPct)].join(" ")}
                                  />
                                </div>
                              </div>
                            </td>
                          )}

                          <td className="px-2 py-1">
                            <span
                              className={[
                                "inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium",
                                adviceAction === "BUY"
                                  ? "border-emerald-500/50 bg-emerald-900/30 text-emerald-200"
                                  : adviceAction === "SELL"
                                    ? "border-rose-500/50 bg-rose-900/30 text-rose-200"
                                    : "border-slate-600 bg-slate-800 text-slate-300",
                              ].join(" ")}
                            >
                              {adviceAction}
                            </span>
                          </td>

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
                            colSpan={canViewAdvanced ? 7 : 6}
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