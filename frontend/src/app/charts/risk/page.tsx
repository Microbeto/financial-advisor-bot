"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { getRiskDashboardChart } from "@/lib/api-client";
import { getCachedChartData, setCachedChartData } from "@/lib/chart-cache";
import type { RiskDashboardResult } from "@/lib/types";
import { RequireAuth } from "@/components/require-auth";

function thinData<T>(arr: T[], max = 200): T[] {
  if (arr.length <= max) return arr;
  const step = Math.ceil(arr.length / max);
  return arr.filter((_, i) => i % step === 0);
}

// Returns a tailwind bg colour class for correlation values -1 to 1
function corrColor(val: number): string {
  if (val >= 0.7)  return "bg-red-700";
  if (val >= 0.4)  return "bg-orange-600";
  if (val >= 0.1)  return "bg-amber-700";
  if (val >= -0.1) return "bg-slate-600";
  if (val >= -0.4) return "bg-sky-700";
  return "bg-blue-800";
}

export default function RiskDashboardPage() {
  const [data,    setData]    = useState<RiskDashboardResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [showingCached, setShowingCached] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  function cacheKey() {
    return "risk-dashboard:current-user";
  }

  async function load(savePng = false) {
    const key = cacheKey();
    if (!savePng) {
      const cached = getCachedChartData<RiskDashboardResult>(key);
      if (cached) {
        setData(cached);
        setShowingCached(true);
      }
    }
    setLoading(true);
    setError(null);
    try {
      const res = await getRiskDashboardChart(null, savePng);
      setData(res);
      setShowingCached(false);
      if (!savePng) setCachedChartData(key, res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const volData = thinData(data?.rolling_vol_series ?? []);
  const ddData  = thinData(data?.drawdown_series    ?? []);

  // Asset volatility bar data
  const assetVolData = Object.entries(data?.asset_volatilities ?? {}).map(
    ([symbol, vol]) => ({ symbol, annVol: +vol.toFixed(2) })
  );

  // Correlation matrix (number[][] indexed by data.symbols)
  const corrSymbols = data?.symbols ?? [];
  const corrMatrix  = data?.correlation_matrix ?? null;

  return (
    <RequireAuth>
      <div className="space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-50">Portfolio Risk Dashboard</h1>
            <p className="mt-1 text-sm text-slate-400">
              VaR / CVaR, rolling volatility, drawdown and asset correlation using ML-weighted exposures.
            </p>
            {showingCached && (
              <p className="mt-1 text-xs text-amber-400">Showing cached data. Refreshing in background…</p>
            )}
          </div>
          <button
            onClick={() => load(false)}
            disabled={loading}
            className="rounded bg-rose-600 px-5 py-1.5 text-sm font-medium text-white hover:bg-rose-500 disabled:opacity-50"
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-700 bg-red-900/30 p-3 text-sm text-red-300">{error}</div>
        )}

        {data && (
          <>
            {/* KPI Strip */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {[
                {
                  label: "Portfolio Value",
                  value: `$${data.portfolio_value.toLocaleString("en-US", { maximumFractionDigits: 0 })}`,
                  color: "text-slate-100",
                },
                {
                  label: `Daily VaR (${data.var_confidence * 100}%)`,
                  value: `${data.var_pct_daily.toFixed(2)}%`,
                  color: "text-red-400",
                },
                {
                  label: "Dollar VaR",
                  value: `$${data.var_dollar.toLocaleString("en-US", { maximumFractionDigits: 0 })}`,
                  color: "text-red-400",
                },
                {
                  label: `CVaR (${data.var_confidence * 100}%)`,
                  value: `${data.cvar_pct_daily.toFixed(2)}%`,
                  color: "text-orange-400",
                },
                {
                  label: "Max Drawdown",
                  value: `${data.max_drawdown_pct.toFixed(2)}%`,
                  color: "text-rose-400",
                },
              ].map((s) => (
                <div key={s.label} className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                  <div className={`text-xl font-bold ${s.color}`}>{s.value}</div>
                  <div className="text-xs text-slate-400 mt-1">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Rolling portfolio volatility */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Rolling 21-Day Portfolio Volatility (Annualised)</h2>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={volData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="index" tick={{ fill: "#94a3b8", fontSize: 9 }} label={{ value: "Day", fill: "#94a3b8", fontSize: 9, position: "insideBottom", offset: -2 }} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `${(v as number).toFixed(1)}%`} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                    formatter={(v: number) => [`${v.toFixed(2)}%`, "Ann. Vol"]}
                    labelStyle={{ color: "#94a3b8" }}
                  />

                  <Area type="monotone" dataKey="vol" name="Ann. Vol" stroke="#F5A623" fill="#F5A623" fillOpacity={0.2} strokeWidth={1.8} dot={false} connectNulls />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Drawdown */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Portfolio Drawdown</h2>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={ddData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="index" tick={{ fill: "#94a3b8", fontSize: 9 }} label={{ value: "Day", fill: "#94a3b8", fontSize: 9, position: "insideBottom", offset: -2 }} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `${(v as number).toFixed(1)}%`} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                    formatter={(v: number) => [`${v.toFixed(2)}%`, "Drawdown"]}
                    labelStyle={{ color: "#94a3b8" }}
                  />
                  <Area type="monotone" dataKey="drawdown" name="Drawdown" stroke="#E84C4C" fill="#E84C4C" fillOpacity={0.25} strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Asset volatility bar chart */}
            {assetVolData.length > 0 && (
              <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                <h2 className="mb-3 text-sm font-semibold text-slate-200">Asset Annualised Volatility (daily σ × √252)</h2>
                <ResponsiveContainer width="100%" height={Math.max(120, assetVolData.length * 36)}>
                  <BarChart data={assetVolData} layout="vertical" margin={{ left: 30, right: 30, top: 5, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                    <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
                    <YAxis type="category" dataKey="symbol" tick={{ fill: "#94a3b8", fontSize: 11 }} width={50} />
                    <Tooltip
                      contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                      formatter={(v: number) => [`${v.toFixed(2)}%`, "Ann. Vol"]}
                    />
                    <Bar dataKey="annVol" name="Ann. Vol %" fill="#4C9BE8" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Correlation matrix */}
            {corrMatrix && corrSymbols.length >= 2 && (
              <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                <h2 className="mb-3 text-sm font-semibold text-slate-200">Asset Correlation Matrix</h2>
                <div className="overflow-x-auto">
                  <table className="text-xs">
                    <thead>
                      <tr>
                        <th className="pr-2 py-1 text-slate-400 font-normal" />
                        {corrSymbols.map((k) => (
                          <th key={k} className="px-1 py-1 text-slate-300 font-semibold text-center w-16">{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {corrSymbols.map((rowKey, ri) => (
                        <tr key={rowKey}>
                          <td className="pr-2 py-0.5 text-slate-300 font-semibold whitespace-nowrap">{rowKey}</td>
                          {corrSymbols.map((colKey, ci) => {
                            const val = corrMatrix[ri]?.[ci] ?? 0;
                            return (
                              <td key={colKey} className={`px-1 py-0.5 text-center font-mono w-16 rounded-sm ${corrColor(val)}`}>
                                {val.toFixed(2)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {/* Legend */}
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-400">
                  {[
                    { label: "≥ 0.7 Highly correlated",   cls: "bg-red-700" },
                    { label: "0.4–0.7 Moderate",           cls: "bg-orange-600" },
                    { label: "0.1–0.4 Weak positive",      cls: "bg-amber-700" },
                    { label: "±0.1 Uncorrelated",          cls: "bg-slate-600" },
                    { label: "–0.4–(–0.1) Weak negative",  cls: "bg-sky-700" },
                    { label: "< –0.4 Negative hedge",      cls: "bg-blue-800" },
                  ].map((r) => (
                    <div key={r.label} className="flex items-center gap-1">
                      <div className={`h-3 w-3 rounded-sm ${r.cls}`} />
                      {r.label}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* VaR explanation */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 text-xs text-slate-400 space-y-1">
              <p><span className="text-slate-200 font-medium">Value at Risk (VaR):</span> With {(data.var_confidence * 100).toFixed(0)}% confidence, the portfolio will not lose more than <span className="text-red-400">{data.var_pct_daily.toFixed(2)}%</span> (${data.var_dollar.toLocaleString("en-US", { maximumFractionDigits: 0 })}) in a single trading day.</p>
              <p><span className="text-slate-200 font-medium">Conditional VaR (CVaR / Expected Shortfall):</span> On the worst days that exceed VaR, the average loss is <span className="text-orange-400">{data.cvar_pct_daily.toFixed(2)}%</span> of portfolio value.</p>
              <p className="text-slate-600">Historical simulation method using 2-year daily returns. For informational purposes only — not financial advice.</p>
            </div>

            <div className="flex justify-end">
              <button
                onClick={() => load(true)}
                disabled={loading}
                className="rounded border border-slate-600 bg-slate-800/70 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-slate-700 disabled:opacity-50"
              >
                Save PNG
              </button>
            </div>
          </>
        )}

        {loading && (
          <div className="py-16 text-center text-rose-400">Computing risk metrics…</div>
        )}
      </div>
    </RequireAuth>
  );
}
