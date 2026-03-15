"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
} from "recharts";
import { getTechnicalChart } from "@/lib/api-client";
import { getCachedChartData, setCachedChartData } from "@/lib/chart-cache";
import type { TechnicalResult, TechnicalPoint } from "@/lib/types";
import { RequireAuth } from "@/components/require-auth";

const SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "META", "NFLX", "JPM", "BRK-B"];
const PERIODS = [
  { label: "6 Months",  days: 180  },
  { label: "1 Year",    days: 365  },
  { label: "2 Years",   days: 730  },
];

function thinData(arr: TechnicalPoint[], target = 250) {
  if (arr.length <= target) return arr;
  const step = Math.ceil(arr.length / target);
  return arr.filter((_, i) => i % step === 0);
}

function rsiColor(v: number | null) {
  if (v === null) return "text-slate-400";
  if (v > 70) return "text-red-400";
  if (v < 30) return "text-emerald-400";
  return "text-violet-400";
}

export default function TechnicalPage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [period, setPeriod] = useState(PERIODS[1]);
  const [data, setData]     = useState<TechnicalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [showingCached, setShowingCached] = useState(false);
  const [error, setError]   = useState<string | null>(null);

  function cacheKey() {
    return `technical:${symbol}:${period.days}`;
  }

  async function load(savePng = false) {
    const key = cacheKey();
    if (!savePng) {
      const cached = getCachedChartData<TechnicalResult>(key);
      if (cached) {
        setData(cached);
        setShowingCached(true);
      }
    }
    setLoading(true);
    setError(null);
    try {
      const res = await getTechnicalChart(symbol, period.days, null, savePng);
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

  const thin = thinData(data?.series ?? []);
  const rsiNow = data?.current_rsi ?? null;
  const macdNow = data?.current_macd ?? null;
  const sigNow  = data?.current_signal ?? null;

  return (
    <RequireAuth>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-50">RSI &amp; MACD Technical Indicators</h1>
          <p className="mt-1 text-sm text-slate-400">
            ML model-driven RSI and MACD views. PNG saves only when enabled.
          </p>
          {showingCached && (
            <p className="mt-1 text-xs text-amber-400">Showing cached data. Refreshing in background…</p>
          )}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-3 rounded-xl border border-slate-700 bg-slate-900/60 p-4 text-sm">
          <label className="flex flex-col gap-1">
            <span className="text-slate-400 text-xs">Symbol</span>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            >
              {SYMBOLS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-slate-400 text-xs">Period</span>
            <select
              value={period.label}
              onChange={(e) => setPeriod(PERIODS.find((p) => p.label === e.target.value) ?? PERIODS[1])}
              className="rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            >
              {PERIODS.map((p) => <option key={p.label}>{p.label}</option>)}
            </select>
          </label>
          <button
            onClick={() => load(false)}
            disabled={loading}
            className="mt-auto rounded bg-violet-600 px-5 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50"
          >
            {loading ? "Loading…" : "Load"}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-700 bg-red-900/30 p-3 text-sm text-red-300">{error}</div>
        )}

        {data && (
          <>
            {/* Summary row */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className={`text-xl font-bold ${rsiColor(rsiNow)}`}>
                  {rsiNow !== null ? rsiNow.toFixed(1) : "–"}
                </div>
                <div className="text-xs text-slate-400 mt-1">Current RSI (14)</div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  {rsiNow !== null && rsiNow > 70 ? "Overbought" : rsiNow !== null && rsiNow < 30 ? "Oversold" : "Neutral"}
                </div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className={`text-xl font-bold ${(macdNow ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {macdNow !== null ? macdNow.toFixed(3) : "–"}
                </div>
                <div className="text-xs text-slate-400 mt-1">MACD Line</div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className="text-xl font-bold text-amber-400">
                  {sigNow !== null ? sigNow.toFixed(3) : "–"}
                </div>
                <div className="text-xs text-slate-400 mt-1">Signal Line</div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className="text-xl font-bold text-sky-400">{data.overbought_count + data.oversold_count}</div>
                <div className="text-xs text-slate-400 mt-1">OB/OS Signals</div>
                <div className="text-[10px] text-slate-500 mt-0.5">{data.overbought_count} OB / {data.oversold_count} OS</div>
              </div>
            </div>

            {/* Price + EMA */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Price &amp; EMAs</h2>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={thin} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} interval={Math.ceil(thin.length / 6)} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `$${v.toFixed(0)}`} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} formatter={(v: number) => [`$${v.toFixed(2)}`]} />
                  <Legend />
                  <Line type="monotone" dataKey="close"  name="Close"  stroke="#4C9BE8" dot={false} strokeWidth={1.8} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* RSI */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">RSI ({data.rsi_period})</h2>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={thin} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} interval={Math.ceil(thin.length / 6)} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} formatter={(v: number) => [v.toFixed(1)]} />
                  <ReferenceLine y={70} stroke="#E84C4C" strokeDasharray="4 2" label={{ value: "OB 70", fill: "#E84C4C", fontSize: 9, position: "right" }} />
                  <ReferenceLine y={30} stroke="#2ECC71" strokeDasharray="4 2" label={{ value: "OS 30", fill: "#2ECC71", fontSize: 9, position: "right" }} />
                  <ReferenceLine y={50} stroke="#64748b" strokeDasharray="2 2" />
                  <Line type="monotone" dataKey="rsi" name="RSI" stroke="#9B59B6" dot={false} strokeWidth={1.6} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* MACD */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">
                MACD ({data.macd_fast}, {data.macd_slow}, {data.macd_signal_period})
              </h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={thin} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} interval={Math.ceil(thin.length / 6)} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} formatter={(v: number) => [v !== null ? v.toFixed(4) : "–"]} />
                  <Legend />
                  <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 2" />
                  <Bar dataKey="histogram" name="Histogram"
                    fill="#4C9BE8"
                    // color by sign via recharts Cell would need import; simplified single color
                  />
                  <Line type="monotone" dataKey="macd"   name="MACD"   stroke="#4C9BE8" dot={false} strokeWidth={1.5} connectNulls />
                  <Line type="monotone" dataKey="signal" name="Signal" stroke="#F5A623" dot={false} strokeWidth={1.2} strokeDasharray="4 2" connectNulls />
                </BarChart>
              </ResponsiveContainer>
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
          <div className="py-16 text-center text-violet-400">Fetching and computing indicators…</div>
        )}
      </div>
    </RequireAuth>
  );
}
