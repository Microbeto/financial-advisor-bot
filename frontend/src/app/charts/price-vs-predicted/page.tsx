// ML model prediction accuracy visualization comparing predicted vs actual stock prices.
"use client";

import { useEffect, useState } from "react";
import {
  ComposedChart,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ReferenceLine,
} from "recharts";
import { getPriceVsPredictedChart } from "@/lib/api-client";
import { getCachedChartData, setCachedChartData } from "@/lib/chart-cache";
import type { PriceVsPredictedResult } from "@/lib/types";
import { RequireAuth } from "@/components/require-auth";

const SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "META", "NFLX"];
const PERIODS  = [
  { label: "3 Months",  days: 90  },
  { label: "6 Months",  days: 180 },
  { label: "1 Year",    days: 365 },
  { label: "18 Months", days: 548 },
];

function thinData<T>(arr: T[], max = 200): T[] {
  if (arr.length <= max) return arr;
  const step = Math.ceil(arr.length / max);
  return arr.filter((_, i) => i % step === 0);
}

// Custom dot renderer for buy/sell signals
function renderSignalDot(props: {
  cx?: number; cy?: number; payload?: { signal: string | null };
}) {
  const { cx, cy, payload } = props;
  if (!payload || !payload.signal || cx == null || cy == null) return <g />;
  const isBuy = payload.signal === "buy";
  return (
    <circle
      key={`${cx}-${cy}`}
      cx={cx}
      cy={cy}
      r={5}
      fill={isBuy ? "#2ECC71" : "#E84C4C"}
      stroke={isBuy ? "#27ae60" : "#c0392b"}
      strokeWidth={1.5}
    />
  );
}

export default function PriceVsPredictedPage() {
  const [symbol,  setSymbol]  = useState("AAPL");
  const [period,  setPeriod]  = useState(PERIODS[2]);
  const [data,    setData]    = useState<PriceVsPredictedResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [showingCached, setShowingCached] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  function cacheKey() {
    return `price-vs-pred:${symbol}:${period.days}`;
  }

  async function load(savePng = false) {
    const key = cacheKey();
    if (!savePng) {
      const cached = getCachedChartData<PriceVsPredictedResult>(key);
      if (cached) {
        setData(cached);
        setShowingCached(true);
      }
    }
    setLoading(true);
    setError(null);
    try {
      const res = await getPriceVsPredictedChart(symbol, period.days, null, savePng);
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

  // Merge price series with signal flags
  const priceData = thinData(
    (data?.series ?? []).map((pt) => ({
      t:         pt.t,
      close:     pt.close,
      predicted: pt.predicted,
      prob_up:   pt.prob_up,
      signal:    null as string | null, // filled below
    }))
  );

  // Overlay buy/sell signals on thinned data
  const signalIndex = new Map<string, string>(
    (data?.signals ?? []).map((s) => [s.t, s.signal])
  );
  priceData.forEach((pt) => {
    const sig = signalIndex.get(pt.t);
    if (sig) pt.signal = sig;
  });

  const probData = thinData(
    (data?.series ?? []).map((pt) => ({ t: pt.t, prob_up: pt.prob_up }))
  );

  const lastPt   = data?.series?.at(-1);
  const buySigs  = (data?.signals ?? []).filter((s) => s.signal === "buy").length;
  const sellSigs = (data?.signals ?? []).filter((s) => s.signal === "sell").length;

  return (
    <RequireAuth>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-50">Historical Price vs ML Predicted</h1>
          <p className="mt-1 text-sm text-slate-400">
            Actual close overlaid with chosen-model up-probability. Buy/sell crossover signals marked.
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
              onChange={(e) => setPeriod(PERIODS.find((p) => p.label === e.target.value) ?? PERIODS[2])}
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
            {/* Key stats */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: "Data Points",          value: data.n_points.toString(),          color: "text-slate-100" },
                { label: "Avg ML Confidence",     value: `${(data.avg_prob_up * 100).toFixed(1)}%`, color: data.avg_prob_up >= 0.5 ? "text-emerald-400" : "text-red-400" },
                { label: "Buy Signals",           value: buySigs.toString(),                color: "text-emerald-400" },
                { label: "Sell Signals",          value: sellSigs.toString(),               color: "text-red-400" },
              ].map((s) => (
                <div key={s.label} className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                  <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
                  <div className="text-xs text-slate-400 mt-1">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Price + predicted overlay */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-1 text-sm font-semibold text-slate-200">Price vs ML Predicted</h2>
              <p className="mb-3 text-xs text-slate-500">
                Green dots = buy crossover (ML prob_up crosses above 0.55). Red dots = sell crossover.
              </p>
              <ResponsiveContainer width="100%" height={360}>
                <ComposedChart data={priceData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} tickFormatter={(v: string) => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `$${v.toFixed(0)}`} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                    formatter={(v: number, name: string) => [`$${v.toFixed(2)}`, name]}
                    labelStyle={{ color: "#94a3b8" }}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px" }} />
                  <Line
                    type="monotone"
                    dataKey="close"
                    name="Actual Close"
                    stroke="#ffffff"
                    strokeWidth={2}
                    dot={renderSignalDot}
                    activeDot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name="ML Predicted"
                    stroke="#9B59B6"
                    strokeWidth={1.5}
                    strokeDasharray="5 2"
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* Up-probability area chart */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-2 text-sm font-semibold text-slate-200">ML Up-Probability</h2>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={probData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} tickFormatter={(v: string) => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "P(Up)"]} />
                  <ReferenceLine y={0.55} stroke="#2ECC71" strokeDasharray="3 2" label={{ value: "Buy 55%", fill: "#2ECC71", fontSize: 8, position: "right" }} />
                  <ReferenceLine y={0.50} stroke="#94a3b8" strokeDasharray="4 2" />
                  <ReferenceLine y={0.45} stroke="#E84C4C" strokeDasharray="3 2" label={{ value: "Sell 45%", fill: "#E84C4C", fontSize: 8, position: "right" }} />
                  <Area type="monotone" dataKey="prob_up" name="P(Up)" stroke="#9B59B6" fill="#9B59B6" fillOpacity={0.3} strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Signal detail table */}
            {data.signals.length > 0 && (
              <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                <h2 className="mb-3 text-sm font-semibold text-slate-200">Signal Log (latest 20)</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-slate-300">
                    <thead>
                      <tr className="border-b border-slate-700 text-slate-400">
                        <th className="py-1.5 pr-4 text-left">Date</th>
                        <th className="py-1.5 pr-4 text-left">Signal</th>
                        <th className="py-1.5 pr-4 text-right">Price</th>
                        <th className="py-1.5 text-right">P(Up)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...data.signals].reverse().slice(0, 20).map((sig) => (
                        <tr key={`${sig.t}-${sig.signal}`} className="border-b border-slate-800/60">
                          <td className="py-1 pr-4 font-mono">{sig.t}</td>
                          <td className={`py-1 pr-4 font-semibold uppercase ${sig.signal === "buy" ? "text-emerald-400" : "text-red-400"}`}>
                            {sig.signal}
                          </td>
                          <td className="py-1 pr-4 text-right font-mono">${sig.price.toFixed(2)}</td>
                          <td className="py-1 text-right font-mono">{(sig.prob_up * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

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
          <div className="py-16 text-center text-violet-400">Loading ML predictions…</div>
        )}
      </div>
    </RequireAuth>
  );
}
