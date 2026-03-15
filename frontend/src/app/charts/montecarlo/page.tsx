"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { getMonteCarloChart } from "@/lib/api-client";
import type { MonteCarloResult } from "@/lib/types";
import { RequireAuth } from "@/components/require-auth";

const SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "META", "NFLX"];
const HORIZONS = [
  { label: "3 Months (63d)",   days: 63  },
  { label: "6 Months (126d)",  days: 126 },
  { label: "1 Year (252d)",    days: 252 },
  { label: "2 Years (504d)",   days: 504 },
];

const PERCENTILE_COLORS: Record<string, string> = {
  "5":  "#E84C4C",
  "25": "#F5A623",
  "50": "#FFD700",
  "75": "#4C9BE8",
  "95": "#2ECC71",
};

const PERCENTILE_CLASSES: Record<string, string> = {
  "5":  "text-red-400",
  "25": "text-amber-400",
  "50": "text-yellow-300",
  "75": "text-sky-400",
  "95": "text-emerald-400",
};

export default function MonteCarloPage() {
  const [symbol, setSymbol]     = useState("AAPL");
  const [horizon, setHorizon]   = useState(HORIZONS[2]);
  const [nSim, setNSim]         = useState(500);
  const [data, setData]         = useState<MonteCarloResult | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);

  async function load(savePng = false) {
    setLoading(true);
    setError(null);
    try {
      const res = await getMonteCarloChart(symbol, 730, nSim, horizon.days, null, savePng);
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Build fan chart data: one row per time step
  const fanData = (() => {
    if (!data) return [];
    const paths = data.percentile_paths;
    const len   = (paths["50"] ?? []).length;
    const step  = Math.max(1, Math.ceil(len / 200));
    const result = [];
    for (let i = 0; i < len; i += step) {
      result.push({
        day: i,
        p5:  (paths["5"]  ?? [])[i] ?? null,
        p25: (paths["25"] ?? [])[i] ?? null,
        p50: (paths["50"] ?? [])[i] ?? null,
        p75: (paths["75"] ?? [])[i] ?? null,
        p95: (paths["95"] ?? [])[i] ?? null,
      });
    }
    return result;
  })();

  // Build distribution histogram
  const distData = (data?.distribution ?? []).map((b) => ({
    price: ((b.bucket_low + b.bucket_high) / 2).toFixed(0),
    count: b.count,
    above: b.bucket_low >= (data?.last_price ?? 0),
  }));

  return (
    <RequireAuth>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-50">Monte Carlo Simulation</h1>
          <p className="mt-1 text-sm text-slate-400">
            ML model-driven price path simulation with percentile fan and final-price distribution.
          </p>
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
            <span className="text-slate-400 text-xs">Horizon</span>
            <select
              value={horizon.label}
              onChange={(e) => setHorizon(HORIZONS.find((h) => h.label === e.target.value) ?? HORIZONS[2])}
              className="rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            >
              {HORIZONS.map((h) => <option key={h.label}>{h.label}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-slate-400 text-xs">Simulations</span>
            <select
              value={nSim}
              onChange={(e) => setNSim(Number(e.target.value))}
              className="rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            >
              {[100, 250, 500, 1000].map((n) => <option key={n}>{n}</option>)}
            </select>
          </label>
          <button
            onClick={() => load(false)}
            disabled={loading}
            className="mt-auto rounded bg-emerald-600 px-5 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            {loading ? "Running…" : "Run Simulation"}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-700 bg-red-900/30 p-3 text-sm text-red-300">{error}</div>
        )}

        {data && (
          <>
            {/* Key metrics */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className="text-xl font-bold text-slate-100">${data.last_price.toFixed(2)}</div>
                <div className="text-xs text-slate-400 mt-1">Current Price</div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className={`text-xl font-bold ${data.expected_return_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {data.expected_return_pct >= 0 ? "+" : ""}{data.expected_return_pct.toFixed(1)}%
                </div>
                <div className="text-xs text-slate-400 mt-1">Expected Return</div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className={`text-xl font-bold ${data.prob_profit_pct >= 50 ? "text-emerald-400" : "text-red-400"}`}>
                  {data.prob_profit_pct.toFixed(1)}%
                </div>
                <div className="text-xs text-slate-400 mt-1">Probability of Profit</div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className="text-xl font-bold text-red-400">${data.p10_price.toFixed(2)}</div>
                <div className="text-xs text-slate-400 mt-1">P10 Downside</div>
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                <div className="text-xl font-bold text-emerald-400">${data.p90_price.toFixed(2)}</div>
                <div className="text-xs text-slate-400 mt-1">P90 Upside</div>
              </div>
            </div>

            {/* Fan chart */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">
                Percentile Fan Chart – {data.n_simulations.toLocaleString()} paths over {data.horizon_days} trading days
              </h2>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={fanData} margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="day" tick={{ fill: "#94a3b8", fontSize: 9 }} label={{ value: "Trading Days", fill: "#94a3b8", fontSize: 10, position: "insideBottom", offset: -2 }} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `$${v.toFixed(0)}`} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                    formatter={(v: number, name: string) => [`$${v.toFixed(2)}`, name]}
                  />
                  <Legend />
                  <ReferenceLine y={data.last_price} stroke="#ffffff" strokeDasharray="3 2" strokeOpacity={0.4} label={{ value: "Current", fill: "#94a3b8", fontSize: 8, position: "right" }} />
                  {(["95", "75", "50", "25", "5"] as const).map((p) => (
                    <Line
                      key={p}
                      type="monotone"
                      dataKey={`p${p}`}
                      name={`P${p}`}
                      stroke={PERCENTILE_COLORS[p]}
                      dot={false}
                      strokeWidth={p === "50" ? 2.5 : 1.2}
                      strokeDasharray={p === "50" ? undefined : p === "5" || p === "95" ? "4 2" : undefined}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Distribution histogram */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Final-Price Distribution</h2>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={distData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="price" tick={{ fill: "#94a3b8", fontSize: 9 }} label={{ value: "Price ($)", fill: "#94a3b8", fontSize: 10, position: "insideBottom", offset: -2 }} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                  <ReferenceLine x={data.last_price.toFixed(0)} stroke="white" strokeDasharray="3 2" label={{ value: "Current", fill: "#94a3b8", fontSize: 8 }} />
                  <Bar dataKey="count" name="Simulations"
                    fill="#4C9BE8"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Percentile breakdown table */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Percentile Breakdown at Day {data.horizon_days}</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-slate-300">
                  <thead>
                    <tr className="border-b border-slate-700 text-xs text-slate-400">
                      <th className="py-2 pr-4 text-left">Percentile</th>
                      <th className="py-2 pr-4 text-right">Price</th>
                      <th className="py-2 pr-4 text-right">Return</th>
                      <th className="py-2 text-left">Interpretation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(["5", "25", "50", "75", "95"] as const).map((p) => {
                      const path  = data.percentile_paths[p] ?? [];
                      const price = path.length > 0 ? path[path.length - 1] : 0;
                      const ret   = data.last_price > 0 ? ((price / data.last_price) - 1) * 100 : 0;
                      const labels: Record<string, string> = {
                        "5":  "Worst-case tail (5% chance below this)",
                        "25": "Downside scenario (25th percentile)",
                        "50": "Median expected outcome (50%)",
                        "75": "Upside scenario (75th percentile)",
                        "95": "Best-case tail (95% chance below this)",
                      };
                      return (
                        <tr key={p} className="border-b border-slate-800/60">
                          <td className={`py-2 pr-4 font-medium ${PERCENTILE_CLASSES[p] ?? "text-slate-300"}`}>P{p}</td>
                          <td className="py-2 pr-4 text-right font-mono">${price.toFixed(2)}</td>
                          <td className={`py-2 pr-4 text-right font-mono ${ret >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                            {ret >= 0 ? "+" : ""}{ret.toFixed(1)}%
                          </td>
                          <td className="py-2 text-xs text-slate-400">{labels[p]}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
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
          <div className="py-16 text-center text-emerald-400">Running {nSim.toLocaleString()} simulations…</div>
        )}
      </div>
    </RequireAuth>
  );
}
