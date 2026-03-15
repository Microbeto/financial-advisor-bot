"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { getBacktestChart } from "@/lib/api-client";
import type { BacktestResult } from "@/lib/types";
import { RequireAuth } from "@/components/require-auth";

const SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "META", "NFLX"];
const BENCHMARKS = ["SPY", "QQQ", "DIA", "IWM"];
const PERIODS: { label: string; days: number }[] = [
  { label: "1 Year",   days: 365  },
  { label: "2 Years",  days: 730  },
  { label: "3 Years",  days: 1095 },
  { label: "5 Years",  days: 1825 },
];

function StatBox({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/70 p-4 text-center">
      <div className={`text-2xl font-bold ${color ?? "text-sky-400"}`}>{value}</div>
      <div className="mt-1 text-xs font-medium text-slate-300">{label}</div>
      {sub && <div className="mt-0.5 text-[10px] text-slate-500">{sub}</div>}
    </div>
  );
}

function thinData(arr: { t: string }[], target = 200) {
  if (arr.length <= target) return arr;
  const step = Math.ceil(arr.length / target);
  return arr.filter((_, i) => i % step === 0);
}

export default function BacktestPage() {
  const [symbol, setSymbol]           = useState("AAPL");
  const [benchmark, setBenchmark]     = useState("SPY");
  const [period, setPeriod]           = useState(PERIODS[3]);
  const [capital, setCapital]         = useState(100000);
  const [commission, setCommission]   = useState(0.001);
  const [slippage, setSlippage]       = useState(2.0);
  const [data, setData]               = useState<BacktestResult | null>(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await getBacktestChart(symbol, benchmark, period.days, capital, commission, slippage);
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const barData = data
    ? [
        { name: "Before Friction", value: data.portfolio_return_before },
        { name: "After Friction",  value: data.portfolio_return_after  },
        { name: "Benchmark",       value: data.benchmark_return        },
        { name: "Alpha",           value: data.alpha                   },
      ]
    : [];

  const riskData = data
    ? [
        { name: "Sharpe Ratio",    value: data.sharpe_ratio             },
        { name: "Max Drawdown (%)", value: Math.abs(data.max_drawdown_pct) },
      ]
    : [];

  const equityData = thinData(data?.equity_series ?? []);
  const ddData     = thinData(data?.drawdown_series ?? []);

  const barColor = (v: number) => (v >= 0 ? "#4C9BE8" : "#E84C4C");

  return (
    <RequireAuth>
      <div className="space-y-6">
        {/* Title */}
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-50">Backtest Performance Report</h1>
            <p className="mt-1 text-sm text-slate-400">
              Buy-and-hold simulation with friction analysis. PNG saved to server model store.
            </p>
          </div>
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
            <span className="text-slate-400 text-xs">Benchmark</span>
            <select
              value={benchmark}
              onChange={(e) => setBenchmark(e.target.value)}
              className="rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            >
              {BENCHMARKS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-slate-400 text-xs">Period</span>
            <select
              value={period.label}
              onChange={(e) => setPeriod(PERIODS.find((p) => p.label === e.target.value) ?? PERIODS[3])}
              className="rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            >
              {PERIODS.map((p) => <option key={p.label}>{p.label}</option>)}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-slate-400 text-xs">Capital ($)</span>
            <input
              type="number"
              value={capital}
              min={1000}
              step={10000}
              onChange={(e) => setCapital(Number(e.target.value))}
              className="w-28 rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-slate-400 text-xs">Commission (%)</span>
            <input
              type="number"
              value={commission}
              min={0}
              max={0.05}
              step={0.0001}
              onChange={(e) => setCommission(Number(e.target.value))}
              className="w-24 rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-slate-400 text-xs">Slippage (bps)</span>
            <input
              type="number"
              value={slippage}
              min={0}
              max={50}
              step={0.5}
              onChange={(e) => setSlippage(Number(e.target.value))}
              className="w-24 rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            />
          </label>

          <button
            onClick={load}
            disabled={loading}
            className="mt-auto rounded bg-sky-600 px-5 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
          >
            {loading ? "Running…" : "Run Backtest"}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-700 bg-red-900/30 p-3 text-sm text-red-300">{error}</div>
        )}

        {data && (
          <>
            {/* Summary stats */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
              <StatBox label="Return (Before)"   value={`${data.portfolio_return_before.toFixed(2)}%`}  color={data.portfolio_return_before >= 0 ? "text-emerald-400" : "text-red-400"} />
              <StatBox label="Return (After)"    value={`${data.portfolio_return_after.toFixed(2)}%`}   color={data.portfolio_return_after  >= 0 ? "text-emerald-400" : "text-red-400"} />
              <StatBox label="Benchmark Return"  value={`${data.benchmark_return.toFixed(2)}%`}         color="text-slate-300" />
              <StatBox label="Alpha"             value={`${data.alpha.toFixed(2)}%`}                    color={data.alpha >= 0 ? "text-emerald-400" : "text-red-400"} />
              <StatBox label="Sharpe Ratio"      value={data.sharpe_ratio.toFixed(2)}                   color={data.sharpe_ratio >= 1 ? "text-emerald-400" : "text-amber-400"} />
              <StatBox label="Max Drawdown"      value={`${data.max_drawdown_pct.toFixed(2)}%`}         color="text-red-400" />
            </div>

            {/* Returns bar chart */}
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                <h2 className="mb-3 text-sm font-semibold text-slate-200">Returns Analysis</h2>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={barData} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={(v) => `${v.toFixed(1)}%`} />
                    <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`]} contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                    <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 2" />
                    <Bar dataKey="value" name="Return (%)"
                      fill="#4C9BE8"
                      label={{ position: "top", fill: "#cbd5e1", fontSize: 10, formatter: (v: number) => `${v.toFixed(2)}%` }}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                <h2 className="mb-3 text-sm font-semibold text-slate-200">Risk-Adjusted Performance</h2>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={riskData} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                    <Bar dataKey="value" name="Value"
                      fill="#2ECC71"
                      label={{ position: "top", fill: "#cbd5e1", fontSize: 11, formatter: (v: number) => v.toFixed(2) }}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Equity curve */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Equity Curve</h2>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={equityData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} interval={Math.ceil(equityData.length / 6)} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                    formatter={(v: number) => [`$${v.toLocaleString()}`]}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="strategy"       name={`${symbol} (before friction)`} stroke="#4C9BE8" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="strategy_after" name={`${symbol} (after friction)`}  stroke="#F5A623" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="benchmark"      name={benchmark}                      stroke="#2ECC71" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Drawdown */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Portfolio Drawdown</h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={ddData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} interval={Math.ceil(ddData.length / 6)} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `${v.toFixed(1)}%`} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} formatter={(v: number) => [`${v.toFixed(2)}%`]} />
                  <ReferenceLine y={0} stroke="#64748b" />
                  <Bar dataKey="drawdown" name="Drawdown %" fill="#E84C4C" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Friction metrics */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-sky-800 bg-slate-900/70 p-4 text-sm font-mono text-slate-300 space-y-1">
                <div className="mb-2 font-sans font-semibold text-slate-200">Trading Friction Metrics</div>
                <div>Total Trades: {data.num_trades}</div>
                <div>Total Friction Cost: ${data.total_friction_cost.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                <div>Avg Friction/Trade:  ${data.avg_friction_per_trade.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                <div>Commission Rate: {data.commission_rate_pct.toFixed(3)}%</div>
                <div>Slippage: {data.slippage_bps} bps</div>
                <div>Friction Drag: {data.friction_drag_pct.toFixed(4)}%</div>
              </div>

              <div className="rounded-xl border border-sky-800 bg-slate-900/70 p-4 text-sm font-mono text-slate-300 space-y-1">
                <div className="mb-2 font-sans font-semibold text-slate-200">Backtest Summary</div>
                <div>Period: {data.period_start} to {data.period_end}</div>
                <div>Initial Capital: ${data.initial_capital.toLocaleString()}</div>
                <div>Portfolio Return (Before): {data.portfolio_return_before.toFixed(2)}%</div>
                <div>Portfolio Return (After): {data.portfolio_return_after.toFixed(2)}%</div>
                <div>Benchmark Return: {data.benchmark_return.toFixed(2)}%</div>
                <div>Alpha: {data.alpha.toFixed(2)}%</div>
                <div>Sharpe Ratio: {data.sharpe_ratio.toFixed(2)}</div>
                <div>Max Drawdown: {data.max_drawdown_pct.toFixed(2)}%</div>
              </div>
            </div>
          </>
        )}

        {!data && !loading && !error && (
          <div className="py-16 text-center text-slate-500">Press &quot;Run Backtest&quot; to generate the report.</div>
        )}
        {loading && (
          <div className="py-16 text-center text-sky-400">Running backtest and saving PNG…</div>
        )}
      </div>
    </RequireAuth>
  );
}
