"use client";

import { useEffect, useState } from "react";
import {
  ComposedChart,
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { getValuationChart } from "@/lib/api-client";
import type { ValuationResult } from "@/lib/types";
import { RequireAuth } from "@/components/require-auth";

const SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "META", "NFLX"];
const PERIODS  = [
  { label: "6 Months",  days: 180 },
  { label: "1 Year",    days: 365 },
  { label: "18 Months", days: 548 },
  { label: "2 Years",   days: 730 },
];
const BB_PERIODS = [10, 14, 20, 30, 50];

function thinData<T>(arr: T[], max = 200): T[] {
  if (arr.length <= max) return arr;
  const step = Math.ceil(arr.length / max);
  return arr.filter((_, i) => i % step === 0);
}

export default function ValuationPage() {
  const [symbol,   setSymbol]   = useState("AAPL");
  const [period,   setPeriod]   = useState(PERIODS[1]);
  const [bbPeriod, setBbPeriod] = useState(20);
  const [data,     setData]     = useState<ValuationResult | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await getValuationChart(symbol, period.days, bbPeriod);
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const chartData = thinData(
    (data?.series ?? []).map((pt) => ({
      t:       pt.t,
      close:   pt.close,
      sma:     pt.sma,
      upper2:  pt.upper2,
      lower2:  pt.lower2,
      upper1:  pt.upper1,
      lower1:  pt.lower1,
      trend:   pt.trend,
    }))
  );

  const pbData = thinData(
    (data?.series ?? []).map((pt) => ({ t: pt.t, pct_b: pt.pct_b }))

  );

  const lastPt = data?.series?.at(-1);
  const pctB = data?.pct_b ?? lastPt?.pct_b ?? null;

  // Price zone label based on %B
  const priceZone = (() => {
    if (pctB == null) return "";
    const pb = pctB;
    if (pb > 1.0) return "Extremely Overbought";
    if (pb > 0.8) return "Overbought";
    if (pb > 0.5) return "Upper Band";
    if (pb > 0.2) return "Lower Band";
    if (pb > 0.0) return "Oversold";
    return "Extremely Oversold";
  })();

  const priceZoneColor = (() => {
    if (pctB == null) return "text-slate-300";
    const pb = pctB;
    if (pb > 0.8 || pb < 0.2) return "text-red-400";
    if (pb > 0.6 || pb < 0.4) return "text-amber-400";
    return "text-emerald-400";
  })();

  return (
    <RequireAuth>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-50">Valuation Confidence Intervals</h1>
          <p className="mt-1 text-sm text-slate-400">
            Bollinger Band envelopes (±1σ / ±2σ), %B oscillator, and linear price trend. PNG saved to server.
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
            <span className="text-slate-400 text-xs">Period</span>
            <select
              value={period.label}
              onChange={(e) => setPeriod(PERIODS.find((p) => p.label === e.target.value) ?? PERIODS[1])}
              className="rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            >
              {PERIODS.map((p) => <option key={p.label}>{p.label}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-slate-400 text-xs">BB Period</span>
            <select
              value={bbPeriod}
              onChange={(e) => setBbPeriod(Number(e.target.value))}
              className="rounded bg-slate-800 px-2 py-1 text-slate-100 border border-slate-600"
            >
              {BB_PERIODS.map((n) => <option key={n}>{n}</option>)}
            </select>
          </label>
          <button
            onClick={load}
            disabled={loading}
            className="mt-auto rounded bg-sky-600 px-5 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
          >
            {loading ? "Loading…" : "Analyze"}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-700 bg-red-900/30 p-3 text-sm text-red-300">{error}</div>
        )}

        {data && (
          <>
            {/* Stat boxes */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              {[
                { label: "Last Price",    value: `$${lastPt?.close?.toFixed(2) ?? "–"}`,                       color: "text-slate-100" },
                { label: "SMA",           value: `$${lastPt?.sma?.toFixed(2) ?? "–"}`,                         color: "text-sky-400" },
                { label: "+2σ Upper",     value: `$${lastPt?.upper2?.toFixed(2) ?? "–"}`,                      color: "text-red-400" },
                { label: "–2σ Lower",     value: `$${lastPt?.lower2?.toFixed(2) ?? "–"}`,                      color: "text-emerald-400" },
                { label: "%B",            value: pctB != null ? pctB.toFixed(3) : "–",                        color: priceZoneColor },
                { label: "Price Zone",    value: priceZone,                                                     color: priceZoneColor },
              ].map((s) => (
                <div key={s.label} className="rounded-lg border border-slate-700 bg-slate-900/70 p-3 text-center">
                  <div className={`text-lg font-bold ${s.color}`}>{s.value}</div>
                  <div className="text-xs text-slate-400 mt-1">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Main Bollinger Band chart */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Price with Bollinger Bands (±1σ / ±2σ)</h2>
              <ResponsiveContainer width="100%" height={380}>
                <ComposedChart data={chartData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} tickFormatter={(v: string) => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(v) => `$${v.toFixed(0)}`} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                    formatter={(v: number, name: string) => [`$${v.toFixed(2)}`, name]}
                    labelStyle={{ color: "#94a3b8" }}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px" }} />
                  {/* ±2σ outer envelope */}
                  <Area type="monotone" dataKey="upper2" name="+2σ" stroke="#E84C4C" strokeWidth={1} fill="#E84C4C" fillOpacity={0.08} dot={false} />
                  <Area type="monotone" dataKey="lower2" name="–2σ" stroke="#2ECC71" strokeWidth={1} fill="#2ECC71" fillOpacity={0.08} dot={false} />
                  {/* ±1σ inner envelope */}
                  <Area type="monotone" dataKey="upper1" name="+1σ" stroke="#F5A623" strokeWidth={1} fill="#F5A623" fillOpacity={0.06} strokeDasharray="4 2" dot={false} />
                  <Area type="monotone" dataKey="lower1" name="–1σ" stroke="#4C9BE8" strokeWidth={1} fill="#4C9BE8" fillOpacity={0.06} strokeDasharray="4 2" dot={false} />
                  {/* SMA and Trend */}
                  <Line type="monotone" dataKey="sma"   name="SMA"   stroke="#94a3b8" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="trend" name="Trend" stroke="#FFD700" strokeWidth={1.5} strokeDasharray="6 2" dot={false} />
                  {/* Close price */}
                  <Line type="monotone" dataKey="close" name="Close" stroke="#ffffff" strokeWidth={2} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* %B oscillator */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-2 text-sm font-semibold text-slate-200">
                %B Oscillator
                <span className="ml-2 text-xs text-slate-400">(0 = lower band, 0.5 = SMA, 1 = upper band)</span>
              </h2>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={pbData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 9 }} tickFormatter={(v: string) => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} domain={[-0.1, 1.1]} tickCount={5} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} formatter={(v: number) => [v.toFixed(3), "%B"]} />
                  <ReferenceLine y={1.0} stroke="#E84C4C" strokeDasharray="3 2" label={{ value: "OB+2σ", fill: "#E84C4C", fontSize: 8, position: "right" }} />
                  <ReferenceLine y={0.8} stroke="#F5A623" strokeDasharray="3 2" label={{ value: "OB+1σ", fill: "#F5A623", fontSize: 8, position: "right" }} />
                  <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="4 2" />
                  <ReferenceLine y={0.2} stroke="#4C9BE8" strokeDasharray="3 2" label={{ value: "OS–1σ", fill: "#4C9BE8", fontSize: 8, position: "right" }} />
                  <ReferenceLine y={0.0} stroke="#2ECC71" strokeDasharray="3 2" label={{ value: "OS–2σ", fill: "#2ECC71", fontSize: 8, position: "right" }} />
                  <Line type="monotone" dataKey="pct_b" name="%B" stroke="#FFD700" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Interpretation guide */}
            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h2 className="mb-3 text-sm font-semibold text-slate-200">Band Interpretation</h2>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 text-xs text-slate-300">
                {[
                  { range: "%B > 1.0",       color: "bg-red-500",     label: "Extremely Overbought – price above +2σ band" },
                  { range: "0.8 – 1.0",      color: "bg-orange-500",  label: "Overbought – price near +2σ upper envelope" },
                  { range: "0.5 – 0.8",      color: "bg-amber-500",   label: "Bullish – price above SMA midline" },
                  { range: "0.2 – 0.5",      color: "bg-sky-500",     label: "Bearish – price below SMA midline" },
                  { range: "0.0 – 0.2",      color: "bg-emerald-500", label: "Oversold – price near –2σ lower envelope" },
                  { range: "%B < 0.0",       color: "bg-green-600",   label: "Extremely Oversold – price below –2σ band" },
                ].map((row) => (
                  <div key={row.range} className="flex items-center gap-2">
                    <div className={`h-3 w-3 shrink-0 rounded-sm ${row.color}`} />
                    <span className="font-mono text-slate-400 w-16 shrink-0">{row.range}</span>
                    <span>{row.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {loading && (
          <div className="py-16 text-center text-sky-400">Calculating Bollinger bands, saving PNG…</div>
        )}
      </div>
    </RequireAuth>
  );
}
