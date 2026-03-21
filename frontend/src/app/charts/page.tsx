// Charts index page with navigation to backtest, technical, Monte Carlo, and valuation visualizations.
"use client";

import Link from "next/link";
import { RequireAuth } from "@/components/require-auth";

const CHART_CARDS = [
  {
    href: "/charts/backtest",
    title: "Backtest Performance",
    description:
      "Return analysis, risk-adjusted performance (Sharpe, max drawdown), trading friction metrics and equity curves versus a benchmark.",
    tone: "border-sky-800/40",
    icon: "📈",
  },
  {
    href: "/charts/technical",
    title: "RSI & MACD Indicators",
    description:
      "Relative Strength Index and Moving Average Convergence/Divergence with overbought/oversold signals for any symbol.",
    tone: "border-violet-800/40",
    icon: "🔬",
  },
  {
    href: "/charts/montecarlo",
    title: "Monte Carlo Simulation",
    description:
      "Geometric Brownian Motion simulation with percentile fan chart (P5–P95), final-price distribution, and probability of profit.",
    tone: "border-emerald-800/40",
    icon: "🎲",
  },
  {
    href: "/charts/valuation",
    title: "Valuation Confidence Intervals",
    description:
      "Bollinger bands (±1σ / ±2σ), %B oscillator, 52-week high/low channel, and linear regression price trend.",
    tone: "border-amber-800/40",
    icon: "📊",
  },
  {
    href: "/charts/price-vs-predicted",
    title: "Price vs Predicted",
    description:
      "Historical close price overlaid with ML up-probability. Crossover buy/sell signals marked directly on the chart.",
    tone: "border-rose-800/40",
    icon: "🤖",
  },
  {
    href: "/charts/risk",
    title: "Risk Dashboard",
    description:
      "Portfolio VaR (95%), CVaR, rolling volatility, drawdown series, and asset correlation heatmap.",
    tone: "border-red-800/40",
    icon: "🛡️",
  },
];

export default function ChartsPage() {
  return (
    <RequireAuth>
      <div className="space-y-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-50">Charts & Visualizations</h1>
          <p className="mt-1 text-sm text-slate-400">
            Interactive ML outcome charts. PNG images are generated only when you enable Save PNG on a chart.
          </p>
        </div>

        <section className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {CHART_CARDS.map((card) => (
              <Link
                key={card.href}
                href={card.href}
                className={`group relative overflow-hidden rounded-xl border ${card.tone} bg-slate-900/40 p-5 text-slate-100 transition hover:bg-slate-900/55`}
              >
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-slate-500/40 to-transparent" />
                <div className="mb-3 text-2xl opacity-90">{card.icon}</div>
                <h2 className="mb-2 text-lg font-semibold text-slate-100">{card.title}</h2>
                <p className="text-sm leading-relaxed text-slate-300">{card.description}</p>
                <span className="absolute bottom-3 right-4 text-xs text-slate-400 group-hover:text-slate-200 transition-colors">
                  Open →
                </span>
              </Link>
            ))}
          </div>
        </section>
      </div>
    </RequireAuth>
  );
}
