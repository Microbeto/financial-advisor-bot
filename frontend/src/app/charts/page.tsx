"use client";

import Link from "next/link";
import { RequireAuth } from "@/components/require-auth";

const CHART_CARDS = [
  {
    href: "/charts/backtest",
    title: "Backtest Performance",
    description:
      "Return analysis, risk-adjusted performance (Sharpe, max drawdown), trading friction metrics and equity curves versus a benchmark.",
    color: "from-sky-700 to-sky-900",
    icon: "📈",
  },
  {
    href: "/charts/technical",
    title: "RSI & MACD Indicators",
    description:
      "Relative Strength Index and Moving Average Convergence/Divergence with overbought/oversold signals for any symbol.",
    color: "from-violet-700 to-violet-900",
    icon: "🔬",
  },
  {
    href: "/charts/montecarlo",
    title: "Monte Carlo Simulation",
    description:
      "Geometric Brownian Motion simulation with percentile fan chart (P5–P95), final-price distribution, and probability of profit.",
    color: "from-emerald-700 to-emerald-900",
    icon: "🎲",
  },
  {
    href: "/charts/valuation",
    title: "Valuation Confidence Intervals",
    description:
      "Bollinger bands (±1σ / ±2σ), %B oscillator, 52-week high/low channel, and linear regression price trend.",
    color: "from-amber-700 to-amber-900",
    icon: "📊",
  },
  {
    href: "/charts/price-vs-predicted",
    title: "Price vs Predicted",
    description:
      "Historical close price overlaid with ML up-probability. Crossover buy/sell signals marked directly on the chart.",
    color: "from-rose-700 to-rose-900",
    icon: "🤖",
  },
  {
    href: "/charts/risk",
    title: "Risk Dashboard",
    description:
      "Portfolio VaR (95%), CVaR, rolling volatility, drawdown series, and asset correlation heatmap.",
    color: "from-red-700 to-red-900",
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
            Interactive analysis charts. Each chart also saves a high-resolution PNG to the server&apos;s model store.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {CHART_CARDS.map((card) => (
            <Link
              key={card.href}
              href={card.href}
              className={`group relative overflow-hidden rounded-xl bg-gradient-to-br ${card.color} border border-white/10 p-6 shadow-lg transition-transform hover:-translate-y-1 hover:shadow-xl`}
            >
              <div className="mb-3 text-3xl">{card.icon}</div>
              <h2 className="mb-2 text-lg font-semibold text-white">{card.title}</h2>
              <p className="text-sm text-white/75 leading-relaxed">{card.description}</p>
              <span className="absolute bottom-4 right-4 text-xs text-white/50 group-hover:text-white/80 transition-colors">
                Open →
              </span>
            </Link>
          ))}
        </div>
      </div>
    </RequireAuth>
  );
}
