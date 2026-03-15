"""
chart_engine.py – Comprehensive analytical chart generation for the Financial Advisor Bot.

Every public function:
  • Accepts pre-fetched price data as List[Dict] with keys {"t": "YYYY-MM-DD", "c": float,
    optionally "o","h","l","v"}
  • Generates a matplotlib PNG saved to    base_dir/visualizations/<chart_type>/<name>_<ts>.png
  • Returns a plain dict with all computed metrics / series the REST API returns as JSON
  • Never raises – all errors are caught and logged
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _charts_dir(base_dir: Path, chart_type: str) -> Path:
    p = base_dir / "visualizations" / chart_type
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mpl():
    """Lazily import matplotlib; returns (plt, patches) or (None, None)."""
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError:
        return None, None


def _closes(points: List[Dict[str, Any]]) -> np.ndarray:
    return np.array([float(p.get("c", 0) or 0) for p in points], dtype=float)


def _dates(points: List[Dict[str, Any]]) -> List[str]:
    return [str(p.get("t", "") or "") for p in points]


def _xtick_labels(labels: List[str], ax: Any, max_ticks: int = 10) -> None:
    n = len(labels)
    if n == 0:
        return
    step = max(1, n // max_ticks)
    idx = list(range(0, n, step))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] if i < n else "" for i in idx],
                       rotation=35, ha="right", fontsize=7)


# ---------------------------------------------------------------------------
# 1 – Backtest Performance Report
# ---------------------------------------------------------------------------

def run_backtest_chart(
    prices: List[Dict[str, Any]],
    symbol: str,
    benchmark_prices: List[Dict[str, Any]],
    benchmark_symbol: str,
    *,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.001,   # 0.1 %
    slippage_bps: float = 2.0,
    base_dir: Path,
    save_image: bool = False,
) -> Dict[str, Any]:
    """
    Buy-and-hold backtest for *symbol* vs *benchmark_symbol*.

    Returns a dict with summary metrics + array data for the frontend charts.
    Saves a 4-panel PNG to disk.
    """
    try:
        closes = _closes(prices)
        bench_closes = _closes(benchmark_prices)
        dates = _dates(prices)

        if len(closes) < 2:
            return {"error": "insufficient price data"}

        # ---------- Equity curves ----------
        # Simple buy-and-hold: buy on day 0, sell on last day
        n = len(closes)
        start_price = float(closes[0])
        end_price   = float(closes[-1])

        # Before friction
        raw_return = (end_price / start_price - 1.0) if start_price > 0 else 0.0
        strategy_equity = (closes / start_price) * initial_capital

        # Friction: one purchase + one sale
        num_trades = 2  # one buy, one sell
        shares = initial_capital / start_price
        buy_cost  = shares * start_price * (commission_rate + slippage_bps / 10_000)
        sell_cost = shares * end_price   * (commission_rate + slippage_bps / 10_000)
        total_friction = buy_cost + sell_cost
        avg_friction   = total_friction / num_trades
        friction_drag  = total_friction / initial_capital

        after_friction_return = raw_return - friction_drag
        after_friction_equity = strategy_equity - total_friction

        # Benchmark equity curve
        if len(bench_closes) >= 2:
            bs = float(bench_closes[0])
            bench_equity = (bench_closes[:n] / bs) * initial_capital if bs > 0 else np.full(n, initial_capital)
            bench_return = (float(bench_closes[min(n, len(bench_closes)) - 1]) / bs - 1.0) if bs > 0 else 0.0
        else:
            bench_equity  = np.full(n, initial_capital)
            bench_return  = 0.0

        alpha = after_friction_return - bench_return

        # ---------- Risk metrics ----------
        daily_returns = np.diff(closes) / closes[:-1]
        trading_days  = 252.0
        ann_vol = float(np.std(daily_returns, ddof=1) * math.sqrt(trading_days)) if len(daily_returns) > 1 else 0.0
        ann_ret = float(after_friction_return)  # already over period; simplified

        # Annualized Sharpe (risk-free = 0)
        daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 1e-9
        daily_mean = float(np.mean(daily_returns)) if len(daily_returns) > 0 else 0.0
        sharpe = (daily_mean / daily_std) * math.sqrt(trading_days) if daily_std > 1e-9 else 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(strategy_equity)
        drawdowns   = (strategy_equity - running_max) / np.where(running_max > 0, running_max, 1.0)
        max_drawdown = float(np.min(drawdowns))

        # ---------- Build time-series for the frontend ----------
        equity_series = [
            {"t": dates[i] if i < len(dates) else str(i),
             "strategy": round(float(strategy_equity[i]), 2),
             "strategy_after": round(float(after_friction_equity[i]), 2),
             "benchmark": round(float(bench_equity[i]) if i < len(bench_equity) else initial_capital, 2)}
            for i in range(n)
        ]
        drawdown_series = [
            {"t": dates[i] if i < len(dates) else str(i),
             "drawdown": round(float(drawdowns[i]) * 100, 4)}
            for i in range(n)
        ]

        summary = {
            "symbol":                  symbol,
            "benchmark_symbol":        benchmark_symbol,
            "period_start":            dates[0] if dates else "",
            "period_end":              dates[-1] if dates else "",
            "initial_capital":         initial_capital,
            "portfolio_return_before": round(raw_return * 100, 4),
            "portfolio_return_after":  round(after_friction_return * 100, 4),
            "benchmark_return":        round(bench_return * 100, 4),
            "alpha":                   round(alpha * 100, 4),
            "sharpe_ratio":            round(sharpe, 4),
            "max_drawdown_pct":        round(max_drawdown * 100, 4),
            "annualized_volatility":   round(ann_vol * 100, 4),
            "num_trades":              num_trades,
            "total_friction_cost":     round(total_friction, 2),
            "avg_friction_per_trade":  round(avg_friction, 2),
            "commission_rate_pct":     round(commission_rate * 100, 3),
            "slippage_bps":            slippage_bps,
            "friction_drag_pct":       round(friction_drag * 100, 4),
            "equity_series":           equity_series,
            "drawdown_series":         drawdown_series,
        }

        # ---------- Save PNG (opt-in) ----------
        if save_image:
            _save_backtest_png(summary, symbol, base_dir)

        return summary

    except Exception as exc:
        logger.warning("run_backtest_chart failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


def _save_backtest_png(s: Dict[str, Any], symbol: str, base_dir: Path) -> None:
    plt, mpatches = _mpl()
    if plt is None:
        return
    try:
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Backtest – Backtest Performance Report", fontsize=14, fontweight="bold", y=0.98)

        # ── Top-left: Returns Analysis (bar chart)
        ax1 = fig.add_subplot(2, 2, 1)
        bar_labels = ["Before\nFriction", "After\nFriction", "Benchmark", "Alpha"]
        bar_values = [
            s["portfolio_return_before"],
            s["portfolio_return_after"],
            s["benchmark_return"],
            s["alpha"],
        ]
        bar_colors = ["#4C9BE8", "#F5A623", "#2ECC71", "#E84C4C"]
        bars = ax1.bar(bar_labels, bar_values, color=bar_colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, bar_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax1.set_ylabel("Return (%)")
        ax1.set_title("Returns Analysis")
        ax1.grid(axis="y", alpha=0.25)

        # ── Top-right: Risk-Adjusted Performance (bar chart)
        ax2 = fig.add_subplot(2, 2, 2)
        ra_labels = ["Sharpe\nRatio", "Max\nDrawdown (%)"]
        ra_values = [s["sharpe_ratio"], abs(s["max_drawdown_pct"])]
        ra_colors = ["#2ECC71", "#E84C4C"]
        bars2 = ax2.bar(ra_labels, ra_values, color=ra_colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars2, ra_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Value")
        ax2.set_title("Risk-Adjusted Performance")
        ax2.grid(axis="y", alpha=0.25)

        # ── Bottom-left: Equity Curve
        ax3 = fig.add_subplot(2, 2, 3)
        eq = s.get("equity_series") or []
        if eq:
            xs = list(range(len(eq)))
            strat   = [p["strategy"] for p in eq]
            strat_a = [p["strategy_after"] for p in eq]
            bench   = [p["benchmark"] for p in eq]
            ax3.plot(xs, strat,   color="#4C9BE8", lw=1.5, label=f"{symbol} (before friction)")
            ax3.plot(xs, strat_a, color="#F5A623", lw=1.5, ls="--", label=f"{symbol} (after friction)")
            ax3.plot(xs, bench,   color="#2ECC71", lw=1.2, label=s["benchmark_symbol"])
            ax3.axhline(s["initial_capital"], color="gray", lw=0.8, ls=":", alpha=0.5)
            _xtick_labels([p["t"] for p in eq], ax3)
            ax3.set_ylabel(f"Portfolio Value ($)")
            ax3.set_title("Equity Curve")
            ax3.legend(fontsize=7, loc="upper left")
            ax3.grid(alpha=0.20)

        # ── Bottom-right: Drawdown
        ax4 = fig.add_subplot(2, 2, 4)
        dd = s.get("drawdown_series") or []
        if dd:
            xs = list(range(len(dd)))
            dvals = [p["drawdown"] for p in dd]
            ax4.fill_between(xs, dvals, 0, alpha=0.45, color="#E84C4C", label="Drawdown")
            ax4.plot(xs, dvals, color="#E84C4C", lw=1.0)
            ax4.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
            _xtick_labels([p["t"] for p in dd], ax4)
            ax4.set_ylabel("Drawdown (%)")
            ax4.set_title("Portfolio Drawdown")
            ax4.grid(alpha=0.20)

        # ── Text boxes for metrics summary
        friction_text = (
            f"Backtest – Trading Friction Metrics\n\n"
            f"Total Trades: {s['num_trades']}\n"
            f"Total Friction Cost: ${s['total_friction_cost']:,.2f}\n"
            f"Avg Friction/Trade: ${s['avg_friction_per_trade']:,.2f}\n"
            f"Commission Rate: {s['commission_rate_pct']:.3f}%\n"
            f"Slippage: {s['slippage_bps']:.1f} bps\n"
            f"Friction Drag: {s['friction_drag_pct']:.3f}%"
        )
        backtest_text = (
            f"Backtest – Backtest Summary\n\n"
            f"Period: {s['period_start']} to {s['period_end']}\n"
            f"Initial Capital: ${s['initial_capital']:,.0f}\n\n"
            f"Portfolio Return (Before): {s['portfolio_return_before']:.2f}%\n"
            f"Portfolio Return (After):  {s['portfolio_return_after']:.2f}%\n"
            f"Benchmark Return: {s['benchmark_return']:.2f}%\n"
            f"Alpha: {s['alpha']:.2f}%\n"
            f"Sharpe Ratio: {s['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {s['max_drawdown_pct']:.2f}%"
        )

        fig.text(0.05, 0.01, friction_text, fontsize=8.5,
                 bbox=dict(boxstyle="round", facecolor="#1A3A5C", alpha=0.85, edgecolor="#4C9BE8"),
                 color="white", verticalalignment="bottom", family="monospace")
        fig.text(0.52, 0.01, backtest_text, fontsize=8.5,
                 bbox=dict(boxstyle="round", facecolor="#1A3A5C", alpha=0.85, edgecolor="#4C9BE8"),
                 color="white", verticalalignment="bottom", family="monospace")

        fig.tight_layout(rect=[0, 0.20, 1, 0.97])

        out   = _charts_dir(base_dir, "backtest")
        fname = out / f"backtest_{symbol}_{_ts()}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("backtest chart saved → %s", fname)
    except Exception as exc:
        logger.debug("_save_backtest_png failed: %s", exc, exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 2 – Technical Indicators: RSI & MACD
# ---------------------------------------------------------------------------

def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Returns RSI array (same length as closes; first `period` entries = NaN)."""
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(len(closes), np.nan)
    if len(gains) < period:
        return rsi

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    for i in range(period, len(closes)):
        idx = i - 1  # delta index
        avg_gain = (avg_gain * (period - 1) + float(gains[idx])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[idx])) / period
        rs = avg_gain / avg_loss if avg_loss > 1e-10 else 1e10
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _compute_ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    ema = np.full_like(arr, np.nan)
    # Find first non-nan
    start = 0
    while start < len(arr) and np.isnan(arr[start]):
        start += 1
    if start >= len(arr):
        return ema
    ema[start] = arr[start]
    for i in range(start + 1, len(arr)):
        if np.isnan(arr[i]):
            ema[i] = ema[i - 1]
        else:
            ema[i] = alpha * arr[i] + (1.0 - alpha) * (ema[i - 1] if not np.isnan(ema[i - 1]) else arr[i])
    return ema


def run_technical_chart(
    prices: List[Dict[str, Any]],
    symbol: str,
    *,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    base_dir: Path,
    save_image: bool = False,
) -> Dict[str, Any]:
    """
    Compute RSI and MACD for *prices*, save a 3-panel chart PNG, and return arrays.
    """
    try:
        closes = _closes(prices)
        dates  = _dates(prices)
        n = len(closes)

        if n < macd_slow + macd_signal + 5:
            return {"error": "insufficient data for technical indicators"}

        rsi = _compute_rsi(closes, rsi_period)
        ema_fast = _compute_ema(closes, macd_fast)
        ema_slow = _compute_ema(closes, macd_slow)
        macd_line    = ema_fast - ema_slow
        signal_line  = _compute_ema(macd_line, macd_signal)
        histogram    = macd_line - signal_line

        def _safe(v: float) -> Optional[float]:
            return None if np.isnan(v) else round(float(v), 4)

        series = [
            {
                "t":         dates[i] if i < len(dates) else str(i),
                "close":     round(float(closes[i]), 4),
                "rsi":       _safe(rsi[i]),
                "macd":      _safe(macd_line[i]),
                "signal":    _safe(signal_line[i]),
                "histogram": _safe(histogram[i]),
            }
            for i in range(n)
        ]

        # Overbought/oversold counts
        valid_rsi = rsi[~np.isnan(rsi)]
        overbought = int(np.sum(valid_rsi > 70))
        oversold   = int(np.sum(valid_rsi < 30))

        result = {
            "symbol":     symbol,
            "rsi_period": rsi_period,
            "macd_fast":  macd_fast,
            "macd_slow":  macd_slow,
            "macd_signal_period": macd_signal,
            "current_rsi":    _safe(rsi[n - 1]),
            "current_macd":   _safe(macd_line[n - 1]),
            "current_signal": _safe(signal_line[n - 1]),
            "overbought_count": overbought,
            "oversold_count":   oversold,
            "series": series,
        }

        if save_image:
            _save_technical_png(result, closes, dates, rsi, macd_line, signal_line, histogram, symbol, base_dir)
        return result

    except Exception as exc:
        logger.warning("run_technical_chart failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


def _save_technical_png(
    result: Dict[str, Any],
    closes: np.ndarray,
    dates: List[str],
    rsi: np.ndarray,
    macd_line: np.ndarray,
    signal_line: np.ndarray,
    histogram: np.ndarray,
    symbol: str,
    base_dir: Path,
) -> None:
    plt, _ = _mpl()
    if plt is None:
        return
    try:
        n  = len(closes)
        xs = np.arange(n)

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.5, 1.5], "hspace": 0.35})

        # Panel 1 – Price with EMAs
        ax1 = axes[0]
        ax1.plot(xs, closes, color="#4C9BE8", lw=1.5, label="Close")
        ema20 = _compute_ema(closes, 20)
        ema50 = _compute_ema(closes, 50)
        ax1.plot(xs, ema20, color="#F5A623", lw=1.0, ls="--", label="EMA 20")
        ax1.plot(xs, ema50, color="#E84C4C", lw=1.0, ls="--", label="EMA 50")
        ax1.set_ylabel("Price ($)")
        ax1.set_title(f"{symbol} – Price with EMA 20 / EMA 50")
        ax1.legend(fontsize=8, loc="upper left")
        ax1.grid(alpha=0.20)

        # Panel 2 – RSI
        ax2 = axes[1]
        ax2.plot(xs, rsi, color="#9B59B6", lw=1.4, label=f"RSI ({result['rsi_period']})")
        ax2.axhline(70, color="#E84C4C", lw=0.9, ls="--", alpha=0.7, label="Overbought (70)")
        ax2.axhline(30, color="#2ECC71", lw=0.9, ls="--", alpha=0.7, label="Oversold (30)")
        ax2.axhline(50, color="gray",    lw=0.6, ls=":",  alpha=0.4)
        ax2.fill_between(xs, rsi, 70, where=rsi > 70, alpha=0.18, color="#E84C4C")
        ax2.fill_between(xs, rsi, 30, where=rsi < 30, alpha=0.18, color="#2ECC71")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI")
        ax2.set_title(f"RSI ({result['rsi_period']}) – Overbought: {result['overbought_count']}x   Oversold: {result['oversold_count']}x")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(alpha=0.20)

        # Panel 3 – MACD
        ax3 = axes[2]
        hist_colors = np.where(histogram >= 0, "#2ECC71", "#E84C4C")
        ax3.bar(xs, histogram, color=hist_colors, alpha=0.65, label="Histogram", width=0.8)
        ax3.plot(xs, macd_line,   color="#4C9BE8", lw=1.3, label=f"MACD ({result['macd_fast']},{result['macd_slow']})")
        ax3.plot(xs, signal_line, color="#F5A623", lw=1.0, ls="--", label=f"Signal ({result['macd_signal_period']})")
        ax3.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.35)
        ax3.set_ylabel("MACD")
        ax3.set_title(f"MACD ({result['macd_fast']},{result['macd_slow']},{result['macd_signal_period']})")
        ax3.legend(fontsize=8, loc="upper left")
        ax3.grid(alpha=0.20)
        _xtick_labels(dates, ax3)

        out   = _charts_dir(base_dir, "technical")
        fname = out / f"technical_{symbol}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("technical chart saved → %s", fname)
    except Exception as exc:
        logger.debug("_save_technical_png failed: %s", exc, exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 3 – Monte Carlo Simulation + Percentile Breakdown
# ---------------------------------------------------------------------------

def run_montecarlo_chart(
    prices: List[Dict[str, Any]],
    symbol: str,
    *,
    n_simulations: int = 500,
    horizon_days: int = 252,
    percentiles: Tuple[int, ...] = (5, 25, 50, 75, 95),
    base_dir: Path,
    save_image: bool = False,
) -> Dict[str, Any]:
    """
    Run Geometric Brownian Motion Monte Carlo simulation for *symbol*.
    Returns percentile fan data + summary statistics.
    """
    try:
        closes = _closes(prices)
        n_hist = len(closes)

        if n_hist < 20:
            return {"error": "insufficient historical data for Monte Carlo"}

        log_rets  = np.diff(np.log(closes[closes > 0]))
        mu        = float(np.mean(log_rets))
        sigma     = float(np.std(log_rets, ddof=1))
        last_price = float(closes[-1])

        rng = np.random.default_rng(seed=42)
        # Simulate paths: shape (n_simulations, horizon_days + 1)
        dt = 1.0
        drift = (mu - 0.5 * sigma ** 2) * dt
        vol   = sigma * math.sqrt(dt)

        Z = rng.standard_normal((n_simulations, horizon_days))
        # Price paths via cumulative product
        log_paths = np.cumsum(drift + vol * Z, axis=1)   # (n_sim, horizon)
        log_paths = np.hstack([np.zeros((n_simulations, 1)), log_paths])  # add t=0
        paths = last_price * np.exp(log_paths)  # (n_sim, horizon+1)

        # Compute percentile paths
        pct_paths: Dict[str, List[float]] = {}
        for p in percentiles:
            pct_paths[str(p)] = [round(float(v), 4) for v in np.percentile(paths, p, axis=0)]

        final_prices = paths[:, -1]
        expected_price   = float(np.mean(final_prices))
        expected_return  = (expected_price / last_price - 1.0) * 100.0
        prob_profit      = float(np.mean(final_prices > last_price)) * 100.0
        downside_10_pct  = float(np.percentile(final_prices, 10))
        upside_90_pct    = float(np.percentile(final_prices, 90))

        # Distribution histogram buckets (20 bins)
        hist_counts, hist_edges = np.histogram(final_prices, bins=20)
        distribution = [
            {"bucket_low": round(float(hist_edges[i]), 2),
             "bucket_high": round(float(hist_edges[i + 1]), 2),
             "count": int(hist_counts[i])}
            for i in range(len(hist_counts))
        ]

        result = {
            "symbol":           symbol,
            "n_simulations":    n_simulations,
            "horizon_days":     horizon_days,
            "last_price":       round(last_price, 4),
            "mu_daily":         round(mu, 6),
            "sigma_daily":      round(sigma, 6),
            "expected_price":   round(expected_price, 4),
            "expected_return_pct": round(expected_return, 4),
            "prob_profit_pct":  round(prob_profit, 2),
            "p10_price":        round(downside_10_pct, 4),
            "p90_price":        round(upside_90_pct, 4),
            "percentile_paths": pct_paths,  # keys: "5","25","50","75","95"
            "distribution":     distribution,
        }

        if save_image:
            _save_montecarlo_png(result, paths, percentiles, symbol, base_dir)
        return result

    except Exception as exc:
        logger.warning("run_montecarlo_chart failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


def _save_montecarlo_png(
    result: Dict[str, Any],
    paths: np.ndarray,
    percentiles: Tuple[int, ...],
    symbol: str,
    base_dir: Path,
) -> None:
    plt, _ = _mpl()
    if plt is None:
        return
    try:
        horizon = paths.shape[1] - 1
        xs      = np.arange(horizon + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"Monte Carlo Simulation – {symbol}  ({result['n_simulations']:,} paths, "
                     f"{result['horizon_days']} trading days)", fontsize=13, fontweight="bold")

        # Panel 1 – Fan chart
        pct_data = result["percentile_paths"]
        fan_pairs = [(5, 95), (25, 75)]
        fan_alphas = [0.18, 0.30]
        fan_colors = ["#4C9BE8", "#4C9BE8"]
        for (lo, hi), alpha, color in zip(fan_pairs, fan_alphas, fan_colors):
            lo_path = np.array(pct_data[str(lo)])
            hi_path = np.array(pct_data[str(hi)])
            ax1.fill_between(xs, lo_path, hi_path, alpha=alpha, color=color,
                             label=f"P{lo}–P{hi} range")

        for p in percentiles:
            path    = np.array(pct_data[str(p)])
            is_med  = p == 50
            ax1.plot(xs, path,
                     lw=2.2 if is_med else 1.0,
                     color="#FFD700" if is_med else "#4C9BE8",
                     ls="-" if is_med else "--",
                     label=f"P{p}  ${path[-1]:,.0f}")

        ax1.axhline(result["last_price"], color="white", lw=0.8, ls=":", alpha=0.5, label="Current price")
        ax1.set_xlabel("Trading Days Forward")
        ax1.set_ylabel(f"Simulated Price ($)")
        ax1.set_title("Percentile Fan Chart")
        ax1.legend(fontsize=8, loc="upper left", framealpha=0.85)
        ax1.grid(alpha=0.18)

        # Annotation box
        ax1.text(0.98, 0.05,
                 f"Expected return: {result['expected_return_pct']:+.1f}%\n"
                 f"P(profit): {result['prob_profit_pct']:.1f}%\n"
                 f"P10 price: ${result['p10_price']:,.2f}\n"
                 f"P90 price: ${result['p90_price']:,.2f}",
                 transform=ax1.transAxes, fontsize=8.5,
                 verticalalignment="bottom", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="#1A1A2E", alpha=0.85, edgecolor="#4C9BE8"),
                 color="white", family="monospace")

        # Panel 2 – Final-price distribution histogram
        dist      = result["distribution"]
        midpoints = [(d["bucket_low"] + d["bucket_high"]) / 2 for d in dist]
        counts    = [d["count"] for d in dist]
        bar_colors = ["#2ECC71" if m >= result["last_price"] else "#E84C4C" for m in midpoints]
        ax2.bar(midpoints, counts,
                width=(midpoints[-1] - midpoints[0]) / max(1, len(midpoints) - 1) * 0.9,
                color=bar_colors, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax2.axvline(result["last_price"],     color="white", lw=1.2, ls="--", label="Current price")
        ax2.axvline(result["expected_price"], color="#FFD700", lw=1.2, ls="-", label="Expected price")
        ax2.set_xlabel(f"Price at Day {horizon} ($)")
        ax2.set_ylabel("Simulation Count")
        ax2.set_title("Final Price Distribution")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.20)

        # Percentile annotations
        for p in percentiles:
            val = float(np.array(result["percentile_paths"][str(p)])[-1])
            ax2.axvline(val, color="#4C9BE8", lw=0.7, ls=":", alpha=0.55)
            ax2.text(val, max(counts) * 0.95, f"P{p}", color="#4C9BE8", fontsize=7,
                     ha="center", va="top", rotation=90, alpha=0.8)

        fig.tight_layout()
        out   = _charts_dir(base_dir, "montecarlo")
        fname = out / f"montecarlo_{symbol}_{_ts()}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("montecarlo chart saved → %s", fname)
    except Exception as exc:
        logger.debug("_save_montecarlo_png failed: %s", exc, exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 4 – Valuation Confidence Intervals (Bollinger Bands + historical range)
# ---------------------------------------------------------------------------

def run_valuation_confidence_chart(
    prices: List[Dict[str, Any]],
    symbol: str,
    *,
    bb_period: int = 20,
    n_std: float = 2.0,
    base_dir: Path,
    save_image: bool = False,
) -> Dict[str, Any]:
    """
    Rolling mean ± 1σ and ± 2σ Bollinger confidence bands.
    Also adds a 52-week high/low channel and a linear regression trend.
    """
    try:
        closes = _closes(prices)
        dates  = _dates(prices)
        n = len(closes)

        if n < bb_period + 5:
            return {"error": "insufficient data for valuation confidence intervals"}

        period = bb_period
        sma    = np.full(n, np.nan)
        upper2 = np.full(n, np.nan)
        lower2 = np.full(n, np.nan)
        upper1 = np.full(n, np.nan)
        lower1 = np.full(n, np.nan)

        for i in range(period - 1, n):
            window = closes[i - period + 1 : i + 1]
            m = float(np.mean(window))
            s = float(np.std(window, ddof=1))
            sma[i]    = m
            upper2[i] = m + n_std * s
            lower2[i] = m - n_std * s
            upper1[i] = m + 1.0 * s
            lower1[i] = m - 1.0 * s

        # 52-week rolling high/low
        win52 = min(252, n)
        high52 = np.array([float(np.max(closes[max(0, i - win52 + 1): i + 1])) for i in range(n)])
        low52  = np.array([float(np.min(closes[max(0, i - win52 + 1): i + 1])) for i in range(n)])

        # Linear regression trend
        xs_reg = np.arange(n, dtype=float)
        if n >= 2:
            _coeffs = np.polyfit(xs_reg, closes, 1)
            slope, intercept = float(_coeffs[0]), float(_coeffs[1])
        else:
            slope, intercept = 0.0, float(closes[0])
        trend = intercept + slope * xs_reg

        def _safe(v: float) -> Optional[float]:
            return None if np.isnan(v) else round(float(v), 4)

        series = [
            {
                "t":       dates[i] if i < len(dates) else str(i),
                "close":   round(float(closes[i]), 4),
                "sma":     _safe(sma[i]),
                "upper2":  _safe(upper2[i]),
                "lower2":  _safe(lower2[i]),
                "upper1":  _safe(upper1[i]),
                "lower1":  _safe(lower1[i]),
                "high52":  round(float(high52[i]), 4),
                "low52":   round(float(low52[i]), 4),
                "trend":   round(float(trend[i]), 4),
            }
            for i in range(n)
        ]

        last = float(closes[-1])
        last_sma    = _safe(sma[-1])
        last_upper2 = _safe(upper2[-1])
        last_lower2 = _safe(lower2[-1])

        pct_b = None
        if last_upper2 is not None and last_lower2 is not None and (last_upper2 - last_lower2) > 0:
            pct_b = round((last - last_lower2) / (last_upper2 - last_lower2), 4)

        result = {
            "symbol":       symbol,
            "bb_period":    period,
            "n_std":        n_std,
            "last_price":   round(last, 4),
            "last_sma":     last_sma,
            "last_upper2":  last_upper2,
            "last_lower2":  last_lower2,
            "pct_b":        pct_b,  # 0=at lower band, 1=at upper band
            "trend_slope":  round(slope, 6),
            "series":       series,
        }

        if save_image:
            _save_valuation_confidence_png(result, closes, dates, sma, upper2, lower2,
                                           upper1, lower1, high52, low52, trend, symbol, base_dir)
        return result

    except Exception as exc:
        logger.warning("run_valuation_confidence_chart failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


def _save_valuation_confidence_png(
    result: Dict[str, Any],
    closes: np.ndarray,
    dates: List[str],
    sma: np.ndarray,
    upper2: np.ndarray,
    lower2: np.ndarray,
    upper1: np.ndarray,
    lower1: np.ndarray,
    high52: np.ndarray,
    low52: np.ndarray,
    trend: np.ndarray,
    symbol: str,
    base_dir: Path,
) -> None:
    plt, _ = _mpl()
    if plt is None:
        return
    try:
        n  = len(closes)
        xs = np.arange(n)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11),
                                        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35})
        fig.suptitle(f"Valuation Confidence Intervals – {symbol}", fontsize=13, fontweight="bold")

        # ── Top panel ──
        ax1.fill_between(xs, upper2, lower2, alpha=0.15, color="#4C9BE8", label="±2σ band")
        ax1.fill_between(xs, upper1, lower1, alpha=0.25, color="#4C9BE8", label="±1σ band")
        ax1.plot(xs, closes,  color="#E8E8FF", lw=1.4, label="Close",   zorder=4)
        ax1.plot(xs, sma,     color="#F5A623", lw=1.2, ls="--", label=f"SMA {result['bb_period']}")
        ax1.plot(xs, upper2,  color="#E84C4C", lw=0.9, ls="--", alpha=0.65, label="Upper 2σ")
        ax1.plot(xs, lower2,  color="#2ECC71", lw=0.9, ls="--", alpha=0.65, label="Lower 2σ")
        ax1.plot(xs, trend,   color="#FFD700", lw=1.1, ls=":", alpha=0.70, label="Linear trend")
        ax1.plot(xs, high52,  color="#E84C4C", lw=0.7, ls=":", alpha=0.35, label="52w High")
        ax1.plot(xs, low52,   color="#2ECC71", lw=0.7, ls=":", alpha=0.35, label="52w Low")
        ax1.set_ylabel("Price ($)")
        ax1.set_title(f"Bollinger Bands ({result['bb_period']} period, ±{result['n_std']}σ)  |  "
                      f"Current: ${result['last_price']:,.2f}  "
                      f"SMA: ${result['last_sma']:,.2f if result['last_sma'] else '–'}  "
                      f"%B: {result['pct_b']:.2f if result['pct_b'] is not None else '–'}")
        ax1.legend(fontsize=7, loc="upper left", framealpha=0.88, ncol=2)
        ax1.grid(alpha=0.18)
        _xtick_labels(dates, ax1)

        # ── Bottom panel: %B oscillator ──
        pct_b_arr = (closes - lower2) / np.where((upper2 - lower2) > 0, upper2 - lower2, 1.0)
        ax2.fill_between(xs, pct_b_arr, 0.5, where=pct_b_arr > 0.5,
                         alpha=0.30, color="#E84C4C", label="Above midband")
        ax2.fill_between(xs, pct_b_arr, 0.5, where=pct_b_arr <= 0.5,
                         alpha=0.30, color="#2ECC71", label="Below midband")
        ax2.plot(xs, pct_b_arr, color="#4C9BE8", lw=1.2)
        ax2.axhline(1.0, color="#E84C4C", lw=0.8, ls="--", alpha=0.55)
        ax2.axhline(0.5, color="gray",    lw=0.6, ls=":",  alpha=0.35)
        ax2.axhline(0.0, color="#2ECC71", lw=0.8, ls="--", alpha=0.55)
        ax2.set_ylabel("%B")
        ax2.set_title("%B Oscillator  (1.0 = price at upper band,  0.0 = price at lower band)")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(alpha=0.18)
        _xtick_labels(dates, ax2)

        out   = _charts_dir(base_dir, "valuation_confidence")
        fname = out / f"valuation_confidence_{symbol}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("valuation_confidence chart saved → %s", fname)
    except Exception as exc:
        logger.debug("_save_valuation_confidence_png failed: %s", exc, exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 5 – Historical Price vs Predicted Price
# ---------------------------------------------------------------------------

def run_price_vs_predicted_chart(
    prices: List[Dict[str, Any]],
    symbol: str,
    predictions: List[Dict[str, Any]],
    *,
    base_dir: Path,
    save_image: bool = False,
) -> Dict[str, Any]:
    """
    Overlay historical close with ML prediction probabilities.

    *predictions* is a list of dicts {"t": "YYYY-MM-DD", "prob_up": float}.
    If *predictions* is empty, a momentum-based pseudo-signal is generated.
    """
    try:
        closes = _closes(prices)
        dates  = _dates(prices)
        n = len(closes)

        if n < 10:
            return {"error": "insufficient price data"}

        # Build prob_up series aligned to prices
        pred_map: Dict[str, float] = {}
        for p in (predictions or []):
            t = str(p.get("t") or "")
            v = p.get("prob_up") or p.get("probability_up")
            if t and v is not None:
                try:
                    pred_map[t] = float(v)
                except Exception:
                    pass

        if pred_map:
            probs = np.array([pred_map.get(dates[i], np.nan) for i in range(n)])
        else:
            # Fallback: 5-day momentum signal re-scaled to 0–1
            mom = np.full(n, np.nan)
            for i in range(5, n):
                if closes[i - 5] > 0:
                    m = (closes[i] - closes[i - 5]) / closes[i - 5]
                    mom[i] = float(1.0 / (1.0 + math.exp(-m * 10)))  # logistic normalise
            probs = mom

        # Predicted price = previous close × expected factor based on prob
        # factor = 1 + (prob - 0.5) * daily_move  where daily_move ≈ 1 %
        daily_move = 0.01
        pred_price = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(probs[i]):
                factor = 1.0 + (float(probs[i]) - 0.5) * 2.0 * daily_move
                pred_price[i] = float(closes[i - 1]) * factor

        # Signal labels: buy when prob crosses above 0.55, sell below 0.45
        buy_signals: List[Dict[str, Any]]  = []
        sell_signals: List[Dict[str, Any]] = []
        for i in range(1, n):
            if np.isnan(probs[i]) or np.isnan(probs[i - 1]):
                continue
            if probs[i - 1] <= 0.55 and probs[i] > 0.55:
                buy_signals.append({"t": dates[i] if i < len(dates) else str(i),
                                    "price": round(float(closes[i]), 4), "index": i})
            elif probs[i - 1] >= 0.45 and probs[i] < 0.45:
                sell_signals.append({"t": dates[i] if i < len(dates) else str(i),
                                     "price": round(float(closes[i]), 4), "index": i})

        def _safe(v: float) -> Optional[float]:
            return None if np.isnan(v) else round(float(v), 4)

        series = [
            {
                "t":           dates[i] if i < len(dates) else str(i),
                "close":       round(float(closes[i]), 4),
                "predicted":   _safe(pred_price[i]),
                "prob_up":     _safe(probs[i]),
            }
            for i in range(n)
        ]

        # Merge buy+sell with a "signal" field for frontend convenience
        combined_signals = (
            [{"t": s["t"], "price": s["price"], "index": s["index"], "signal": "buy",  "prob_up": 1.0} for s in buy_signals]
            + [{"t": s["t"], "price": s["price"], "index": s["index"], "signal": "sell", "prob_up": 0.0} for s in sell_signals]
        )
        combined_signals.sort(key=lambda x: x["index"])

        valid_probs = [p for p in probs.tolist() if not (isinstance(p, float) and p != p)]  # filter NaN
        avg_prob_up = round(float(np.nanmean(probs)) if len(valid_probs) > 0 else 0.5, 4)

        result = {
            "symbol":        symbol,
            "n_points":      n,
            "buy_signals":   buy_signals[:50],
            "sell_signals":  sell_signals[:50],
            "signals":       combined_signals[:100],
            "avg_prob_up":   avg_prob_up,
            "series":        series,
        }

        if save_image:
            _save_price_vs_predicted_png(result, closes, dates, pred_price, probs, symbol, base_dir)
        return result

    except Exception as exc:
        logger.warning("run_price_vs_predicted_chart failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


def _save_price_vs_predicted_png(
    result: Dict[str, Any],
    closes: np.ndarray,
    dates: List[str],
    pred_price: np.ndarray,
    probs: np.ndarray,
    symbol: str,
    base_dir: Path,
) -> None:
    plt, _ = _mpl()
    if plt is None:
        return
    try:
        n  = len(closes)
        xs = np.arange(n)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11),
                                        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.35})
        fig.suptitle(f"Historical Price vs ML Predicted Price – {symbol}", fontsize=13, fontweight="bold")

        # Price panel
        ax1.plot(xs, closes,     color="#4C9BE8", lw=1.5, label="Actual Close",    zorder=4)
        ax1.plot(xs, pred_price, color="#F5A623", lw=1.2, ls="--", label="ML Predicted", alpha=0.85, zorder=3)

        # Buy / sell markers
        buy_idx  = [b["index"] for b in result["buy_signals"]]
        sell_idx = [s["index"] for s in result["sell_signals"]]
        if buy_idx:
            ax1.scatter(buy_idx, closes[buy_idx], color="#2ECC71", marker="^", s=55, zorder=5,
                        label=f"Buy signal ({len(buy_idx)})")
        if sell_idx:
            ax1.scatter(sell_idx, closes[sell_idx], color="#E84C4C", marker="v", s=55, zorder=5,
                        label=f"Sell signal ({len(sell_idx)})")

        ax1.set_ylabel("Price ($)")
        ax1.set_title("Actual vs ML Predicted Price with Entry/Exit Signals")
        ax1.legend(fontsize=8, loc="upper left")
        ax1.grid(alpha=0.18)
        _xtick_labels(dates, ax1)

        # Probability panel
        ax2.fill_between(xs, probs, 0.5, where=probs > 0.5, alpha=0.35, color="#2ECC71", label="Bullish (>50%)")
        ax2.fill_between(xs, probs, 0.5, where=probs <= 0.5, alpha=0.35, color="#E84C4C", label="Bearish (<50%)")
        ax2.plot(xs, probs, color="#4C9BE8", lw=1.2)
        ax2.axhline(0.5,  color="gray",    lw=0.8, ls="--", alpha=0.45)
        ax2.axhline(0.55, color="#2ECC71", lw=0.7, ls=":",  alpha=0.55, label="Buy threshold (0.55)")
        ax2.axhline(0.45, color="#E84C4C", lw=0.7, ls=":",  alpha=0.55, label="Sell threshold (0.45)")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("P(Up)")
        ax2.set_title("ML Up-Probability Signal")
        ax2.legend(fontsize=8, loc="upper left", ncol=2)
        ax2.grid(alpha=0.18)
        _xtick_labels(dates, ax2)

        out   = _charts_dir(base_dir, "price_vs_predicted")
        fname = out / f"price_vs_predicted_{symbol}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("price_vs_predicted chart saved → %s", fname)
    except Exception as exc:
        logger.debug("_save_price_vs_predicted_png failed: %s", exc, exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 6 – Risk Dashboard
# ---------------------------------------------------------------------------

def run_risk_dashboard_chart(
    portfolio: Dict[str, Any],
    price_histories: Dict[str, List[Dict[str, Any]]],
    *,
    base_dir: Path,
    var_confidence: float = 0.95,
    save_image: bool = False,
) -> Dict[str, Any]:
    """
    Portfolio risk dashboard:
    - VaR (historical simulation) at *var_confidence*
    - Conditional VaR (CVaR / Expected Shortfall)
    - Rolling 20-day volatility per holding
    - Portfolio-level drawdown
    - Asset correlation matrix (if ≥ 2 holdings)
    """
    try:
        holdings = portfolio.get("holdings") or []
        if not holdings:
            return {"error": "no holdings in portfolio"}

        # Build returns matrix
        returns_by_sym: Dict[str, np.ndarray] = {}
        weights_by_sym: Dict[str, float] = {}
        total_value = 0.0

        for h in holdings:
            sym = str(h.get("symbol", "") or "").upper()
            qty = float(h.get("quantity", 0) or 0)
            avg = float(h.get("avg_price", 0) or 0)
            pts = price_histories.get(sym) or []
            if not pts or qty <= 0 or avg <= 0:
                continue
            closes = _closes(pts)
            if len(closes) < 2:
                continue
            market_val = qty * float(closes[-1]) if closes[-1] > 0 else qty * avg
            total_value += market_val
            rets = np.diff(closes) / closes[:-1]
            returns_by_sym[sym] = rets
            weights_by_sym[sym] = market_val

        if not returns_by_sym:
            return {"error": "no valid holdings with price data"}

        # Normalise weights
        for sym in weights_by_sym:
            weights_by_sym[sym] = weights_by_sym[sym] / total_value if total_value > 0 else 1.0 / len(returns_by_sym)

        # Align return arrays to same length
        min_len = min(len(r) for r in returns_by_sym.values())
        syms    = sorted(returns_by_sym.keys())
        ret_matrix = np.array([returns_by_sym[s][-min_len:] for s in syms])  # (n_assets, T)
        weights    = np.array([weights_by_sym[s] for s in syms])

        # Portfolio daily returns
        port_returns = weights @ ret_matrix  # (T,)

        # VaR & CVaR
        var_pct   = 1.0 - var_confidence
        var_val   = float(np.percentile(port_returns, var_pct * 100))
        cvar_val  = float(np.mean(port_returns[port_returns <= var_val]))
        var_dollar  = abs(var_val) * total_value
        cvar_dollar = abs(cvar_val) * total_value

        # Portfolio drawdown
        equity = np.cumprod(1.0 + port_returns) * total_value
        peak   = np.maximum.accumulate(equity)
        dd     = (equity - peak) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.min(dd))

        # Rolling 20-day portfolio vol
        roll_vol = np.full(len(port_returns), np.nan)
        for i in range(20, len(port_returns)):
            roll_vol[i] = float(np.std(port_returns[i - 20: i], ddof=1)) * math.sqrt(252) * 100

        # Per-asset annualized volatility
        asset_vols: Dict[str, float] = {}
        for s in syms:
            r = returns_by_sym[s]
            asset_vols[s] = round(float(np.std(r, ddof=1)) * math.sqrt(252) * 100, 4) if len(r) > 1 else 0.0

        # Correlation matrix
        corr_matrix: Optional[List[List[float]]] = None
        if len(syms) >= 2:
            C = np.corrcoef(ret_matrix)
            corr_matrix = [[round(float(C[i, j]), 4) for j in range(len(syms))] for i in range(len(syms))]

        result = {
            "portfolio_value":  round(total_value, 2),
            "var_confidence":   var_confidence,
            "var_pct_daily":    round(var_val * 100, 4),
            "var_dollar":       round(var_dollar, 2),
            "cvar_pct_daily":   round(cvar_val * 100, 4),
            "cvar_dollar":      round(cvar_dollar, 2),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "asset_volatilities": asset_vols,
            "symbols":          syms,
            "correlation_matrix": corr_matrix,
            "rolling_vol_series": [
                {"index": i, "vol": round(float(roll_vol[i]), 4) if not np.isnan(roll_vol[i]) else None}
                for i in range(len(roll_vol))
            ],
            "drawdown_series": [
                {"index": i, "drawdown": round(float(dd[i]) * 100, 4)}
                for i in range(len(dd))
            ],
        }

        if save_image:
            _save_risk_dashboard_png(result, port_returns, dd, roll_vol, syms, ret_matrix, base_dir)
        return result

    except Exception as exc:
        logger.warning("run_risk_dashboard_chart failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


def _save_risk_dashboard_png(
    result: Dict[str, Any],
    port_returns: np.ndarray,
    dd: np.ndarray,
    roll_vol: np.ndarray,
    syms: List[str],
    ret_matrix: np.ndarray,
    base_dir: Path,
) -> None:
    plt, _ = _mpl()
    if plt is None:
        return
    try:
        n_assets = len(syms)
        has_corr  = n_assets >= 2

        fig = plt.figure(figsize=(18, 13))
        fig.suptitle("Portfolio Risk Dashboard", fontsize=14, fontweight="bold")

        gs_rows, gs_cols = (3, 3) if has_corr else (2, 2)
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(gs_rows, gs_cols, figure=fig, hspace=0.45, wspace=0.35)

        # Panel 1 – Return distribution with VaR/CVaR
        ax1 = fig.add_subplot(gs[0, :2])
        hist_counts, hist_edges = np.histogram(port_returns * 100, bins=40)
        mids = (hist_edges[:-1] + hist_edges[1:]) / 2
        bar_c = ["#E84C4C" if m < result["var_pct_daily"] else "#2ECC71" for m in mids]
        ax1.bar(mids, hist_counts, width=(hist_edges[1] - hist_edges[0]) * 0.9,
                color=bar_c, alpha=0.82, edgecolor="none")
        ax1.axvline(result["var_pct_daily"],  color="#E84C4C", lw=1.8, ls="--",
                    label=f"VaR {result['var_confidence']*100:.0f}%: {result['var_pct_daily']:.2f}%")
        ax1.axvline(result["cvar_pct_daily"], color="#FF8C00", lw=1.8, ls="-.",
                    label=f"CVaR: {result['cvar_pct_daily']:.2f}%")
        ax1.set_xlabel("Daily Return (%)")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Return Distribution  |  VaR(${result['var_dollar']:,.0f})  CVaR(${result['cvar_dollar']:,.0f})")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.22)

        # Panel 2 – Asset volatility bar
        ax2 = fig.add_subplot(gs[0, 2])
        vol_syms = sorted(result["asset_volatilities"].keys())
        vol_vals = [result["asset_volatilities"][s] for s in vol_syms]
        colors_v = [f"#{int(v*4):02x}6C88" if v < 25 else "#E84C4C" for v in vol_vals]
        ax2.barh(vol_syms, vol_vals, color="#4C9BE8", edgecolor="white", linewidth=0.4)
        for s, v in zip(vol_syms, vol_vals):
            ax2.text(v + 0.3, vol_syms.index(s), f"{v:.1f}%", va="center", fontsize=8)
        ax2.set_xlabel("Ann. Volatility (%)")
        ax2.set_title("Asset Volatilities")
        ax2.grid(axis="x", alpha=0.22)

        # Panel 3 – Rolling portfolio volatility
        ax3 = fig.add_subplot(gs[1, :2])
        rv_x = [p["index"] for p in result["rolling_vol_series"] if p["vol"] is not None]
        rv_y = [p["vol"] for p in result["rolling_vol_series"] if p["vol"] is not None]
        ax3.plot(rv_x, rv_y, color="#F5A623", lw=1.5)
        ax3.fill_between(rv_x, rv_y, alpha=0.20, color="#F5A623")
        ax3.set_ylabel("Ann. Vol (%)")
        ax3.set_title("Rolling 20-Day Portfolio Volatility (Annualised)")
        ax3.grid(alpha=0.20)

        # Panel 4 – Drawdown
        ax4 = fig.add_subplot(gs[1, 2])
        dd_x = [p["index"] for p in result["drawdown_series"]]
        dd_y = [p["drawdown"] for p in result["drawdown_series"]]
        ax4.fill_between(dd_x, dd_y, 0, alpha=0.40, color="#E84C4C")
        ax4.plot(dd_x, dd_y, color="#E84C4C", lw=1.0)
        ax4.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.4)
        ax4.set_ylabel("Drawdown (%)")
        ax4.set_title(f"Portfolio Drawdown  |  Max: {result['max_drawdown_pct']:.2f}%")
        ax4.grid(alpha=0.20)

        # Panel 5 – Correlation heatmap (if ≥ 2 assets)
        if has_corr and result["correlation_matrix"] is not None:
            ax5 = fig.add_subplot(gs[2, :])
            corr = np.array(result["correlation_matrix"])
            im = ax5.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax5.set_xticks(range(n_assets))
            ax5.set_yticks(range(n_assets))
            ax5.set_xticklabels(syms, fontsize=8)
            ax5.set_yticklabels(syms, fontsize=8)
            for i in range(n_assets):
                for j in range(n_assets):
                    ax5.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                             fontsize=7, color="black" if abs(corr[i, j]) < 0.5 else "white")
            fig.colorbar(im, ax=ax5, fraction=0.02)
            ax5.set_title("Asset Correlation Matrix")

        out   = _charts_dir(base_dir, "risk_dashboard")
        fname = out / f"risk_dashboard_{_ts()}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("risk_dashboard chart saved → %s", fname)
    except Exception as exc:
        logger.debug("_save_risk_dashboard_png failed: %s", exc, exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass
