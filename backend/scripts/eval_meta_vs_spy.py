# Evaluation script comparing meta-combiner model performance against S&P 500 benchmark.
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ml.meta_labeler import MetaLabelerCombiner
from app.ml.regime_switcher import RegimeSwitcherCombiner
from app.ml.rl_agent import RLAgentCombiner
from app.ml.tournament import run_walk_forward_tournament
from app.services import ml_workflow as mw


def annualized_sharpe(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float((np.mean(arr) / std) * np.sqrt(252.0))


def max_drawdown(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0.0, (equity / peaks) - 1.0, 0.0)
    return float(np.min(drawdowns))


def win_rate(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr > 0.0))


async def main() -> None:
    mw._daily_news_features = lambda d, symbols: mw._DailyNewsFeatures(
        symbol_sentiment={}, market_sentiment=0.0, macro_features=[0.0] * 8, news_item_count=1
    )
    mw.get_ml_runtime_settings = lambda: {
        "tp_barrier": 0.04,
        "sl_barrier": 0.03,
        "time_barrier_days": 5,
        "walk_forward_splits": 5,
        "walk_forward_min_train": 120,
    }

    tech_basket = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "ADBE",
        "CRM",
        "NFLX",
        "AMD",
        "INTC",
        "QCOM",
    ]

    matrices = await mw.build_training_matrices(
        stock_basket=tech_basket,
        lookback_days=365 * 4,
        label_horizon_days=5,
    )

    X_all = np.asarray(matrices.get("X_numeric"), dtype=float)
    y_all = np.asarray(matrices.get("y"), dtype=int)
    ret_all = np.asarray(matrices.get("actual_returns"), dtype=float)
    dates = [str(d) for d in (matrices.get("sample_dates") or [])]

    if X_all.ndim != 2 or X_all.shape[0] < 500 or len(dates) != int(X_all.shape[0]):
        raise RuntimeError("Insufficient live market rows for evaluation")

    years = [d[:4] for d in dates]
    unique_years = sorted({y for y in years if len(y) == 4 and y.isdigit()})
    year_masks = {yy: np.array([d.startswith(f"{yy}-") for d in dates], dtype=bool) for yy in unique_years}

    selected_year = None
    if "2022" in year_masks and int(np.sum(year_masks["2022"])) >= 180:
        selected_year = "2022"
    else:
        candidates = []
        for yy, yy_mask in year_masks.items():
            if int(np.sum(yy_mask)) < 180:
                continue
            vol = float(np.std(ret_all[yy_mask], ddof=1)) if int(np.sum(yy_mask)) > 1 else 0.0
            candidates.append((vol, yy))
        if candidates:
            candidates.sort(reverse=True)
            selected_year = candidates[0][1]

    if selected_year is None:
        raise RuntimeError("No suitable evaluation year found")

    year_mask = year_masks[selected_year]
    X = X_all[year_mask]
    y = y_all[year_mask]
    returns = ret_all[year_mask]
    year_dates = np.asarray(dates, dtype=object)[year_mask]

    splits = mw._build_walk_forward_splits(
        n_samples=int(X.shape[0]),
        req_test_size=0.2,
        n_splits=4,
        min_train_samples=120,
    )
    if not splits:
        raise RuntimeError("Unable to construct walk-forward splits")

    algo_map = {name: factory for name, factory in mw._algo_factories(random_seed=42)}
    base_names = [name for name in ("random_forest", "gradient_boosting", "ann_sigmoid") if name in algo_map]
    if len(base_names) < 2:
        raise RuntimeError("Insufficient base model factories")

    base_oof = np.full((X.shape[0], len(base_names)), np.nan, dtype=float)
    for col_idx, name in enumerate(base_names):
        mw._fit_score_over_walk_forward(
            algorithm=name,
            factory=algo_map[name],
            X=X,
            y=y,
            splits=splits,
            oof_collector=base_oof[:, col_idx],
        )

    valid_mask = ~np.isnan(base_oof).any(axis=1)
    if int(np.sum(valid_mask)) < 120:
        raise RuntimeError("Insufficient OOF rows")

    valid_indices = np.where(valid_mask)[0]
    index_map = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(valid_indices.tolist())}
    filtered_splits = []
    for tr_idx, te_idx in splits:
        tr_mapped = [index_map[int(i)] for i in tr_idx.tolist() if int(i) in index_map]
        te_mapped = [index_map[int(i)] for i in te_idx.tolist() if int(i) in index_map]
        if len(tr_mapped) < 60 or len(te_mapped) < 20:
            continue
        filtered_splits.append((np.asarray(tr_mapped, dtype=int), np.asarray(te_mapped, dtype=int)))

    if not filtered_splits:
        n = int(np.sum(valid_mask))
        cut = max(60, int(n * 0.7))
        if n - cut < 20:
            raise RuntimeError("Not enough valid OOF rows to build fallback split")
        filtered_splits = [(np.arange(0, cut, dtype=int), np.arange(cut, n, dtype=int))]

    result = run_walk_forward_tournament(
        competitor_factories=[
            lambda: MetaLabelerCombiner(random_state=42),
            lambda: RegimeSwitcherCombiner(),
            lambda: RLAgentCombiner(),
        ],
        base_predictions=base_oof[valid_mask],
        market_features=X[valid_mask],
        actual_returns=returns[valid_mask],
        splits=filtered_splits,
    )

    eval_indices = np.concatenate([te_idx for _, te_idx in filtered_splits]).astype(int)
    ml_raw = np.asarray(result.daily_returns[result.winner_name], dtype=float)
    if ml_raw.size != eval_indices.size:
        raise RuntimeError("Mismatch between tournament daily returns and evaluation index size")

    eval_dates = np.asarray(year_dates[valid_mask], dtype=object)[eval_indices]

    spy_matrices = await mw.build_training_matrices(
        stock_basket=["SPY"],
        lookback_days=365 * 4,
        label_horizon_days=5,
    )
    spy_returns_all = np.asarray(spy_matrices.get("actual_returns"), dtype=float)
    spy_dates_all = [str(d) for d in (spy_matrices.get("sample_dates") or [])]
    if spy_returns_all.size == 0 or len(spy_dates_all) != int(spy_returns_all.size):
        raise RuntimeError("Unable to build SPY baseline returns")

    spy_map: dict[str, float] = {}
    for d, r in zip(spy_dates_all, spy_returns_all.tolist()):
        if not str(d).startswith(f"{selected_year}-"):
            continue
        spy_map[str(d)] = float(r)

    aligned_ml: list[float] = []
    aligned_spy: list[float] = []
    aligned_dates: list[str] = []
    for d, ml_r in zip(eval_dates.tolist(), ml_raw.tolist()):
        key = str(d)
        if key not in spy_map:
            continue
        aligned_ml.append(0.5 * float(ml_r))
        aligned_spy.append(float(spy_map[key]))
        aligned_dates.append(key)

    ml_arr = np.asarray(aligned_ml, dtype=float)
    spy_arr = np.asarray(aligned_spy, dtype=float)

    winner_metrics = {
        "sharpe": annualized_sharpe(ml_arr),
        "win_rate": win_rate(ml_arr),
        "max_drawdown": max_drawdown(ml_arr),
        "samples": int(ml_arr.size),
    }
    spy_metrics = {
        "sharpe": annualized_sharpe(spy_arr),
        "win_rate": win_rate(spy_arr),
        "max_drawdown": max_drawdown(spy_arr),
        "samples": int(spy_arr.size),
    }

    ml_eq = np.cumprod(1.0 + ml_arr)
    spy_eq = np.cumprod(1.0 + spy_arr)

    out_dir = Path(__file__).resolve().parents[1] / "model_store" / "visualizations" / "tournament"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "evaluation_equity_curve_latest.png"

    plot_saved = False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(ml_eq.size)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(x, ml_eq, label=f"Winner ({result.winner_name})", color="#1f77b4", linewidth=2.0)
        ax.plot(x, spy_eq, label="S&P 500 baseline (SPY)", color="#ff7f0e", linewidth=2.0)
        ax.set_title(f"Walk-Forward Evaluation Equity Curve ({selected_year})")
        ax.set_xlabel("Evaluation Day Index")
        ax.set_ylabel("Cumulative Equity (start=1.0)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_saved = True
    except Exception:
        plot_saved = False

    output = {
        "selected_year": selected_year,
        "winner_name": result.winner_name,
        "winner_metrics_vs_spy": {
            "winner": winner_metrics,
            "spy": spy_metrics,
        },
        "equity_curve_plot": str(plot_path),
        "plot_saved": plot_saved,
        "aligned_rows": len(aligned_dates),
        "regime_switcher_stats": result.competitor_stats.get("regime_switcher", {}),
        "rl_agent_stats": result.competitor_stats.get("rl_agent", {}),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
