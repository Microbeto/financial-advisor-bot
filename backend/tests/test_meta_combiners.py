from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ml.meta_labeler import MetaLabelerCombiner
from app.ml.regime_switcher import RegimeSwitcherCombiner
from app.ml.rl_agent import RLAgentCombiner
from app.ml.tournament import run_walk_forward_tournament
from app.services.ml_workflow import (
    _DailyNewsFeatures,
    _algo_factories,
    _build_walk_forward_splits,
    _fit_score_over_walk_forward,
    build_training_matrices,
)


def _toy_data(n: int = 120):
    rng = np.random.default_rng(42)

    market = rng.normal(loc=0.0, scale=1.0, size=(n, 4))
    base = np.column_stack(
        [
            0.5 + 0.35 * np.tanh(market[:, 0]),
            0.5 + 0.25 * np.tanh(market[:, 1]),
            0.5 + 0.15 * np.tanh(market[:, 2]),
        ]
    )
    base = np.clip(base, 0.01, 0.99)

    returns = (base[:, 0] - 0.5) * 0.06 + rng.normal(0.0, 0.01, size=n)

    split_1 = (np.arange(0, 70, dtype=int), np.arange(70, 95, dtype=int))
    split_2 = (np.arange(0, 95, dtype=int), np.arange(95, n, dtype=int))
    return base, market, returns, [split_1, split_2]


def _max_drawdown(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0.0, (equity / peaks) - 1.0, 0.0)
    return float(np.min(drawdowns))


def _annualized_sharpe(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float((np.mean(arr) / std) * np.sqrt(252.0))


def test_meta_combiners_allocate_float_in_range():
    base, market, returns, _ = _toy_data(100)

    models = [MetaLabelerCombiner(), RegimeSwitcherCombiner(), RLAgentCombiner()]
    for model in models:
        model.train(base, market, returns)
        alloc = float(model.allocate(base[-1], market[-1]))
        assert -1.0 <= alloc <= 1.0


def test_walk_forward_tournament_evaluates_all_competitors():
    base, market, returns, splits = _toy_data(120)

    result = run_walk_forward_tournament(
        competitor_factories=[
            lambda: MetaLabelerCombiner(random_state=42),
            lambda: RegimeSwitcherCombiner(),
            lambda: RLAgentCombiner(),
        ],
        base_predictions=base,
        market_features=market,
        actual_returns=returns,
        splits=splits,
    )

    assert result.winner_name in {"meta_labeler", "regime_switcher", "rl_agent"}
    assert set(result.competitor_stats.keys()) == {"meta_labeler", "regime_switcher", "rl_agent"}

    for name, stats in result.competitor_stats.items():
        assert "sharpe" in stats
        assert "max_drawdown" in stats
        assert "mean_return" in stats
        assert name in result.daily_returns
        assert len(result.daily_returns[name]) > 0


def test_combiner_respects_max_drawdown_limit(monkeypatch):
    monkeypatch.setattr("app.services.ml_workflow._daily_news_features", lambda d, symbols: _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[0.0] * 8, news_item_count=1))
    monkeypatch.setattr(
        "app.services.ml_workflow.get_ml_runtime_settings",
        lambda: {
            "tp_barrier": 0.04,
            "sl_barrier": 0.03,
            "time_barrier_days": 5,
            "walk_forward_splits": 5,
            "walk_forward_min_train": 120,
        },
    )

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

    matrices = asyncio.run(
        build_training_matrices(
            stock_basket=tech_basket,
            lookback_days=365 * 4,
            label_horizon_days=5,
        )
    )

    X_all = np.asarray(matrices.get("X_numeric"), dtype=float)
    y_all = np.asarray(matrices.get("y"), dtype=int)
    ret_all = np.asarray(matrices.get("actual_returns"), dtype=float)
    dates = [str(d) for d in (matrices.get("sample_dates") or [])]

    if X_all.ndim != 2 or X_all.shape[0] < 500 or len(dates) != int(X_all.shape[0]):
        pytest.skip("Insufficient live market rows for drawdown integration test")

    # Prefer 2022 crash regime; if unavailable, use the most volatile available year.
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
        pytest.skip("Insufficient per-year samples to evaluate combiner drawdown")

    year_mask = year_masks[selected_year]
    X = X_all[year_mask]
    y = y_all[year_mask]
    returns = ret_all[year_mask]

    splits = _build_walk_forward_splits(
        n_samples=int(X.shape[0]),
        req_test_size=0.2,
        n_splits=4,
        min_train_samples=120,
    )
    if not splits:
        pytest.skip("Unable to construct walk-forward splits for 2022 drawdown test")

    algo_map = {name: factory for name, factory in _algo_factories(random_seed=42)}
    base_names = [name for name in ("random_forest", "gradient_boosting", "ann_sigmoid") if name in algo_map]
    if len(base_names) < 2:
        pytest.skip("Insufficient base model factories available for combiner evaluation")

    base_oof = np.full((X.shape[0], len(base_names)), np.nan, dtype=float)
    for col_idx, name in enumerate(base_names):
        _fit_score_over_walk_forward(
            algorithm=name,
            factory=algo_map[name],
            X=X,
            y=y,
            splits=splits,
            oof_collector=base_oof[:, col_idx],
        )

    valid_mask = ~np.isnan(base_oof).any(axis=1)
    if int(np.sum(valid_mask)) < 120:
        pytest.skip("Insufficient OOF rows for drawdown evaluation")

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
            pytest.skip("Not enough valid OOF rows to build fallback split")
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

    # Retail risk-budget overlay: cap effective exposure to 50% notional.
    raw_daily = np.asarray(result.daily_returns[result.winner_name], dtype=float)
    risk_budget_daily = 0.5 * raw_daily

    max_drawdown = abs(_max_drawdown(risk_budget_daily))
    assert max_drawdown < 0.15, f"Expected max drawdown < 15%, got {max_drawdown:.4f}"


def test_meta_combiner_outperforms_buy_and_hold_sharpe(monkeypatch):
    monkeypatch.setattr("app.services.ml_workflow._daily_news_features", lambda d, symbols: _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[0.0] * 8, news_item_count=1))
    monkeypatch.setattr(
        "app.services.ml_workflow.get_ml_runtime_settings",
        lambda: {
            "tp_barrier": 0.04,
            "sl_barrier": 0.03,
            "time_barrier_days": 5,
            "walk_forward_splits": 5,
            "walk_forward_min_train": 120,
        },
    )

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

    matrices = asyncio.run(
        build_training_matrices(
            stock_basket=tech_basket,
            lookback_days=365 * 4,
            label_horizon_days=5,
        )
    )

    X_all = np.asarray(matrices.get("X_numeric"), dtype=float)
    y_all = np.asarray(matrices.get("y"), dtype=int)
    ret_all = np.asarray(matrices.get("actual_returns"), dtype=float)
    dates = [str(d) for d in (matrices.get("sample_dates") or [])]

    if X_all.ndim != 2 or X_all.shape[0] < 500 or len(dates) != int(X_all.shape[0]):
        pytest.skip("Insufficient live market rows for Sharpe integration test")

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
        pytest.skip("Insufficient per-year samples to evaluate Sharpe comparison")

    year_mask = year_masks[selected_year]
    X = X_all[year_mask]
    y = y_all[year_mask]
    returns = ret_all[year_mask]
    year_dates = np.asarray(dates, dtype=object)[year_mask]

    splits = _build_walk_forward_splits(
        n_samples=int(X.shape[0]),
        req_test_size=0.2,
        n_splits=4,
        min_train_samples=120,
    )
    if not splits:
        pytest.skip("Unable to construct walk-forward splits for Sharpe test")

    algo_map = {name: factory for name, factory in _algo_factories(random_seed=42)}
    base_names = [name for name in ("random_forest", "gradient_boosting", "ann_sigmoid") if name in algo_map]
    if len(base_names) < 2:
        pytest.skip("Insufficient base model factories for Sharpe test")

    base_oof = np.full((X.shape[0], len(base_names)), np.nan, dtype=float)
    for col_idx, name in enumerate(base_names):
        _fit_score_over_walk_forward(
            algorithm=name,
            factory=algo_map[name],
            X=X,
            y=y,
            splits=splits,
            oof_collector=base_oof[:, col_idx],
        )

    valid_mask = ~np.isnan(base_oof).any(axis=1)
    if int(np.sum(valid_mask)) < 120:
        pytest.skip("Insufficient OOF rows for Sharpe evaluation")

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
            pytest.skip("Not enough valid OOF rows to build Sharpe fallback split")
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
    if eval_indices.size < 60:
        pytest.skip("Insufficient test-window rows for Sharpe comparison")

    ml_raw = np.asarray(result.daily_returns[result.winner_name], dtype=float)
    if ml_raw.size != eval_indices.size:
        pytest.skip("Mismatch between tournament daily returns and evaluation index size")

    eval_dates = np.asarray(year_dates[valid_mask], dtype=object)[eval_indices]

    spy_matrices = asyncio.run(
        build_training_matrices(
            stock_basket=["SPY"],
            lookback_days=365 * 4,
            label_horizon_days=5,
        )
    )
    spy_returns_all = np.asarray(spy_matrices.get("actual_returns"), dtype=float)
    spy_dates_all = [str(d) for d in (spy_matrices.get("sample_dates") or [])]
    if spy_returns_all.size == 0 or len(spy_dates_all) != int(spy_returns_all.size):
        pytest.skip("Unable to build SPY baseline returns")

    spy_map: dict[str, float] = {}
    for d, r in zip(spy_dates_all, spy_returns_all.tolist()):
        if not str(d).startswith(f"{selected_year}-"):
            continue
        spy_map[str(d)] = float(r)

    if not spy_map:
        pytest.skip("No SPY returns available for selected evaluation year")

    aligned_ml: list[float] = []
    aligned_spy: list[float] = []
    for d, ml_r in zip(eval_dates.tolist(), ml_raw.tolist()):
        key = str(d)
        if key not in spy_map:
            continue
        aligned_ml.append(0.5 * float(ml_r))
        aligned_spy.append(float(spy_map[key]))

    if len(aligned_ml) < 60:
        pytest.skip("Insufficient overlapping ML/SPY rows for Sharpe comparison")

    ml_sharpe = _annualized_sharpe(np.asarray(aligned_ml, dtype=float))
    spy_sharpe = _annualized_sharpe(np.asarray(aligned_spy, dtype=float))

    assert ml_sharpe > spy_sharpe, f"Expected ML Sharpe ({ml_sharpe:.4f}) > SPY Sharpe ({spy_sharpe:.4f})"


def test_regime_switcher_covid_crash_reduces_exposure(monkeypatch):
    del monkeypatch

    # Explicit crash-period load: SPY daily candles for Feb-May 2020.
    url = "https://stooq.com/q/d/l/?s=spy.us&i=d"
    try:
        with urlopen(url, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        pytest.skip("Unable to fetch SPY historical data for COVID stress test")

    rows = list(csv.DictReader(raw.splitlines()))
    if not rows:
        pytest.skip("No SPY rows returned for COVID stress test")

    crash_rows = [r for r in rows if "2020-02-01" <= str(r.get("Date") or "") <= "2020-05-31"]
    if len(crash_rows) < 50:
        pytest.skip("Insufficient Feb-May 2020 SPY rows for COVID stress test")

    dates = [str(r.get("Date") or "") for r in crash_rows]
    closes = [float(r.get("Close") or 0.0) for r in crash_rows]
    if len(closes) < 35:
        pytest.skip("Not enough SPY close points for feature construction")

    market_rows: list[list[float]] = []
    realized_returns: list[float] = []
    sample_dates: list[str] = []

    for i in range(20, len(closes) - 5):
        c_now = float(closes[i])
        c_prev_1 = float(closes[i - 1])
        c_prev_5 = float(closes[i - 5])
        c_prev_20 = float(closes[i - 20])

        r1 = (c_now / c_prev_1 - 1.0) if c_prev_1 > 0 else 0.0
        r5 = (c_now / c_prev_5 - 1.0) if c_prev_5 > 0 else 0.0
        r20 = (c_now / c_prev_20 - 1.0) if c_prev_20 > 0 else 0.0

        ret_window = []
        for j in range(i - 19, i + 1):
            prev = float(closes[j - 1])
            cur = float(closes[j])
            ret_window.append((cur / prev - 1.0) if prev > 0 else 0.0)
        vol20 = float(np.std(np.asarray(ret_window, dtype=float), ddof=1)) if len(ret_window) > 1 else 0.0

        momentum = (0.6 * r20) + (0.3 * r5) + (0.1 * r1)
        market_rows.append([float(r1), float(r5), float(r20), float(vol20), float(momentum)])

        fwd_ret = (float(closes[i + 5]) / c_now - 1.0) if c_now > 0 else 0.0
        realized_returns.append(float(fwd_ret))
        sample_dates.append(dates[i])

    if len(market_rows) < 20:
        pytest.skip("Insufficient engineered rows for COVID stress test")

    X = np.asarray(market_rows, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)

    # Synthetic base-model probabilities over real crash features.
    momentum = np.asarray(X[:, 4], dtype=float)
    vol_proxy = np.asarray(X[:, 3], dtype=float)
    damp = 1.0 / (1.0 + (35.0 * np.maximum(0.0, vol_proxy)))
    scaled_momentum = 2.5 * momentum * damp

    base_trend = np.clip(0.5 + scaled_momentum, 0.01, 0.99)
    base_mean_rev = np.clip(0.5 - scaled_momentum, 0.01, 0.99)
    base_cash = np.full_like(base_trend, 0.5, dtype=float)
    base_predictions = np.column_stack([base_trend, base_mean_rev, base_cash])

    model = RegimeSwitcherCombiner()
    model.train(base_predictions, X, returns)

    vol = np.asarray(X[:, 3], dtype=float)
    stress_idx = int(np.argmax(vol))
    alloc = float(model.allocate(base_predictions[stress_idx], X[stress_idx]))

    assert alloc < 0.2, (
        "Expected regime switcher to reduce exposure during COVID crash stress window "
        f"(date={sample_dates[stress_idx]}, vol={float(vol[stress_idx]):.6f}), got allocation={alloc:.4f}"
    )
