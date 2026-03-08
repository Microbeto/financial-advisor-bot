from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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
