from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.ml_workflow import (
    DataValidationError,
    _DailyNewsFeatures,
    _algo_factories,
    _build_walk_forward_splits,
    _data_completeness_from_news_coverage,
    _fit_score_over_walk_forward,
    _rank_models,
    _select_prune_candidates,
    _sentiment_score_text,
    _technical_features,
    _validate_news_coverage_or_raise,
    _weighted_score,
    build_training_matrices,
    get_tournament_competitor_stats,
)


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if int(np.sum(mask)) < 3:
        return 0.0

    xr = pd.Series(xs[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(ys[mask]).rank(method="average").to_numpy(dtype=float)

    corr = np.corrcoef(xr, yr)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def test_sentiment_scoring_positive_negative():
    pos = _sentiment_score_text("Company beats earnings and gets upgrade with strong growth")
    neg = _sentiment_score_text("Company misses estimates and faces lawsuit with weak guidance")

    assert pos > 0.0
    assert neg < 0.0


def test_technical_features_shape_and_values():
    closes = [float(100 + i) for i in range(40)]
    feats = _technical_features(closes)

    assert len(feats) == 5
    assert feats[1] > 0.0
    assert feats[2] > 0.0


def test_rank_models_orders_by_weighted_score():
    m1 = {
        "algorithm": "a",
        "metrics": {"accuracy": 0.80, "precision": 0.80, "recall": 0.80, "f1": 0.80},
        "score": _weighted_score({"accuracy": 0.80, "precision": 0.80, "recall": 0.80, "f1": 0.80}),
    }
    m2 = {
        "algorithm": "b",
        "metrics": {"accuracy": 0.60, "precision": 0.60, "recall": 0.60, "f1": 0.60},
        "score": _weighted_score({"accuracy": 0.60, "precision": 0.60, "recall": 0.60, "f1": 0.60}),
    }

    ranked = _rank_models([m2, m1])

    assert ranked[0]["algorithm"] == "a"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2


def test_weighted_score_prefers_f1():
    high_f1 = _weighted_score({"accuracy": 0.70, "precision": 0.70, "recall": 0.70, "f1": 0.90})
    low_f1 = _weighted_score({"accuracy": 0.80, "precision": 0.80, "recall": 0.80, "f1": 0.50})

    assert high_f1 > low_f1


def test_feature_stability_for_flat_prices():
    closes = list(np.repeat(100.0, 50))
    feats = _technical_features(closes)

    assert feats[0] == 0.0
    assert feats[1] == 0.0
    assert feats[2] == 0.0


def test_prune_candidates_only_when_at_capacity():
    docs = [{"model_id": "a", "is_selected": True}, {"model_id": "b", "is_selected": False}]
    out = _select_prune_candidates(docs, max_models=10)
    assert out == []


def test_prune_candidates_non_selected_at_capacity():
    docs = [{"model_id": "s", "is_selected": True}] + [
        {"model_id": f"m{i}", "is_selected": False} for i in range(1, 10)
    ]
    out = _select_prune_candidates(docs, max_models=10)
    assert len(out) == 9
    assert all(not x["is_selected"] for x in out)


def test_tournament_stats_returns_empty_for_non_tournament(monkeypatch):
    monkeypatch.setattr(
        "app.services.ml_workflow._selected_or_latest_model",
        lambda model_id=None: {"model_id": "rf_1", "algorithm": "random_forest"},
    )

    out = get_tournament_competitor_stats()
    assert out["model_id"] == "rf_1"
    assert out["algorithm"] == "random_forest"
    assert out["winner_name"] is None
    assert out["competitor_stats"] == {}


def test_tournament_stats_extracts_stats_from_bundle(monkeypatch):
    monkeypatch.setattr(
        "app.services.ml_workflow._selected_or_latest_model",
        lambda model_id=None: {"model_id": "mat_1", "algorithm": "multi_armed_tournament"},
    )
    monkeypatch.setattr(
        "app.services.ml_workflow._load_model",
        lambda doc: {
            "winner_name": "rl_agent",
            "competitor_stats": {
                "rl_agent": {
                    "sharpe": 1.2,
                    "max_drawdown": -0.15,
                    "mean_return": 0.01,
                    "volatility": 0.08,
                    "sample_count": 64,
                }
            },
        },
    )

    out = get_tournament_competitor_stats(model_id="mat_1")
    assert out["model_id"] == "mat_1"
    assert out["algorithm"] == "multi_armed_tournament"
    assert out["winner_name"] == "rl_agent"
    assert out["competitor_stats"]["rl_agent"]["sharpe"] == 1.2


def test_news_coverage_gate_passes_within_threshold(monkeypatch):
    monkeypatch.setattr("app.services.ml_workflow.ML_ENFORCE_NEWS_COVERAGE", True)
    monkeypatch.setattr("app.services.ml_workflow.ML_MAX_NEWS_MISSING_RATIO", 0.10)

    daily_ctx = {
        "2026-03-01": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=5),
        "2026-03-02": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=2),
        "2026-03-03": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=0),
        "2026-03-04": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=3),
        "2026-03-05": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=4),
        "2026-03-06": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=1),
        "2026-03-07": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=6),
        "2026-03-08": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=2),
        "2026-03-09": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=3),
        "2026-03-10": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=2),
    }

    report = _validate_news_coverage_or_raise(daily_ctx, symbols=["AAPL", "MSFT"])
    assert report["window_days"] == 10
    assert report["missing_days"] == 1
    assert abs(float(report["missing_ratio"]) - 0.1) < 1e-9


def test_news_coverage_gate_fails_when_missing_exceeds_threshold(monkeypatch):
    monkeypatch.setattr("app.services.ml_workflow.ML_ENFORCE_NEWS_COVERAGE", True)
    monkeypatch.setattr("app.services.ml_workflow.ML_MAX_NEWS_MISSING_RATIO", 0.10)

    daily_ctx = {
        "2026-03-01": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=0),
        "2026-03-02": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=0),
        "2026-03-03": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=0),
        "2026-03-04": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=0),
        "2026-03-05": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=1),
        "2026-03-06": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=2),
        "2026-03-07": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=3),
        "2026-03-08": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=2),
        "2026-03-09": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=1),
        "2026-03-10": _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[], news_item_count=1),
    }

    try:
        _validate_news_coverage_or_raise(daily_ctx, symbols=["AAPL", "MSFT"])
        assert False, "Expected DataValidationError when missing ratio exceeds threshold"
    except DataValidationError as exc:
        assert exc.code == "news_coverage_threshold_exceeded"
        assert float(exc.details.get("missing_ratio") or 0.0) > 0.10


def test_data_completeness_from_news_coverage_ratio():
    out = _data_completeness_from_news_coverage(
        {
            "window_days": 100,
            "missing_days": 2,
            "missing_ratio": 0.02,
        }
    )

    assert abs(float(out["news_coverage_ratio"]) - 0.98) < 1e-9
    assert abs(float(out["news_missing_ratio"]) - 0.02) < 1e-9
    assert int(out["news_window_days"]) == 100
    assert int(out["news_missing_days"]) == 2


def test_model_information_coefficient_is_positive(monkeypatch):
    # Keep IC evaluation focused on market/price signal by freezing daily news context.
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

    # Representative large-cap S&P 500 subset to keep runtime practical while preserving breadth.
    sp500_basket = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "XOM", "JNJ",
        "V", "PG", "MA", "HD", "COST", "AVGO", "ABBV", "MRK", "PEP", "KO",
        "BAC", "WMT", "TMO", "CRM", "ADBE", "ACN", "MCD", "DHR", "LIN", "CSCO",
    ]

    matrices = asyncio.run(
        build_training_matrices(
            stock_basket=sp500_basket,
            lookback_days=365 * 3,
            label_horizon_days=5,
        )
    )

    X = np.asarray(matrices.get("X_numeric"), dtype=float)
    y = np.asarray(matrices.get("y"), dtype=int)
    actual_returns = np.asarray(matrices.get("actual_returns"), dtype=float)

    if X.ndim != 2 or X.shape[0] < 140:
        pytest.skip("Insufficient live market rows for IC integration test in this environment")

    splits = _build_walk_forward_splits(
        n_samples=int(X.shape[0]),
        req_test_size=0.2,
        n_splits=5,
        min_train_samples=120,
    )
    if not splits:
        pytest.skip("Unable to construct walk-forward splits for IC integration test")

    algo_map = {name: factory for name, factory in _algo_factories(random_seed=42)}
    factory = algo_map.get("random_forest")
    assert factory is not None

    oof_prob = np.full(X.shape[0], np.nan, dtype=float)
    _fit_score_over_walk_forward(
        algorithm="random_forest",
        factory=factory,
        X=X,
        y=y,
        splits=splits,
        oof_collector=oof_prob,
    )

    valid_mask = np.isfinite(oof_prob) & np.isfinite(actual_returns)
    if int(np.sum(valid_mask)) < 80:
        pytest.skip("Insufficient out-of-fold samples for robust IC evaluation")

    ic = _spearman_rank_correlation(oof_prob[valid_mask], actual_returns[valid_mask])
    assert ic > 0.02, f"Expected IC > 0.02, got {ic:.4f}"


def test_model_beats_random_permutation(monkeypatch):
    # Freeze news side-features so the test measures price/label structure consistently.
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

    sp500_basket = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "XOM", "JNJ",
        "V", "PG", "MA", "HD", "COST", "AVGO", "ABBV", "MRK", "PEP", "KO",
        "BAC", "WMT", "TMO", "CRM", "ADBE", "ACN", "MCD", "DHR", "LIN", "CSCO",
    ]

    matrices = asyncio.run(
        build_training_matrices(
            stock_basket=sp500_basket,
            lookback_days=365 * 3,
            label_horizon_days=5,
        )
    )

    X = np.asarray(matrices.get("X_numeric"), dtype=float)
    y = np.asarray(matrices.get("y"), dtype=int)
    if X.ndim != 2 or X.shape[0] < 140:
        pytest.skip("Insufficient live market rows for permutation reality-check test")

    splits = _build_walk_forward_splits(
        n_samples=int(X.shape[0]),
        req_test_size=0.2,
        n_splits=5,
        min_train_samples=120,
    )
    if not splits:
        pytest.skip("Unable to construct walk-forward splits for permutation reality-check")

    algo_map = {name: factory for name, factory in _algo_factories(random_seed=42)}
    factory = algo_map.get("random_forest")
    assert factory is not None

    def _walk_forward_macro_f1(y_input: np.ndarray) -> float:
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        for tr_idx, te_idx in splits:
            model = factory()
            model.fit(X[tr_idx], y_input[tr_idx])
            probs = np.asarray(model.predict_proba(X[te_idx])[:, 1], dtype=float)
            preds = np.where(probs >= 0.5, 1, 0)

            y_true_all.extend([int(v) for v in y_input[te_idx]])
            y_pred_all.extend([int(v) for v in preds])

        if not y_true_all:
            return 0.0

        return float(
            f1_score(
                np.asarray(y_true_all, dtype=int),
                np.asarray(y_pred_all, dtype=int),
                average="macro",
                zero_division=0,
            )
        )

    real_f1 = _walk_forward_macro_f1(y)

    rng = np.random.default_rng(20260308)
    permutation_f1: list[float] = []
    for _ in range(50):
        y_perm = rng.permutation(y)
        permutation_f1.append(_walk_forward_macro_f1(y_perm))

    perm_arr = np.asarray(permutation_f1, dtype=float)
    perm_mean = float(np.mean(perm_arr)) if perm_arr.size else 0.0
    perm_std = float(np.std(perm_arr, ddof=1)) if perm_arr.size > 1 else 0.0

    # One-sample t-style separation of real score versus permutation mean.
    # This checks the real model outperforms the average random-label model with margin.
    if perm_std <= 1e-12:
        t_stat = float("inf") if real_f1 > perm_mean else 0.0
    else:
        t_stat = (real_f1 - perm_mean) / (perm_std / np.sqrt(float(perm_arr.size)))

    assert real_f1 > perm_mean, f"Expected real macro F1 ({real_f1:.4f}) > mean permuted macro F1 ({perm_mean:.4f})"
    assert t_stat > 2.0, f"Expected t-stat > 2.0 for permutation separation, got {t_stat:.4f}"
