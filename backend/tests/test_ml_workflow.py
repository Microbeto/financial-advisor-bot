# Unit tests for ML workflow including data validation, feature engineering, and model training.
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

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
    _triple_barrier_label,
    _doc_to_model_info,
    _data_completeness_from_news_coverage,
    _fit_score_over_walk_forward,
    _daily_news_features,
    _rank_models,
    _select_prune_candidates,
    _sentiment_score_text,
    _technical_features,
    _validate_news_coverage_or_raise,
    _weighted_score,
    backfill_model_nlp_pipeline,
    build_training_matrices,
    get_tournament_competitor_stats,
    predict_for_basket,
    select_best_models,
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


# Test: sentiment scoring positive negative.
def test_sentiment_scoring_positive_negative():
    pos = _sentiment_score_text("Company beats earnings and gets upgrade with strong growth")
    neg = _sentiment_score_text("Company misses estimates and faces lawsuit with weak guidance")

    assert pos > 0.0
    assert neg < 0.0


# Test: technical features shape and values.
def test_technical_features_shape_and_values():
    closes = [float(100 + i) for i in range(40)]
    feats = _technical_features(closes)

    assert len(feats) == 5
    assert feats[1] > 0.0
    assert feats[2] > 0.0


# Test: rank models orders by weighted score.
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


# Test: weighted score prefers f1.
def test_weighted_score_prefers_f1():
    high_f1 = _weighted_score({"accuracy": 0.70, "precision": 0.70, "recall": 0.70, "f1": 0.90})
    low_f1 = _weighted_score({"accuracy": 0.80, "precision": 0.80, "recall": 0.80, "f1": 0.50})

    assert high_f1 > low_f1


# Test: weighted score prefers lower future price error.
def test_weighted_score_prefers_lower_future_price_error():
    low_error = _weighted_score(
        {
            "accuracy": 0.66,
            "precision": 0.66,
            "recall": 0.66,
            "f1": 0.66,
            "future_price_mae_pct": 1.2,
        }
    )
    high_error = _weighted_score(
        {
            "accuracy": 0.82,
            "precision": 0.82,
            "recall": 0.82,
            "f1": 0.82,
            "future_price_mae_pct": 8.5,
        }
    )

    assert low_error > high_error


# Test: weighted score prioritizes institutional risk metrics.
def test_weighted_score_prioritizes_institutional_risk_metrics():
    strong_institutional = _weighted_score(
        {
            "accuracy": 0.58,
            "precision": 0.58,
            "recall": 0.58,
            "f1": 0.58,
            "annualized_sharpe": 1.8,
            "max_drawdown": -0.08,
            "future_price_mae_pct": 2.4,
        }
    )
    weak_institutional = _weighted_score(
        {
            "accuracy": 0.86,
            "precision": 0.86,
            "recall": 0.86,
            "f1": 0.86,
            "annualized_sharpe": 0.2,
            "max_drawdown": -0.35,
            "future_price_mae_pct": 1.6,
        }
    )

    assert strong_institutional > weak_institutional


# Test: rank models prefers lower future price error when available.
def test_rank_models_prefers_lower_future_price_error_when_available():
    lower_price_error = {
        "algorithm": "low_price_err",
        "metrics": {
            "accuracy": 0.62,
            "precision": 0.62,
            "recall": 0.62,
            "f1": 0.62,
            "future_price_mae_pct": 1.1,
        },
        "score": 0.1,
    }
    higher_price_error = {
        "algorithm": "high_price_err",
        "metrics": {
            "accuracy": 0.86,
            "precision": 0.86,
            "recall": 0.86,
            "f1": 0.86,
            "future_price_mae_pct": 9.4,
        },
        "score": 0.9,
    }

    ranked = _rank_models([higher_price_error, lower_price_error])
    assert ranked[0]["algorithm"] == "low_price_err"


# Test: rank models prioritizes sharpe and drawdown with price factor.
def test_rank_models_prioritizes_sharpe_and_drawdown_with_price_factor():
    institutional_winner = {
        "algorithm": "institutional_winner",
        "metrics": {
            "accuracy": 0.60,
            "precision": 0.60,
            "recall": 0.60,
            "f1": 0.60,
            "annualized_sharpe": 1.5,
            "max_drawdown": -0.10,
            "future_price_mae_pct": 2.1,
        },
        "score": 0.0,
    }
    directional_only_candidate = {
        "algorithm": "directional_only",
        "metrics": {
            "accuracy": 0.90,
            "precision": 0.90,
            "recall": 0.90,
            "f1": 0.90,
            "annualized_sharpe": 0.1,
            "max_drawdown": -0.45,
            "future_price_mae_pct": 1.2,
        },
        "score": 1.0,
    }

    ranked = _rank_models([directional_only_candidate, institutional_winner])
    assert ranked[0]["algorithm"] == "institutional_winner"


# Test: feature stability for flat prices.
def test_feature_stability_for_flat_prices():
    closes = list(np.repeat(100.0, 50))
    feats = _technical_features(closes)

    assert feats[0] == 0.0
    assert feats[1] == 0.0
    assert feats[2] == 0.0


# Test: triple barrier uses volatility scaled thresholds.
def test_triple_barrier_uses_volatility_scaled_thresholds():
    # Returns here are sub-1%, so static TP=2.0 would never trigger.
    closes = [100.0, 100.2, 100.0, 100.3, 100.1, 100.5, 101.2, 101.5]
    label = _triple_barrier_label(closes, idx=5, horizon=2, tp=2.0, sl=1.5)
    assert label == 1


# Test: walk forward splits apply purge gap.
def test_walk_forward_splits_apply_purge_gap():
    splits = _build_walk_forward_splits(
        n_samples=300,
        req_test_size=0.2,
        n_splits=4,
        min_train_samples=80,
        purge_days=10,
    )
    assert splits

    for tr_idx, te_idx in splits:
        assert len(tr_idx) >= 80
        te_start = int(te_idx[0])
        assert int(np.max(tr_idx)) <= (te_start - 11)


# Test: prune candidates only when at capacity.
def test_prune_candidates_only_when_at_capacity():
    docs = [{"model_id": "a", "is_selected": True}, {"model_id": "b", "is_selected": False}]
    out = _select_prune_candidates(docs, max_models=10)
    assert out == []


# Test: prune candidates non selected at capacity.
def test_prune_candidates_non_selected_at_capacity():
    docs = [{"model_id": "s", "is_selected": True}] + [
        {"model_id": f"m{i}", "is_selected": False} for i in range(1, 10)
    ]
    out = _select_prune_candidates(docs, max_models=10)
    assert len(out) == 9
    assert all(not x["is_selected"] for x in out)


# Test: tournament stats returns empty for non tournament.
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


# Test: tournament stats extracts stats from bundle.
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


# Test: news coverage gate passes within threshold.
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


# Test: news coverage gate fails when missing exceeds threshold.
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


# Test: data completeness from news coverage ratio.
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


# Test: model information coefficient is positive.
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


# Test: model beats random permutation.
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


# Test: inflation shock rotates out of tech into defensive or cash.
def test_inflation_shock_rotates_out_of_tech_into_defensive_or_cash(monkeypatch):
    class _DummyFinBert:
        def score_texts(self, texts):
            out = []
            for t in texts:
                s = str(t or "").lower()
                score = 0.0
                if "rate hike" in s or "federal reserve" in s or "inflation" in s:
                    score -= 0.7
                if "defensive" in s or "resilient" in s:
                    score += 0.2
                out.append(float(max(-1.0, min(1.0, score))))
            return out

    def _mock_news_for_date(date, symbols, limit=350, include_market=True):
        del date, limit, include_market
        news = [
            SimpleNamespace(title="Federal Reserve signals another rate hike", summary="Persistent inflation keeps policy restrictive.", symbol=""),
            SimpleNamespace(title="Markets brace for consecutive rate hike path", summary="Higher-for-longer outlook pressures risk assets.", symbol=""),
            SimpleNamespace(title="Inflation surprise raises odds of aggressive rate hike", summary="Treasury yields jump as Fed hawkishness rises.", symbol=""),
        ]

        # Symbol-level headlines remain weak for tech and mixed for defensives.
        for sym in (symbols or []):
            s = str(sym or "").upper()
            if s in {"XLU", "XLP", "JNJ", "PG", "KO"}:
                news.append(SimpleNamespace(title=f"{s} seen as defensive amid rate hike cycle", summary="Cash flows remain resilient.", symbol=s))
            else:
                news.append(SimpleNamespace(title=f"{s} valuation pressured by rate hike repricing", summary="Duration-sensitive growth weakens.", symbol=s))
        return news

    class _SentimentDrivenModel:
        # Feature layout: tech[0:5], symbol_sent[5], market_sent[6], macro[7:]
        def predict_proba(self, x):
            arr = np.asarray(x, dtype=float)
            sym_sent = arr[:, 5]
            mkt_sent = arr[:, 6]
            p_up = np.clip(0.5 + (0.30 * sym_sent) + (0.20 * mkt_sent), 0.01, 0.99)
            return np.column_stack([1.0 - p_up, p_up])

    async def _mock_histories(symbols, days=120, concurrency=8):
        del days, concurrency
        out = {}
        for sym in symbols:
            base = 100.0 + (sum(ord(ch) for ch in str(sym)) % 11)
            pts = [{"c": float(base + (0.05 * i))} for i in range(90)]
            out[str(sym)] = {"points": pts}
        return out

    monkeypatch.setattr("app.services.ml_workflow._get_finbert", lambda: _DummyFinBert())
    monkeypatch.setattr("app.services.ml_workflow.get_news_for_date", _mock_news_for_date)
    monkeypatch.setattr("app.services.ml_workflow.get_price_histories", _mock_histories)
    monkeypatch.setattr("app.services.ml_workflow.iso_date_utc", lambda: "2022-06-16")
    monkeypatch.setattr(
        "app.services.ml_workflow._selected_or_latest_model",
        lambda model_id=None: {"model_id": str(model_id or "inflation_stub"), "algorithm": "random_forest"},
    )
    monkeypatch.setattr("app.services.ml_workflow._load_model", lambda doc: _SentimentDrivenModel())

    basket = ["NVDA", "TSLA", "AMD", "XLU", "XLP", "JNJ"]

    # Verify consecutive hawkish days are translated into negative market sentiment.
    daily = [
        _daily_news_features("2022-06-13", basket),
        _daily_news_features("2022-06-14", basket),
        _daily_news_features("2022-06-15", basket),
    ]
    assert all(float(d.market_sentiment) < -0.2 for d in daily)

    pred = asyncio.run(predict_for_basket(stock_basket=basket, lookback_days=120, model_id="inflation_stub"))

    probs = {it.symbol: float(it.probability_up) for it in pred.items}
    raw_w = {sym: max(0.0, p - 0.5) for sym, p in probs.items()}
    total = float(sum(raw_w.values()))
    weights = {sym: (w / total if total > 0 else 0.0) for sym, w in raw_w.items()}
    cash_weight = 1.0 - float(sum(weights.values()))

    tech_weight = float(sum(weights.get(s, 0.0) for s in ["NVDA", "TSLA", "AMD"]))
    defensive_weight = float(sum(weights.get(s, 0.0) for s in ["XLU", "XLP", "JNJ"]))

    assert (defensive_weight > tech_weight) or (cash_weight >= 0.5), (
        "Expected inflation shock rotation out of high-beta tech into defensive sectors or cash "
        f"(tech={tech_weight:.3f}, defensive={defensive_weight:.3f}, cash={cash_weight:.3f})"
    )


# Test: doc to model info reads news coverage ratio from metadata data completeness.
def test_doc_to_model_info_reads_news_coverage_ratio_from_metadata_data_completeness():
    doc = {
        "model_id": "legacy_1",
        "algorithm": "random_forest",
        "family": "tree_ensemble",
        "feature_type": "numeric",
        "metrics": {"accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1": 0.7, "roc_auc": 0.7},
        "score": 0.7,
        "rank": 1,
        "sample_count": 100,
        "metadata": {
            "data_completeness": {
                "news_coverage_ratio": 0.975,
                "news_missing_ratio": 0.025,
                "news_window_days": 200,
                "news_missing_days": 5,
            }
        },
    }

    out = _doc_to_model_info(doc)
    assert out.news_coverage_ratio is not None
    assert abs(float(out.news_coverage_ratio) - 0.975) < 1e-9


# Test: doc to model info reads news coverage ratio from run doc.
def test_doc_to_model_info_reads_news_coverage_ratio_from_run_doc(monkeypatch):
    class _FakeRuns:
        @staticmethod
        def find_one(query, projection=None):
            del projection
            if query.get("run_id") == "run_legacy":
                return {
                    "news_coverage": {
                        "window_days": 100,
                        "missing_days": 3,
                        "missing_ratio": 0.03,
                    }
                }
            return None

    monkeypatch.setattr("app.services.ml_workflow._runs_col", lambda: _FakeRuns())

    doc = {
        "model_id": "legacy_2",
        "run_id": "run_legacy",
        "algorithm": "svm_rbf",
        "family": "svm",
        "feature_type": "numeric",
        "metrics": {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6, "roc_auc": 0.6},
        "score": 0.6,
        "rank": 2,
        "sample_count": 100,
    }

    out = _doc_to_model_info(doc)
    assert out.news_coverage_ratio is not None
    assert abs(float(out.news_coverage_ratio) - 0.97) < 1e-9


# Test: predict for basket rejects model pipeline mismatch.
def test_predict_for_basket_rejects_model_pipeline_mismatch(monkeypatch):
    class _ToyModel:
        @staticmethod
        def predict_proba(x):
            arr = np.asarray(x, dtype=float)
            n = arr.shape[0]
            p_up = np.full(n, 0.6, dtype=float)
            return np.column_stack([1.0 - p_up, p_up])

    async def _mock_histories(symbols, days=120, concurrency=8):
        del days, concurrency
        return {str(sym): {"points": [{"c": float(100 + i)} for i in range(60)]} for sym in symbols}

    monkeypatch.setattr("app.services.ml_workflow._active_nlp_pipeline", lambda: "lexicon")
    monkeypatch.setattr(
        "app.services.ml_workflow._daily_news_features",
        lambda d, symbols, nlp_pipeline=None: _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[0.0] * 8, news_item_count=1, nlp_pipeline=str(nlp_pipeline or "lexicon")),
    )
    monkeypatch.setattr("app.services.ml_workflow.get_price_histories", _mock_histories)
    monkeypatch.setattr(
        "app.services.ml_workflow._selected_or_latest_model",
        lambda model_id=None: {"model_id": str(model_id or "mismatch_stub"), "algorithm": "random_forest", "nlp_pipeline": "finbert"},
    )
    monkeypatch.setattr("app.services.ml_workflow._load_model", lambda doc: _ToyModel())

    with pytest.raises(RuntimeError, match="incompatible NLP pipeline"):
        asyncio.run(predict_for_basket(stock_basket=["AAPL", "MSFT"], lookback_days=120, model_id="mismatch_stub"))


# Test: predict for basket uses selected docs matching runtime pipeline.
def test_predict_for_basket_uses_selected_docs_matching_runtime_pipeline(monkeypatch):
    class _ToyModel:
        @staticmethod
        def predict_proba(x):
            arr = np.asarray(x, dtype=float)
            n = arr.shape[0]
            p_up = np.full(n, 0.7, dtype=float)
            return np.column_stack([1.0 - p_up, p_up])

    class _FakeCursor:
        def __init__(self, docs):
            self._docs = list(docs)

        def sort(self, _spec):
            return self

        def __iter__(self):
            return iter(self._docs)

    class _FakeRegistry:
        @staticmethod
        def find(query):
            if query.get("is_selected") and query.get("nlp_pipeline") == "lexicon":
                return _FakeCursor([
                    {"model_id": "lex_ok", "algorithm": "random_forest", "nlp_pipeline": "lexicon", "artifact_path": "stub"}
                ])
            return _FakeCursor([])

    async def _mock_histories(symbols, days=120, concurrency=8):
        del days, concurrency
        return {str(sym): {"points": [{"c": float(100 + i)} for i in range(60)]} for sym in symbols}

    monkeypatch.setattr("app.services.ml_workflow._active_nlp_pipeline", lambda: "lexicon")
    monkeypatch.setattr(
        "app.services.ml_workflow._daily_news_features",
        lambda d, symbols, nlp_pipeline=None: _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[0.0] * 8, news_item_count=1, nlp_pipeline=str(nlp_pipeline or "lexicon")),
    )
    monkeypatch.setattr("app.services.ml_workflow.get_price_histories", _mock_histories)
    monkeypatch.setattr("app.services.ml_workflow._registry_col", lambda: _FakeRegistry())
    monkeypatch.setattr("app.services.ml_workflow._selected_or_latest_model", lambda model_id=None: None)
    monkeypatch.setattr("app.services.ml_workflow._load_model", lambda doc: _ToyModel())

    out = asyncio.run(predict_for_basket(stock_basket=["AAPL", "MSFT"], lookback_days=120, model_id=None))
    assert out.model_id == "mix[lex_ok]"
    assert len(out.items) == 2


# Test: select best models filters by pipeline.
def test_select_best_models_filters_by_pipeline(monkeypatch):
    class _FakeCursor:
        def __init__(self, docs):
            self._docs = list(docs)

        def sort(self, _spec):
            return self

        def __iter__(self):
            return iter(self._docs)

    class _FakeRegistry:
        def __init__(self):
            self.docs = [
                {"model_id": "fin_1", "score": 0.9, "created_at": 1, "nlp_pipeline": "finbert", "is_selected": False},
                {"model_id": "lex_1", "score": 0.8, "created_at": 2, "nlp_pipeline": "lexicon", "is_selected": False},
            ]

        def find(self, query):
            out = []
            for d in self.docs:
                if "nlp_pipeline" in query and d.get("nlp_pipeline") != query.get("nlp_pipeline"):
                    continue
                out.append(dict(d))
            return _FakeCursor(out)

        def update_many(self, query, update):
            for d in self.docs:
                if "nlp_pipeline" in query and d.get("nlp_pipeline") != query.get("nlp_pipeline"):
                    continue
                if "model_id" in query and isinstance(query.get("model_id"), dict):
                    allowed = set(query["model_id"].get("$in") or [])
                    if d.get("model_id") not in allowed:
                        continue
                for k, v in (update.get("$set") or {}).items():
                    d[k] = v

    fake_col = _FakeRegistry()
    monkeypatch.setattr("app.services.ml_workflow._registry_col", lambda: fake_col)

    out = select_best_models(top_k=1, nlp_pipeline="lexicon")
    assert out.selected_model_id == "lex_1"
    assert any(d.get("model_id") == "lex_1" and d.get("is_selected") for d in fake_col.docs)
    assert all(not d.get("is_selected") for d in fake_col.docs if d.get("model_id") != "lex_1")


# Test: backfill model nlp pipeline uses run metadata.
def test_backfill_model_nlp_pipeline_uses_run_metadata(monkeypatch):
    class _FakeCursor:
        def __init__(self, docs):
            self._docs = list(docs)

        def sort(self, _spec):
            return self

        def __iter__(self):
            return iter(self._docs)

    class _FakeRegistry:
        def __init__(self):
            self.docs = [{"model_id": "m1", "run_id": "r1", "artifact_path": "", "metadata": {}}]

        def find(self, _query):
            return _FakeCursor([dict(x) for x in self.docs])

        def update_one(self, query, update):
            mid = str(query.get("model_id") or "")
            for d in self.docs:
                if str(d.get("model_id") or "") != mid:
                    continue
                for k, v in (update.get("$set") or {}).items():
                    if k == "metadata.nlp_pipeline":
                        md = dict(d.get("metadata") or {})
                        md["nlp_pipeline"] = v
                        d["metadata"] = md
                    else:
                        d[k] = v

    class _FakeRuns:
        @staticmethod
        def find_one(query, projection=None):
            del projection
            if query.get("run_id") == "r1":
                return {"nlp_pipeline": "cascade", "result": {"nlp_pipeline": "cascade"}}
            return None

    fake_col = _FakeRegistry()
    monkeypatch.setattr("app.services.ml_workflow._registry_col", lambda: fake_col)
    monkeypatch.setattr("app.services.ml_workflow._runs_col", lambda: _FakeRuns())

    dry = backfill_model_nlp_pipeline(dry_run=True)
    assert dry["inspected"] == 1
    assert dry["changes"][0]["nlp_pipeline"] == "cascade"

    applied = backfill_model_nlp_pipeline(dry_run=False)
    assert applied["updated"] == 1
    assert fake_col.docs[0].get("nlp_pipeline") == "cascade"
