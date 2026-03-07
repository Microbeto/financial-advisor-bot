from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.ml_workflow import (
    DataValidationError,
    _DailyNewsFeatures,
    _data_completeness_from_news_coverage,
    _rank_models,
    _select_prune_candidates,
    _sentiment_score_text,
    _technical_features,
    _validate_news_coverage_or_raise,
    _weighted_score,
    get_tournament_competitor_stats,
)


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
