from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ml.meta_labeler import MetaLabelerCombiner
from app.ml.regime_switcher import RegimeSwitcherCombiner
from app.ml.rl_agent import RLAgentCombiner
from app.ml.tournament import run_walk_forward_tournament


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
