from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .meta_interface import MetaCombiner


@dataclass
class TournamentResult:
    winner_name: str
    winner_model: MetaCombiner
    competitor_stats: Dict[str, Dict[str, float]]
    daily_returns: Dict[str, List[float]]


def _sharpe_ratio(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float((np.mean(returns) / std) * np.sqrt(252.0))


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0.0, (equity / peaks) - 1.0, 0.0)
    return float(np.min(drawdowns))


def run_walk_forward_tournament(
    competitor_factories: Sequence[Callable[[], MetaCombiner]],
    base_predictions: np.ndarray,
    market_features: np.ndarray,
    actual_returns: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> TournamentResult:
    competitor_names = [factory().name for factory in competitor_factories]
    return_log: Dict[str, List[float]] = {name: [] for name in competitor_names}

    for tr_idx, te_idx in splits:
        x_train_base = base_predictions[tr_idx]
        x_train_market = market_features[tr_idx]
        y_train_ret = actual_returns[tr_idx]

        fold_models: List[MetaCombiner] = []
        fold_names: List[str] = []
        for factory in competitor_factories:
            model = factory()
            model.train(x_train_base, x_train_market, y_train_ret)
            fold_models.append(model)
            fold_names.append(model.name)

        for row_idx in te_idx:
            base_row = base_predictions[row_idx]
            market_row = market_features[row_idx]
            actual_ret = float(actual_returns[row_idx])
            for model, name in zip(fold_models, fold_names):
                alloc = float(model.allocate(base_row, market_row))
                ret = alloc * actual_ret
                return_log[name].append(float(ret))

    stats: Dict[str, Dict[str, float]] = {}
    for name, vals in return_log.items():
        arr = np.asarray(vals, dtype=float)
        stats[name] = {
            "sharpe": _sharpe_ratio(arr),
            "max_drawdown": _max_drawdown(arr),
            "mean_return": float(np.mean(arr)) if arr.size else 0.0,
            "volatility": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "sample_count": float(arr.size),
        }

    ranked = sorted(
        stats.items(),
        key=lambda kv: (float(kv[1].get("sharpe", 0.0)), -abs(float(kv[1].get("max_drawdown", 0.0)))),
        reverse=True,
    )
    winner_name = ranked[0][0] if ranked else competitor_names[0]

    winner_model: MetaCombiner | None = None
    for factory in competitor_factories:
        model = factory()
        if model.name == winner_name:
            winner_model = model
            break
    if winner_model is None:
        winner_model = competitor_factories[0]()

    winner_model.train(base_predictions, market_features, actual_returns)
    return TournamentResult(
        winner_name=winner_name,
        winner_model=winner_model,
        competitor_stats=stats,
        daily_returns=return_log,
    )
