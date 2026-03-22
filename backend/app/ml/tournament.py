# Walk-forward tournament framework for comparing and ranking meta-combiner models.
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .meta_interface import MetaCombiner, scalar_allocation


DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_ANNUAL_RISK_FREE_RATE = 0.04
DEFAULT_MAX_WINNER_DRAWDOWN_ABS = 0.30


@dataclass
class TournamentResult:
    winner_name: str
    winner_model: MetaCombiner
    competitor_stats: Dict[str, Dict[str, float]]
    daily_returns: Dict[str, List[float]]


def _cost_rate_from_bps(cost_bps: float) -> float:
    bps = float(max(0.0, cost_bps))
    return float(bps / 10000.0)


def _sharpe_ratio(returns: np.ndarray, annual_risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE) -> float:
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    daily_rf = float(annual_risk_free_rate / 252.0)
    excess_mean = float(np.mean(returns) - daily_rf)
    return float((excess_mean / std) * np.sqrt(252.0))


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0.0, (equity / peaks) - 1.0, 0.0)
    return float(np.min(drawdowns))


def _is_stateful_competitor(model: MetaCombiner) -> bool:
    return type(model).step_update is not MetaCombiner.step_update


def run_walk_forward_tournament(
    competitor_factories: Sequence[Callable[[], MetaCombiner]],
    base_predictions: np.ndarray,
    market_features: np.ndarray,
    actual_returns: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    annual_risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE,
) -> TournamentResult:
    # Initialize competitors and detect which ones keep state across folds.
    seeded_models = [factory() for factory in competitor_factories]
    competitor_names = [model.name for model in seeded_models]
    stateful_flags = {
        model.name: _is_stateful_competitor(model)
        for model in seeded_models
    }
    persistent_models = {
        model.name: model
        for model in seeded_models
        if stateful_flags.get(model.name, False)
    }
    persistent_allocations: Dict[str, np.ndarray] = {
        name: np.zeros(1, dtype=float)
        for name in competitor_names
        if stateful_flags.get(name, False)
    }
    initialized_stateful: set[str] = set()
    return_log: Dict[str, List[float]] = {name: [] for name in competitor_names}
    trading_cost = _cost_rate_from_bps(transaction_cost_bps)

    # Process folds in chronological order to preserve temporal causality.
    ordered_splits = sorted(
        splits,
        key=lambda pair: int(pair[1][0]) if len(pair[1]) else int(pair[0][0]),
    )

    # Train/evaluate all competitors fold by fold in walk-forward mode.
    for tr_idx, te_idx in ordered_splits:
        x_train_base = base_predictions[tr_idx]
        x_train_market = market_features[tr_idx]
        y_train_ret = actual_returns[tr_idx]

        fold_models: List[MetaCombiner] = []
        fold_names: List[str] = []
        current_allocations: Dict[str, np.ndarray] = {}
        # Reuse stateful models, but retrain stateless models each fold.
        for seeded_model, factory in zip(seeded_models, competitor_factories):
            name = seeded_model.name
            if stateful_flags.get(name, False):
                model = persistent_models[name]
                if name not in initialized_stateful:
                    model.train(x_train_base, x_train_market, y_train_ret)
                    initialized_stateful.add(name)
                current_allocations[name] = np.asarray(
                    persistent_allocations.get(name, np.zeros(1, dtype=float)),
                    dtype=float,
                ).reshape(-1)
            else:
                model = factory()
                model.train(x_train_base, x_train_market, y_train_ret)
                current_allocations[name] = np.zeros(1, dtype=float)
            fold_models.append(model)
            fold_names.append(name)

        # Score each competitor on every out-of-sample row with transaction costs.
        for row_idx in te_idx:
            base_row = base_predictions[row_idx]
            market_row = market_features[row_idx]
            actual_ret = float(actual_returns[row_idx])
            for model, name in zip(fold_models, fold_names):
                prev_alloc_vec = np.asarray(
                    current_allocations.get(name, np.zeros(1, dtype=float)),
                    dtype=float,
                ).reshape(-1)
                prev_alloc = scalar_allocation(prev_alloc_vec)
                alloc_vec = model.allocate(
                    base_row,
                    market_row,
                    current_allocation=prev_alloc_vec,
                )
                current_allocations[name] = np.asarray(alloc_vec, dtype=float).reshape(-1)
                alloc = scalar_allocation(alloc_vec)
                turnover = abs(alloc - prev_alloc)
                ret = (alloc * actual_ret) - (trading_cost * turnover)
                return_log[name].append(float(ret))
                model.step_update(
                    actual_return=np.array([actual_ret], dtype=float),
                    realized_base_predictions=base_row,
                    realized_market_features=market_row,
                )

        # Carry forward current allocations only for stateful competitors.
        for name in fold_names:
            if stateful_flags.get(name, False):
                persistent_allocations[name] = np.asarray(
                    current_allocations.get(name, np.zeros(1, dtype=float)),
                    dtype=float,
                ).reshape(-1)

    # Aggregate per-competitor risk/return statistics from daily walk-forward returns.
    stats: Dict[str, Dict[str, float]] = {}
    for name, vals in return_log.items():
        arr = np.asarray(vals, dtype=float)
        stats[name] = {
            "sharpe": _sharpe_ratio(arr, annual_risk_free_rate=annual_risk_free_rate),
            "max_drawdown": _max_drawdown(arr),
            "mean_return": float(np.mean(arr)) if arr.size else 0.0,
            "volatility": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "sample_count": float(arr.size),
        }

    # Pick winner by Sharpe with drawdown-aware tie-break, preferring eligible competitors.
    eligible = [
        kv
        for kv in stats.items()
        if abs(float(kv[1].get("max_drawdown", 0.0))) <= float(DEFAULT_MAX_WINNER_DRAWDOWN_ABS)
    ]
    ranked_pool = eligible if eligible else list(stats.items())
    ranked = sorted(
        ranked_pool,
        key=lambda kv: (float(kv[1].get("sharpe", 0.0)), -abs(float(kv[1].get("max_drawdown", 0.0)))),
        reverse=True,
    )
    winner_name = ranked[0][0] if ranked else competitor_names[0]

    # Resolve winner instance (stateful carry-over if available, otherwise retrain fresh).
    winner_model: MetaCombiner | None = None
    if stateful_flags.get(winner_name, False):
        winner_model = persistent_models.get(winner_name)

    if winner_model is None:
        for factory in competitor_factories:
            model = factory()
            if model.name == winner_name:
                winner_model = model
                break
    if winner_model is None:
        winner_model = competitor_factories[0]()

    if not stateful_flags.get(winner_name, False):
        winner_model.train(base_predictions, market_features, actual_returns)

    return TournamentResult(
        winner_name=winner_name,
        winner_model=winner_model,
        competitor_stats=stats,
        daily_returns=return_log,
    )
