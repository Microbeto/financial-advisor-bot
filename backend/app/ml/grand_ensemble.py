from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .meta_interface import MetaCombiner, normalize_allocation_vector, scalar_allocation


class GrandEnsembleCombiner(MetaCombiner):
    # Meta-ensemble of sub-combiners; blends their allocations via performance-weighted softmax
    name = "grand_ensemble"

    def __init__(
        self,
        sub_factories: List[Callable[[], MetaCombiner]],
        ewma_span: int = 20,
    ) -> None:
        self._sub_models: List[MetaCombiner] = [f() for f in sub_factories]  # Instantiate all sub-combiners
        n = len(self._sub_models)
        self._ewma_alpha = 2.0 / (float(max(2, int(ewma_span))) + 1.0)  # EWMA decay coefficient
        self._perf_ewma = np.zeros(n, dtype=float)  # Running EWMA of realized PnL per sub-model
        self._last_allocations = np.zeros(n, dtype=float)  # Cache sub-allocations for step_update feedback

    def _performance_weights(self) -> np.ndarray:
        # Softmax over EWMA performance scores → dynamic sub-model blending weights
        scores = self._perf_ewma - np.max(self._perf_ewma)  # Subtract max for numerical stability
        exp_scores = np.exp(scores)
        return exp_scores / float(np.sum(exp_scores))  # Normalize to probability simplex

    def train(
        self,
        base_predictions: np.ndarray,
        market_features: np.ndarray,
        actual_returns: np.ndarray,
    ) -> None:
        # Delegate training to every sub-combiner independently
        for model in self._sub_models:
            model.train(base_predictions, market_features, actual_returns)

    def predict_confidence(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
    ) -> np.ndarray:
        # Weighted average of each sub-combiner's confidence using performance weights
        weights = self._performance_weights()
        confidences = np.array(
            [
                float(np.mean(np.asarray(m.predict_confidence(today_base_predictions, today_market_features), dtype=float)))
                for m in self._sub_models
            ],
            dtype=float,
        )
        blended = float(np.clip(np.dot(weights, confidences), 0.0, 1.0))  # Weighted blend, bounded [0, 1]
        return np.array([blended], dtype=float)

    def allocate(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
        current_allocation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Blend each sub-combiner's allocation using performance-derived weights
        weights = self._performance_weights()
        allocs = np.array(
            [
                scalar_allocation(m.allocate(today_base_predictions, today_market_features, current_allocation))
                for m in self._sub_models
            ],
            dtype=float,
        )
        self._last_allocations = allocs  # Cache for performance attribution in step_update
        blended = float(np.dot(weights, allocs))  # Weighted average allocation
        return normalize_allocation_vector(np.array([blended], dtype=float))

    def step_update(
        self,
        actual_return: np.ndarray,
        realized_base_predictions: Optional[np.ndarray] = None,
        realized_market_features: Optional[np.ndarray] = None,
    ) -> None:
        # Propagate feedback to sub-combiners and update EWMA performance tracking
        ret_arr = np.asarray(actual_return, dtype=float).reshape(-1)
        if ret_arr.size == 0:
            return
        ret = float(ret_arr[0])
        for i, model in enumerate(self._sub_models):
            model.step_update(actual_return, realized_base_predictions, realized_market_features)  # Forward update
            realized_pnl = self._last_allocations[i] * ret  # Attribution: sub-model's contribution to PnL
            self._perf_ewma[i] = (
                self._ewma_alpha * realized_pnl + (1.0 - self._ewma_alpha) * self._perf_ewma[i]
            )  # EWMA update: recent performance weighted more heavily
