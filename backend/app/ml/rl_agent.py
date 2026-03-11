from __future__ import annotations

import numpy as np

from .meta_interface import MetaCombiner, normalize_allocation_vector


class RLAgentCombiner(MetaCombiner):
    name = "rl_agent"

    def __init__(self, learning_rate: float = 0.05, temperature: float = 0.8, transaction_cost_rate: float = 0.05) -> None:
        self.learning_rate = float(max(1e-4, learning_rate))
        self.temperature = float(max(1e-4, temperature))
        self.transaction_cost_rate = float(np.clip(transaction_cost_rate, 0.0, 0.5))
        self._q = np.array([], dtype=float)
        self._weights = np.array([], dtype=float)
        self._last_signal = np.array([], dtype=float)

    def train(self, base_predictions: np.ndarray, market_features: np.ndarray, actual_returns: np.ndarray) -> None:
        if base_predictions.ndim != 2:
            raise ValueError("base_predictions must be 2D")

        n_samples, n_bases = base_predictions.shape
        if n_samples == 0 or n_bases == 0:
            raise ValueError("Insufficient training rows for RLAgent")

        returns = np.asarray(actual_returns, dtype=float).reshape(-1)
        if returns.shape[0] != n_samples:
            raise ValueError("actual_returns length mismatch")

        self._q = np.zeros(n_bases, dtype=float)
        for i in range(n_samples):
            signal = (2.0 * np.asarray(base_predictions[i], dtype=float)) - 1.0
            reward = signal * float(returns[i])
            self._q += self.learning_rate * (reward - self._q)

        scaled = self._q / self.temperature
        scaled -= float(np.max(scaled))
        exp_scores = np.exp(scaled)
        denom = float(np.sum(exp_scores))
        if denom <= 0.0:
            self._weights = np.ones(n_bases, dtype=float) / float(n_bases)
        else:
            self._weights = exp_scores / denom

    def predict_confidence(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> np.ndarray:
        del today_market_features
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        if base_vec.size == 0:
            return np.zeros(1, dtype=float)

        signal = (2.0 * base_vec) - 1.0
        self._last_signal = np.asarray(signal, dtype=float)

        if self._weights.size != signal.size:
            weights = np.ones(signal.size, dtype=float) / float(signal.size)
        else:
            weights = self._weights

        confidence = float(np.clip(np.sum(weights * np.abs(signal)), 0.0, 1.0))
        return np.array([confidence], dtype=float)

    def allocate(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
        current_allocation: np.ndarray | None = None,
    ) -> np.ndarray:
        del today_market_features
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        if base_vec.size == 0:
            return np.zeros(1, dtype=float)

        signal = (2.0 * base_vec) - 1.0
        self._last_signal = np.asarray(signal, dtype=float)
        if self._weights.size != signal.size:
            weights = np.ones(signal.size, dtype=float) / float(signal.size)
        else:
            weights = self._weights

        target = float(np.clip(np.dot(weights, signal), -1.0, 1.0))

        prev = 0.0
        if current_allocation is not None:
            prev_vec = np.asarray(current_allocation, dtype=float).reshape(-1)
            if prev_vec.size:
                prev = float(np.clip(prev_vec[0], -1.0, 1.0))

        turnover = abs(target - prev)
        trade_fraction = max(0.0, 1.0 - (self.transaction_cost_rate * turnover))
        adjusted = prev + trade_fraction * (target - prev)
        return normalize_allocation_vector(np.array([adjusted], dtype=float))

    def step_update(
        self,
        actual_return: np.ndarray,
        realized_base_predictions: np.ndarray | None = None,
        realized_market_features: np.ndarray | None = None,
    ) -> None:
        del realized_market_features
        ret_vec = np.asarray(actual_return, dtype=float).reshape(-1)
        if ret_vec.size == 0:
            return

        if realized_base_predictions is not None:
            signal = (2.0 * np.asarray(realized_base_predictions, dtype=float).reshape(-1)) - 1.0
        else:
            signal = self._last_signal

        if signal.size == 0:
            return

        if self._q.size != signal.size:
            self._q = np.zeros(signal.size, dtype=float)

        reward = signal * float(ret_vec[0])
        self._q += self.learning_rate * (reward - self._q)

        scaled = self._q / self.temperature
        scaled -= float(np.max(scaled))
        exp_scores = np.exp(scaled)
        denom = float(np.sum(exp_scores))
        if denom <= 0.0:
            self._weights = np.ones(signal.size, dtype=float) / float(signal.size)
        else:
            self._weights = exp_scores / denom
