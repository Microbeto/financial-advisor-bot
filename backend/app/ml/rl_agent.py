from __future__ import annotations

import numpy as np

from .meta_interface import MetaCombiner


class RLAgentCombiner(MetaCombiner):
    name = "rl_agent"

    def __init__(self, learning_rate: float = 0.05, temperature: float = 0.8) -> None:
        self.learning_rate = float(max(1e-4, learning_rate))
        self.temperature = float(max(1e-4, temperature))
        self._q = np.array([], dtype=float)
        self._weights = np.array([], dtype=float)

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

    def allocate(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> float:
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        if base_vec.size == 0:
            return 0.0

        signal = (2.0 * base_vec) - 1.0

        if self._weights.size != signal.size:
            weights = np.ones(signal.size, dtype=float) / float(signal.size)
        else:
            weights = self._weights

        alloc = float(np.dot(weights, signal))
        return float(np.clip(alloc, -1.0, 1.0))
