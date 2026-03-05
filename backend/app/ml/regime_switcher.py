from __future__ import annotations

from typing import Dict

import numpy as np

from .meta_interface import MetaCombiner


class RegimeSwitcherCombiner(MetaCombiner):
    name = "regime_switcher"

    def __init__(self) -> None:
        self._vol_threshold = 0.0
        self._best_by_regime: Dict[str, int] = {"low": 0, "high": 0}
        self._n_bases = 0

    def train(self, base_predictions: np.ndarray, market_features: np.ndarray, actual_returns: np.ndarray) -> None:
        if base_predictions.ndim != 2:
            raise ValueError("base_predictions must be 2D")

        n_samples, n_bases = base_predictions.shape
        if n_samples == 0 or n_bases == 0:
            raise ValueError("Insufficient training rows for RegimeSwitcher")

        if market_features.ndim != 2 or market_features.shape[0] != n_samples:
            raise ValueError("market_features shape mismatch")

        returns = np.asarray(actual_returns, dtype=float).reshape(-1)
        if returns.shape[0] != n_samples:
            raise ValueError("actual_returns length mismatch")

        vol_idx = 3 if market_features.shape[1] > 3 else 0
        vol = np.asarray(market_features[:, vol_idx], dtype=float)
        self._vol_threshold = float(np.median(vol))

        base_signals = (2.0 * base_predictions) - 1.0
        base_rewards = base_signals * returns[:, None]

        self._n_bases = n_bases
        self._best_by_regime = {}
        for regime_name, mask in {
            "low": vol < self._vol_threshold,
            "high": vol >= self._vol_threshold,
        }.items():
            if int(np.sum(mask)) == 0:
                self._best_by_regime[regime_name] = 0
                continue

            avg_reward = np.mean(base_rewards[mask], axis=0)
            self._best_by_regime[regime_name] = int(np.argmax(avg_reward))

    def allocate(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> float:
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        market_vec = np.asarray(today_market_features, dtype=float).reshape(-1)

        if base_vec.size == 0:
            return 0.0
        if self._n_bases and base_vec.size != self._n_bases:
            raise ValueError("Base prediction size mismatch")

        vol = float(market_vec[3]) if market_vec.size > 3 else (float(market_vec[0]) if market_vec.size else 0.0)
        regime = "high" if vol >= self._vol_threshold else "low"

        idx = int(self._best_by_regime.get(regime, 0))
        idx = max(0, min(idx, base_vec.size - 1))
        signal = float((2.0 * base_vec[idx]) - 1.0)
        return float(np.clip(signal, -1.0, 1.0))
