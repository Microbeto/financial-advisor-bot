from __future__ import annotations

from typing import Dict

import numpy as np

from .meta_interface import MetaCombiner, normalize_allocation_vector


class RegimeSwitcherCombiner(MetaCombiner):
    name = "regime_switcher"

    def __init__(self, online_rate: float = 0.1) -> None:
        self._vol_threshold = 0.0
        self._best_by_regime: Dict[str, int] = {"low": 0, "high": 0}
        self._regime_rewards: Dict[str, np.ndarray] = {"low": np.array([], dtype=float), "high": np.array([], dtype=float)}
        self._n_bases = 0
        self._online_rate = float(np.clip(online_rate, 1e-3, 1.0))
        self._last_base = np.array([], dtype=float)
        self._last_market = np.array([], dtype=float)

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
        self._regime_rewards = {
            "low": np.zeros(n_bases, dtype=float),
            "high": np.zeros(n_bases, dtype=float),
        }
        for regime_name, mask in {
            "low": vol < self._vol_threshold,
            "high": vol >= self._vol_threshold,
        }.items():
            if int(np.sum(mask)) == 0:
                self._best_by_regime[regime_name] = 0
                continue

            avg_reward = np.mean(base_rewards[mask], axis=0)
            self._regime_rewards[regime_name] = np.asarray(avg_reward, dtype=float)
            self._best_by_regime[regime_name] = int(np.argmax(avg_reward))

    def predict_confidence(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> np.ndarray:
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        market_vec = np.asarray(today_market_features, dtype=float).reshape(-1)

        if base_vec.size == 0:
            return np.zeros(1, dtype=float)
        if self._n_bases and base_vec.size != self._n_bases:
            raise ValueError("Base prediction size mismatch")

        vol = float(market_vec[3]) if market_vec.size > 3 else (float(market_vec[0]) if market_vec.size else 0.0)
        regime = "high" if vol >= self._vol_threshold else "low"

        idx = int(self._best_by_regime.get(regime, 0))
        idx = max(0, min(idx, base_vec.size - 1))
        self._last_base = np.asarray(base_vec, dtype=float)
        self._last_market = np.asarray(market_vec, dtype=float)
        confidence = float(np.clip(abs((2.0 * base_vec[idx]) - 1.0), 0.0, 1.0))
        return np.array([confidence], dtype=float)

    def allocate(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
        current_allocation: np.ndarray | None = None,
    ) -> np.ndarray:
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        market_vec = np.asarray(today_market_features, dtype=float).reshape(-1)
        if base_vec.size == 0:
            return np.zeros(1, dtype=float)

        vol = float(market_vec[3]) if market_vec.size > 3 else (float(market_vec[0]) if market_vec.size else 0.0)
        regime = "high" if vol >= self._vol_threshold else "low"
        idx = int(self._best_by_regime.get(regime, 0))
        idx = max(0, min(idx, base_vec.size - 1))
        target = float(np.clip((2.0 * base_vec[idx]) - 1.0, -1.0, 1.0))

        prev = 0.0
        if current_allocation is not None:
            prev_vec = np.asarray(current_allocation, dtype=float).reshape(-1)
            if prev_vec.size:
                prev = float(prev_vec[0])

        adjusted = prev + ((1.0 - self._online_rate) * (target - prev))
        self._last_base = np.asarray(base_vec, dtype=float)
        self._last_market = np.asarray(market_vec, dtype=float)
        return normalize_allocation_vector(np.array([adjusted], dtype=float))

    def step_update(
        self,
        actual_return: np.ndarray,
        realized_base_predictions: np.ndarray | None = None,
        realized_market_features: np.ndarray | None = None,
    ) -> None:
        ret_vec = np.asarray(actual_return, dtype=float).reshape(-1)
        if ret_vec.size == 0:
            return

        base_vec = np.asarray(realized_base_predictions, dtype=float).reshape(-1) if realized_base_predictions is not None else self._last_base
        market_vec = np.asarray(realized_market_features, dtype=float).reshape(-1) if realized_market_features is not None else self._last_market
        if base_vec.size == 0:
            return

        vol = float(market_vec[3]) if market_vec.size > 3 else (float(market_vec[0]) if market_vec.size else 0.0)
        regime = "high" if vol >= self._vol_threshold else "low"

        rewards = ((2.0 * base_vec) - 1.0) * float(ret_vec[0])
        current = self._regime_rewards.get(regime)
        if current is None or current.size != rewards.size:
            current = np.zeros(rewards.size, dtype=float)
        blended = (1.0 - self._online_rate) * current + self._online_rate * rewards
        self._regime_rewards[regime] = np.asarray(blended, dtype=float)
        self._best_by_regime[regime] = int(np.argmax(blended))
