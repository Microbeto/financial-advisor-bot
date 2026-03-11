from __future__ import annotations

from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .meta_interface import MetaCombiner, normalize_allocation_vector


class MetaLabelerCombiner(MetaCombiner):
    name = "meta_labeler"

    def __init__(self, random_state: int = 42, size_gain: float = 2.0) -> None:
        self.random_state = int(random_state)
        self.size_gain = float(max(0.1, size_gain))
        self._models: List[RandomForestClassifier] = []
        self._n_bases = 0
        self._last_direction = 0.0

    def train(self, base_predictions: np.ndarray, market_features: np.ndarray, actual_returns: np.ndarray) -> None:
        if base_predictions.ndim != 2:
            raise ValueError("base_predictions must be 2D")

        n_samples, n_bases = base_predictions.shape
        if n_samples == 0 or n_bases == 0:
            raise ValueError("Insufficient training rows for MetaLabeler")

        returns = np.asarray(actual_returns, dtype=float).reshape(-1)
        if returns.shape[0] != n_samples:
            raise ValueError("actual_returns length mismatch")

        features = np.column_stack([base_predictions, market_features])
        outcome_sign = np.where(returns >= 0.0, 1, 0)

        self._models = []
        self._n_bases = n_bases

        for j in range(n_bases):
            base_sign = np.where(base_predictions[:, j] >= 0.5, 1, 0)
            labels = np.where(base_sign == outcome_sign, 1, 0)

            model = RandomForestClassifier(
                n_estimators=180,
                max_depth=6,
                random_state=self.random_state + j,
                class_weight="balanced",
                n_jobs=-1,
            )
            model.fit(features, labels)
            self._models.append(model)

    def predict_confidence(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> np.ndarray:
        if not self._models:
            return np.zeros(1, dtype=float)

        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        market_vec = np.asarray(today_market_features, dtype=float).reshape(-1)
        if base_vec.size != self._n_bases:
            raise ValueError("Base prediction size mismatch")

        feat = np.concatenate([base_vec, market_vec]).reshape(1, -1)
        success_probs = np.array([m.predict_proba(feat)[0, 1] for m in self._models], dtype=float)

        score_vec = (2.0 * base_vec) - 1.0
        direction = float(np.dot(success_probs, score_vec) / max(1e-8, float(np.sum(success_probs))))
        self._last_direction = float(np.clip(direction, -1.0, 1.0))

        raw_conf = float(np.dot(success_probs, np.abs(score_vec)) / max(1e-8, float(np.sum(np.abs(score_vec)))))
        conf = float(np.clip(raw_conf, 0.0, 1.0))
        return np.array([conf], dtype=float)

    def allocate(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
        current_allocation: np.ndarray | None = None,
    ) -> np.ndarray:
        del current_allocation
        confidence = self.predict_confidence(today_base_predictions, today_market_features)
        conf = float(confidence[0]) if confidence.size else 0.5
        magnitude = float(np.clip(np.tanh((conf - 0.5) * self.size_gain * 2.0), 0.0, 1.0))
        alloc = magnitude * (1.0 if self._last_direction >= 0.0 else -1.0)
        return normalize_allocation_vector(np.array([alloc], dtype=float))
