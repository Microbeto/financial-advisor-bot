from __future__ import annotations

from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .meta_interface import MetaCombiner


class MetaLabelerCombiner(MetaCombiner):
    name = "meta_labeler"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = int(random_state)
        self._models: List[RandomForestClassifier] = []
        self._n_bases = 0

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

    def allocate(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> float:
        if not self._models:
            return 0.0

        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        market_vec = np.asarray(today_market_features, dtype=float).reshape(-1)
        if base_vec.size != self._n_bases:
            raise ValueError("Base prediction size mismatch")

        feat = np.concatenate([base_vec, market_vec]).reshape(1, -1)
        success_probs = np.array([m.predict_proba(feat)[0, 1] for m in self._models], dtype=float)

        score_vec = (2.0 * base_vec) - 1.0
        weighted = float(np.dot(success_probs, score_vec) / max(1e-8, float(np.sum(success_probs))))
        return float(np.clip(weighted, -1.0, 1.0))
