# Meta-labeler combiner using random forest to calibrate consensus predictions and confidence signals.
from __future__ import annotations

import math

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from .meta_interface import MetaCombiner, normalize_allocation_vector


class MetaLabelerCombiner(MetaCombiner):
    name = "meta_labeler"
    supports_cross_sectional = True

    def __init__(
        self,
        random_state: int = 42,
        side_threshold: float = 0.02,
        max_gross_exposure: float = 1.0,
    ) -> None:
        # Initialize model parameters and internal state variables
        self.random_state = int(random_state)
        self.side_threshold = float(np.clip(side_threshold, 0.0, 0.25))
        self.max_gross_exposure = float(max(0.0, max_gross_exposure))
        self._meta_model: CalibratedClassifierCV | RandomForestClassifier | None = None
        self._raw_model: RandomForestClassifier | None = None
        self._fallback_confidence = 0.5
        self._n_bases = 0
        self._last_side = np.zeros(1, dtype=float)
        self._meta_feature_importances = np.array([], dtype=float)

    def _consensus_side(self, consensus_prob: np.ndarray) -> np.ndarray:
        # Convert consensus probabilities to directional signals (1/-1/0 for long/short/abstain)
        p = np.asarray(consensus_prob, dtype=float)
        side = np.where(p >= 0.5, 1.0, -1.0)
        abstain = np.abs(p - 0.5) < self.side_threshold
        side[abstain] = 0.0
        return side

    def train(self, base_predictions: np.ndarray, market_features: np.ndarray, actual_returns: np.ndarray) -> None:
        # Train meta-model to assess and calibrate confidence in consensus predictions
        if base_predictions.ndim != 2:
            raise ValueError("base_predictions must be 2D")

        n_samples, n_bases = base_predictions.shape
        if n_samples == 0 or n_bases == 0:
            raise ValueError("Insufficient training rows for MetaLabeler")

        returns = np.asarray(actual_returns, dtype=float).reshape(-1)
        if returns.shape[0] != n_samples:
            raise ValueError("actual_returns length mismatch")

        market = np.asarray(market_features, dtype=float)
        if market.ndim != 2 or market.shape[0] != n_samples:
            raise ValueError("market_features shape mismatch")

        # Compute consensus probability and outcome labels for training
        consensus_prob = np.mean(base_predictions, axis=1)
        side = self._consensus_side(consensus_prob)
        outcome_sign = np.where(returns >= 0.0, 1.0, -1.0)
        labels = np.where(side == 0.0, 0, np.where(side == outcome_sign, 1, 0)).astype(int)

        # Stack features (direction signal, consensus probability, and market features)
        features = np.column_stack([side, consensus_prob, market])

        self._n_bases = n_bases
        self._meta_model = None
        self._raw_model = None
        self._meta_feature_importances = np.array([], dtype=float)

        # Check if sufficient class variety exists for training
        uniq, counts = np.unique(labels, return_counts=True)
        if uniq.size < 2:
            self._fallback_confidence = float(np.clip(float(uniq[0]) if uniq.size == 1 else 0.5, 0.0, 1.0))
            return

        # Train raw RandomForest model and extract feature importances
        min_class_count = int(np.min(counts)) if counts.size else 0

        raw_model = RandomForestClassifier(
            n_estimators=220,
            max_depth=6,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        raw_model.fit(features, labels)
        self._raw_model = raw_model
        self._meta_feature_importances = np.asarray(raw_model.feature_importances_, dtype=float).reshape(-1)

        # Apply calibration if sufficient samples exist, otherwise use raw model
        cv_folds = min(3, min_class_count)
        if cv_folds >= 2:
            calibrated = CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=220,
                    max_depth=6,
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                method="sigmoid",
                cv=int(cv_folds),
            )
            calibrated.fit(features, labels)
            self._meta_model = calibrated
        else:
            self._meta_model = raw_model

    def predict_confidence(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> np.ndarray:
        # Generate calibrated confidence scores for today's predictions based on meta-model
        base_arr = np.asarray(today_base_predictions, dtype=float)
        market_arr = np.asarray(today_market_features, dtype=float)

        if base_arr.ndim == 1:
            base_mat = base_arr.reshape(1, -1)
        elif base_arr.ndim == 2:
            base_mat = base_arr
        else:
            raise ValueError("today_base_predictions must be 1D or 2D")

        # Normalize market features to 2D array
        if market_arr.ndim == 1:
            market_mat = market_arr.reshape(1, -1)
        elif market_arr.ndim == 2:
            market_mat = market_arr
        else:
            raise ValueError("today_market_features must be 1D or 2D")

        # Validate input dimensions match trained model
        if base_mat.shape[0] != market_mat.shape[0]:
            raise ValueError("Row count mismatch between base predictions and market features")
        if self._n_bases and base_mat.shape[1] != self._n_bases:
            raise ValueError("Base prediction size mismatch")

        # Build feature vector and retrieve confidence from meta-model
        consensus_prob = np.mean(base_mat, axis=1)
        side = self._consensus_side(consensus_prob)
        self._last_side = np.asarray(side, dtype=float)

        feat = np.column_stack([side, consensus_prob, market_mat])
        if self._meta_model is None:
            conf = np.full(feat.shape[0], self._fallback_confidence, dtype=float)
        else:
            conf = np.asarray(self._meta_model.predict_proba(feat)[:, 1], dtype=float)

        return np.clip(conf, 0.0, 1.0)

    def allocate(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
        current_allocation: np.ndarray | None = None,
    ) -> np.ndarray:
        # Convert confidence scores and signals into normalized portfolio allocation weights
        del current_allocation
        confidence = np.asarray(self.predict_confidence(today_base_predictions, today_market_features), dtype=float).reshape(-1)
        if confidence.size == 0:
            return np.zeros(1, dtype=float)

        # Transform confidence to magnitude using error function (maps to [0,1])  
        p = np.clip(confidence, 1e-6, 1.0 - 1e-6)
        z = (p - 0.5) / np.sqrt(p * (1.0 - p))
        phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        magnitude = np.clip((2.0 * phi) - 1.0, 0.0, 1.0)

        # Combine direction signal with confidence magnitude
        side = np.asarray(self._last_side, dtype=float).reshape(-1)
        if side.size != magnitude.size:
            side = np.ones_like(magnitude, dtype=float)
        raw_alloc = side * magnitude

        # Scale allocations to respect maximum gross exposure constraint
        if raw_alloc.size > 1:
            gross = float(np.sum(np.abs(raw_alloc)))
            if gross > 1e-12 and self.max_gross_exposure > 0.0:
                weights = (raw_alloc / gross) * self.max_gross_exposure
            else:
                weights = np.zeros_like(raw_alloc)
            return normalize_allocation_vector(weights)

        # Handle single-asset case by clipping to max exposure
        single = float(raw_alloc[0]) if raw_alloc.size else 0.0
        single = float(np.clip(single, -self.max_gross_exposure, self.max_gross_exposure))
        return normalize_allocation_vector(np.array([single], dtype=float))
