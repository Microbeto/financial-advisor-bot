from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class MetaCombiner(ABC):
    name: str = "meta_combiner"

    @abstractmethod
    def train(
        self,
        base_predictions: np.ndarray,
        market_features: np.ndarray,
        actual_returns: np.ndarray,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_confidence(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def allocate(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
        current_allocation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def step_update(
        self,
        actual_return: np.ndarray,
        realized_base_predictions: Optional[np.ndarray] = None,
        realized_market_features: Optional[np.ndarray] = None,
    ) -> None:
        del actual_return, realized_base_predictions, realized_market_features


def scalar_allocation(allocation: np.ndarray) -> float:
    vec = np.asarray(allocation, dtype=float).reshape(-1)
    if vec.size == 0:
        return 0.0
    return float(np.clip(vec[0], -1.0, 1.0))


def normalize_allocation_vector(allocation: np.ndarray) -> np.ndarray:
    vec = np.asarray(allocation, dtype=float).reshape(-1)
    if vec.size == 0:
        return np.zeros(1, dtype=float)
    return np.clip(vec, -1.0, 1.0)
