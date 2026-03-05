from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MetaCombiner(ABC):
    name: str = "meta_combiner"

    @abstractmethod
    def train(self, base_predictions: np.ndarray, market_features: np.ndarray, actual_returns: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def allocate(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> float:
        raise NotImplementedError
