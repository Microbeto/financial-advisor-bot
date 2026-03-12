from __future__ import annotations
from typing import Dict
import numpy as np
from .meta_interface import MetaCombiner, normalize_allocation_vector


class RegimeSwitcherCombiner(MetaCombiner):
    # Softmax ensemble with continuous regimes, EWMA tracking, and hysteresis filtering
    name = "regime_switcher"

    def __init__(
        self,
        online_rate: float = 0.1,
        vol_feature_idx: int = 3,
        rolling_window_days: int = 60,
        ewma_span: int = 20,
        softmax_temperature: float = 1.0,
        hysteresis_threshold: float = 0.01,
    ) -> None:
        # Volatility feature index configuration
        self._vol_feature_idx = int(vol_feature_idx)
        
        # Rolling window for regime boundary detection (avoids lookahead bias)
        self._vol_threshold = 0.0  # Median volatility from window
        self._vol_rolling_mean = 0.0  # Mean for z-score normalization
        self._vol_rolling_std = 1.0  # Std dev for z-score normalization
        self._rolling_window_size = max(1, int(rolling_window_days))  # Window length
        
        # Regime rewards tracking: historical means and EWMA for online updates
        self._regime_rewards: Dict[str, np.ndarray] = {
            "low": np.array([], dtype=float),
            "high": np.array([], dtype=float),
        }
        self._ewma_rewards: Dict[str, np.ndarray] = {
            "low": np.array([], dtype=float),
            "high": np.array([], dtype=float),
        }
        self._ewma_span = max(2, int(ewma_span))  # Memory window for EWMA
        self._ewma_alpha = 2.0 / (float(self._ewma_span) + 1.0)  # Decay coefficient
        
        # Softmax temperature and hysteresis controls
        self._softmax_temperature = float(np.clip(softmax_temperature, 0.1, 10.0))
        self._last_model_weights = None
        self._hysteresis_threshold = float(np.clip(hysteresis_threshold, 0.0, 0.1))
        self._last_regime_score = 0.5
        
        # Tracking for step_update() calls
        self._n_bases = 0  # Number of base models
        self._online_rate = float(np.clip(online_rate, 1e-3, 1.0))  # Allocation smoothing
        self._last_base = np.array([], dtype=float)  # Previous predictions cache
        self._last_market = np.array([], dtype=float)  # Previous features cache
        self._sample_count = 0  # Number of training samples

    def _extract_volatility(self, market_vec: np.ndarray) -> float:
        # Extract volatility by configured index with safe bounds checking
        market_vec = np.asarray(market_vec, dtype=float).reshape(-1)
        if market_vec.size == 0:  # Handle empty input
            return 0.0
        idx = min(self._vol_feature_idx, market_vec.size - 1)  # Clamp index to array bounds
        return float(market_vec[idx])  # Return volatility value

    def _calculate_regime_score(self, vol: float) -> float:
        # Z-score normalized volatility mapped to [0, 1] via sigmoid
        if self._vol_rolling_std <= 1e-8:  # Fallback if std dev is near zero
            return 1.0 if vol >= self._vol_threshold else 0.0  # Return binary fallback
        z_score = (vol - self._vol_rolling_mean) / self._vol_rolling_std  # Normalize to std devs
        regime_score = 1.0 / (1.0 + np.exp(-z_score))  # Sigmoid maps z-score to (0, 1)
        return float(np.clip(regime_score, 0.0, 1.0))  # Ensure strict bounds

    def _softmax_weights(self, rewards: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        # Convert rewards to normalized softmax weights with temperature control
        rewards = np.asarray(rewards, dtype=float).reshape(-1)
        if rewards.size == 0:  # Handle empty rewards
            return np.array([], dtype=float)
        scaled = rewards / float(temperature)  # Temperature-scaled rewards
        scaled = scaled - np.max(scaled)  # Subtract max for numerical stability
        exp_scaled = np.exp(scaled)  # Exponential transformation
        weights = exp_scaled / np.sum(exp_scaled)  # Normalize to probabilities
        return np.asarray(weights, dtype=float)  # Return probability weights

    def _apply_hysteresis(self, new_score: float, threshold: float = 0.01) -> float:
        # Suppress regime changes smaller than threshold to reduce transaction costs
        if abs(new_score - self._last_regime_score) < float(threshold):
            return self._last_regime_score
        return float(new_score)

    def train(self, base_predictions: np.ndarray, market_features: np.ndarray, actual_returns: np.ndarray) -> None:
        # Train on historical data with rolling window to avoid lookahead bias
        if base_predictions.ndim != 2:  # Validate input shape
            raise ValueError("base_predictions must be 2D")

        n_samples, n_bases = base_predictions.shape  # Extract dimensions
        if n_samples == 0 or n_bases == 0:  # Check minimum data
            raise ValueError("Insufficient training rows for RegimeSwitcher")

        if market_features.ndim != 2 or market_features.shape[0] != n_samples:  # Validate alignment
            raise ValueError("market_features shape mismatch")

        returns = np.asarray(actual_returns, dtype=float).reshape(-1)  # Convert returns to array
        if returns.shape[0] != n_samples:  # Check alignment
            raise ValueError("actual_returns length mismatch")

        # Extract volatility safely by configured index
        vol = np.array(
            [self._extract_volatility(market_features[i, :]) for i in range(n_samples)],  # Per-sample extraction
            dtype=float
        )
        
        # Use rolling window median to avoid lookahead bias
        window_start = max(0, n_samples - self._rolling_window_size)  # Last N samples only
        window_vol = vol[window_start:]  # Extract window
        self._vol_threshold = float(np.median(window_vol)) if len(window_vol) > 0 else 0.0  # Regime boundary
        self._vol_rolling_mean = float(np.mean(window_vol)) if len(window_vol) > 0 else 0.0  # For z-score
        self._vol_rolling_std = float(np.std(window_vol, ddof=1)) if len(window_vol) > 1 else 1.0  # For z-score

        # Compute normalized signals and rewards per sample
        base_signals = (2.0 * base_predictions) - 1.0  # Map [0, 1] → [-1, 1]
        base_rewards = base_signals * returns[:, None]  # Model-agnostic reward (signal × return)
        self._n_bases = n_bases  # Store model count
        self._sample_count = n_samples  # Store sample count

        # Initialize regime rewards and EWMA tracking
        self._regime_rewards = {"low": np.zeros(n_bases, dtype=float), "high": np.zeros(n_bases, dtype=float)}  # Reset means
        self._ewma_rewards = {"low": np.zeros(n_bases, dtype=float), "high": np.zeros(n_bases, dtype=float)}  # Reset EWMA
        
        # Compute mean rewards for each regime as warm start
        for regime_name, mask in {"low": vol < self._vol_threshold, "high": vol >= self._vol_threshold}.items():  # Partition by regime
            if int(np.sum(mask)) > 0:  # Only if regime has samples
                avg_reward = np.mean(base_rewards[mask], axis=0)  # Mean reward per model in regime
                self._regime_rewards[regime_name] = np.asarray(avg_reward, dtype=float)  # Store mean
                self._ewma_rewards[regime_name] = np.asarray(avg_reward, dtype=float)  # Initialize EWMA with mean

    def predict_confidence(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> np.ndarray:
        # Ensemble confidence via softmax-weighted base predictions
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)  # Convert to 1D array
        market_vec = np.asarray(today_market_features, dtype=float).reshape(-1)  # Convert to 1D array

        if base_vec.size == 0:  # Handle empty predictions
            return np.zeros(1, dtype=float)
        if self._n_bases and base_vec.size != self._n_bases:  # Validate input size
            raise ValueError("Base prediction size mismatch")

        vol = self._extract_volatility(market_vec)  # Get volatility
        regime_score = self._calculate_regime_score(vol)  # Continuous regime [0, 1]
        regime_score = self._apply_hysteresis(regime_score, self._hysteresis_threshold)  # Suppress churn
        self._last_regime_score = regime_score  # Cache regime for step_update

        rewards_low = self._ewma_rewards.get("low", np.zeros(base_vec.size, dtype=float))  # Low-vol rewards
        rewards_high = self._ewma_rewards.get("high", np.zeros(base_vec.size, dtype=float))  # High-vol rewards
        blended_rewards = (1.0 - regime_score) * rewards_low + regime_score * rewards_high  # Blend by regime

        model_weights = self._softmax_weights(blended_rewards, self._softmax_temperature)  # Get model weights
        self._last_model_weights = model_weights  # Cache for step_update

        signals = (2.0 * base_vec) - 1.0  # Map [0, 1] → [-1, 1]
        weighted_confidence = float(np.abs(np.dot(model_weights, signals)))  # Dot product with weights
        self._last_base = np.asarray(base_vec, dtype=float)  # Cache predictions
        self._last_market = np.asarray(market_vec, dtype=float)  # Cache features

        confidence = float(np.clip(weighted_confidence, 0.0, 1.0))  # Bound to [0, 1]
        return np.array([confidence], dtype=float)  # Return as array

    def allocate(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
        current_allocation: np.ndarray | None = None,
    ) -> np.ndarray:
        # Allocate via softmax-weighted ensemble with exponential smoothing
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)  # Convert to 1D
        market_vec = np.asarray(today_market_features, dtype=float).reshape(-1)  # Convert to 1D
        
        if base_vec.size == 0:  # Handle empty predictions
            return np.zeros(1, dtype=float)

        vol = self._extract_volatility(market_vec)  # Get volatility
        regime_score = self._calculate_regime_score(vol)  # Continuous regime prob
        regime_score = self._apply_hysteresis(regime_score, self._hysteresis_threshold)  # Suppress churn
        self._last_regime_score = regime_score  # Cache for step_update

        rewards_low = self._ewma_rewards.get("low", np.zeros(base_vec.size, dtype=float))  # Low-vol EWMA
        rewards_high = self._ewma_rewards.get("high", np.zeros(base_vec.size, dtype=float))  # High-vol EWMA
        blended_rewards = (1.0 - regime_score) * rewards_low + regime_score * rewards_high  # Weighted blend

        model_weights = self._softmax_weights(blended_rewards, self._softmax_temperature)  # Model probabilities
        signals = (2.0 * base_vec) - 1.0  # Normalize predictions to [-1, 1]
        target = float(np.clip(np.dot(model_weights, signals), -1.0, 1.0))  # Ensemble signal

        prev = 0.0  # Previous allocation default
        if current_allocation is not None:  # Use previous if available
            prev_vec = np.asarray(current_allocation, dtype=float).reshape(-1)
            if prev_vec.size:  # Extract scalar
                prev = float(prev_vec[0])

        adjusted = prev + (self._online_rate * (target - prev))  # Exponential smoothing
        self._last_base = np.asarray(base_vec, dtype=float)  # Cache predictions
        self._last_market = np.asarray(market_vec, dtype=float)  # Cache features
        self._last_model_weights = model_weights  # Cache weights

        return normalize_allocation_vector(np.array([adjusted], dtype=float))  # Return normalized

    def step_update(
        self,
        actual_return: np.ndarray,
        realized_base_predictions: np.ndarray | None = None,
        realized_market_features: np.ndarray | None = None,
    ) -> None:
        # Update EWMA rewards for both regimes proportional to regime probability
        ret_vec = np.asarray(actual_return, dtype=float).reshape(-1)  # Convert return to array
        if ret_vec.size == 0:  # Skip if empty
            return

        base_vec = (  # Use provided predictions or cached
            np.asarray(realized_base_predictions, dtype=float).reshape(-1)
            if realized_base_predictions is not None
            else self._last_base
        )
        market_vec = (  # Use provided features or cached
            np.asarray(realized_market_features, dtype=float).reshape(-1)
            if realized_market_features is not None
            else self._last_market
        )
        if base_vec.size == 0:  # Skip if no data
            return

        vol = self._extract_volatility(market_vec)  # Get volatility
        regime_score = self._calculate_regime_score(vol)  # Identify regime
        regime_score = self._apply_hysteresis(regime_score, self._hysteresis_threshold)  # Apply hysteresis
        rewards = ((2.0 * base_vec) - 1.0) * float(ret_vec[0])  # Realized reward: signal × return

        # Update both regime EWMAs weighted by continuous regime score
        for regime_name, regime_weight in [("low", 1.0 - regime_score), ("high", regime_score)]:  # Loop each regime
            if regime_weight > 1e-6:  # Only update if regime is probable
                current_ewma = self._ewma_rewards.get(regime_name)  # Get current EWMA
                if current_ewma is None or current_ewma.size != rewards.size:  # Initialize if needed
                    current_ewma = np.zeros(rewards.size, dtype=float)  # Fresh EWMA
                updated_ewma = self._ewma_alpha * rewards + (1.0 - self._ewma_alpha) * current_ewma  # EWMA update
                self._ewma_rewards[regime_name] = np.asarray(updated_ewma, dtype=float)  # Store updated EWMA
