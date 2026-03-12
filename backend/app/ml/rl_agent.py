from __future__ import annotations

import numpy as np

from .meta_interface import MetaCombiner, normalize_allocation_vector


class RLAgentCombiner(MetaCombiner):
    name = "rl_agent"

    def __init__(
        self,
        learning_rate: float = 0.05,
        temperature: float = 0.8,
        transaction_cost_rate: float = 0.05,
        reward_cost_rate: float = 0.001,
        gamma: float = 0.95,
    ) -> None:
        self.learning_rate = float(max(1e-4, learning_rate))
        self.temperature = float(max(1e-4, temperature))
        self.transaction_cost_rate = float(np.clip(transaction_cost_rate, 0.0, 0.5))
        self.reward_cost_rate = float(max(0.0, reward_cost_rate))
        self.gamma = float(np.clip(gamma, 0.0, 0.999))
        self._theta = np.zeros((0, 0), dtype=float)
        self._weights = np.array([], dtype=float)
        self._last_signal = np.array([], dtype=float)
        self._last_market_state = np.array([], dtype=float)
        self._pending_state = np.array([], dtype=float)
        self._pending_signal = np.array([], dtype=float)
        self._pending_reward = np.array([], dtype=float)

    def _state_vector(self, market_features: np.ndarray) -> np.ndarray:
        # Normalize market context and append bias term for linear contextual Q-values.
        vec = np.asarray(market_features, dtype=float).reshape(-1)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if vec.size == 0:
            vec = np.zeros(1, dtype=float)
        return np.concatenate([vec, np.array([1.0], dtype=float)])

    def _ensure_theta(self, n_models: int, state_dim: int) -> None:
        if self._theta.shape == (n_models, state_dim):
            return
        new_theta = np.zeros((n_models, state_dim), dtype=float)
        if self._theta.size:
            r = min(n_models, self._theta.shape[0])
            c = min(state_dim, self._theta.shape[1])
            new_theta[:r, :c] = self._theta[:r, :c]
        self._theta = new_theta

    def _q_values(self, state: np.ndarray) -> np.ndarray:
        if self._theta.size == 0:
            return np.array([], dtype=float)
        return np.asarray(self._theta @ state, dtype=float)

    def _softmax_weights(self, q_values: np.ndarray) -> np.ndarray:
        if q_values.size == 0:
            return np.array([], dtype=float)
        scaled = q_values / self.temperature
        scaled -= float(np.max(scaled))
        exp_scores = np.exp(scaled)
        denom = float(np.sum(exp_scores))
        if denom <= 0.0:
            return np.ones(q_values.size, dtype=float) / float(q_values.size)
        return exp_scores / denom

    def train(self, base_predictions: np.ndarray, market_features: np.ndarray, actual_returns: np.ndarray) -> None:
        if base_predictions.ndim != 2:
            raise ValueError("base_predictions must be 2D")

        n_samples, n_bases = base_predictions.shape
        if n_samples == 0 or n_bases == 0:
            raise ValueError("Insufficient training rows for RLAgent")

        returns = np.asarray(actual_returns, dtype=float).reshape(-1)
        if returns.shape[0] != n_samples:
            raise ValueError("actual_returns length mismatch")

        market = np.asarray(market_features, dtype=float)
        if market.ndim != 2 or market.shape[0] != n_samples:
            raise ValueError("market_features shape mismatch")

        init_state = self._state_vector(market[0])
        self._ensure_theta(n_bases, init_state.size)
        prev_signal = np.array([], dtype=float)

        # TD(0): update theta_i for each action i using contextual bootstrap target.
        for t in range(n_samples):
            state_t = self._state_vector(market[t])
            self._ensure_theta(n_bases, state_t.size)
            signal_t = (2.0 * np.asarray(base_predictions[t], dtype=float).reshape(-1)) - 1.0
            if signal_t.size != n_bases:
                raise ValueError("base_predictions row width mismatch")

            if prev_signal.size == signal_t.size:
                turnover = np.abs(signal_t - prev_signal)
            else:
                turnover = np.zeros(signal_t.size, dtype=float)
            reward_t = (signal_t * float(returns[t])) - (self.reward_cost_rate * turnover)

            if t + 1 < n_samples:
                state_next = self._state_vector(market[t + 1])
                bootstrap = float(np.max(self._q_values(state_next)))
            else:
                bootstrap = 0.0

            q_t = self._q_values(state_t)
            td_target = reward_t + (self.gamma * bootstrap)
            td_error = td_target - q_t
            self._theta += self.learning_rate * np.outer(td_error, state_t)
            prev_signal = signal_t

        last_state = self._state_vector(market[-1])
        self._weights = self._softmax_weights(self._q_values(last_state))
        self._last_signal = (2.0 * np.asarray(base_predictions[-1], dtype=float).reshape(-1)) - 1.0
        self._last_market_state = np.asarray(last_state, dtype=float)
        self._pending_state = np.array([], dtype=float)
        self._pending_signal = np.array([], dtype=float)
        self._pending_reward = np.array([], dtype=float)

    def predict_confidence(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> np.ndarray:
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        if base_vec.size == 0:
            return np.zeros(1, dtype=float)

        signal = (2.0 * base_vec) - 1.0
        self._last_signal = np.asarray(signal, dtype=float)
        state = self._state_vector(today_market_features)
        self._last_market_state = np.asarray(state, dtype=float)

        if self._theta.shape == (signal.size, state.size):
            weights = self._softmax_weights(self._q_values(state))
            self._weights = np.asarray(weights, dtype=float)
        elif self._weights.size != signal.size:
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
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)
        if base_vec.size == 0:
            return np.zeros(1, dtype=float)

        signal = (2.0 * base_vec) - 1.0
        self._last_signal = np.asarray(signal, dtype=float)
        state = self._state_vector(today_market_features)
        self._last_market_state = np.asarray(state, dtype=float)

        if self._theta.shape == (signal.size, state.size):
            weights = self._softmax_weights(self._q_values(state))
            self._weights = np.asarray(weights, dtype=float)
        elif self._weights.size != signal.size:
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
        ret_vec = np.asarray(actual_return, dtype=float).reshape(-1)
        if ret_vec.size == 0:
            return

        if realized_base_predictions is not None:
            signal = (2.0 * np.asarray(realized_base_predictions, dtype=float).reshape(-1)) - 1.0
        else:
            signal = self._last_signal

        if signal.size == 0:
            return

        if realized_market_features is not None:
            current_state = self._state_vector(realized_market_features)
            self._last_market_state = np.asarray(current_state, dtype=float)
        elif self._last_market_state.size:
            current_state = np.asarray(self._last_market_state, dtype=float)
        else:
            current_state = self._state_vector(np.array([0.0], dtype=float))

        self._ensure_theta(signal.size, current_state.size)

        # Apply TD update for the previous transition now that S_{t+1} is available.
        if self._pending_state.size and self._pending_reward.size == signal.size:
            q_prev = self._q_values(self._pending_state)
            bootstrap = float(np.max(self._q_values(current_state)))
            td_target = self._pending_reward + (self.gamma * bootstrap)
            td_error = td_target - q_prev
            self._theta += self.learning_rate * np.outer(td_error, self._pending_state)

        if self._pending_signal.size == signal.size:
            turnover = np.abs(signal - self._pending_signal)
        else:
            turnover = np.zeros(signal.size, dtype=float)

        # Reward shaping: penalize turnover directly in the value-learning target.
        reward = (signal * float(ret_vec[0])) - (self.reward_cost_rate * turnover)

        # Queue current transition; it will be bootstrapped on the next update.
        self._pending_state = np.asarray(current_state, dtype=float)
        self._pending_signal = np.asarray(signal, dtype=float)
        self._pending_reward = np.asarray(reward, dtype=float)
        self._weights = self._softmax_weights(self._q_values(current_state))
        self._last_signal = np.asarray(signal, dtype=float)
