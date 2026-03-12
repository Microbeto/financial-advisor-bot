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
        self.learning_rate = float(max(1e-4, learning_rate))  # TD step size
        self.temperature = float(max(1e-4, temperature))  # Softmax exploration control
        self.transaction_cost_rate = float(np.clip(transaction_cost_rate, 0.0, 0.5))  # Allocation smoothing friction
        self.reward_cost_rate = float(max(0.0, reward_cost_rate))  # Reward-shaping turnover penalty
        self.gamma = float(np.clip(gamma, 0.0, 0.999))  # Discount factor for continuation value
        self._theta = np.zeros((0, 0), dtype=float)  # Linear contextual weights: action x state
        self._weights = np.array([], dtype=float)  # Current policy over base models
        self._last_signal = np.array([], dtype=float)  # Last realized action signal
        self._last_market_state = np.array([], dtype=float)  # Last encoded state vector
        self._pending_state = np.array([], dtype=float)  # S_t queued for next TD bootstrap
        self._pending_signal = np.array([], dtype=float)  # a_t signal queued for turnover penalty
        self._pending_reward = np.array([], dtype=float)  # r_{t+1} vector queued for TD target

    def _state_vector(self, market_features: np.ndarray) -> np.ndarray:
        # Normalize market context and append bias term for linear contextual Q-values.
        vec = np.asarray(market_features, dtype=float).reshape(-1)  # Ensure 1D state shape
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)  # Remove NaN/Inf instability
        if vec.size == 0:  # Keep a valid state even when features are missing
            vec = np.zeros(1, dtype=float)
        return np.concatenate([vec, np.array([1.0], dtype=float)])  # Append intercept term

    def _ensure_theta(self, n_models: int, state_dim: int) -> None:
        if self._theta.shape == (n_models, state_dim):  # No resize needed
            return
        new_theta = np.zeros((n_models, state_dim), dtype=float)  # Fresh parameter matrix
        if self._theta.size:  # Preserve overlapping learned parameters on shape changes
            r = min(n_models, self._theta.shape[0])
            c = min(state_dim, self._theta.shape[1])
            new_theta[:r, :c] = self._theta[:r, :c]
        self._theta = new_theta  # Commit resized parameters

    def _q_values(self, state: np.ndarray) -> np.ndarray:
        if self._theta.size == 0:  # Uninitialized agent has no value estimates
            return np.array([], dtype=float)
        return np.asarray(self._theta @ state, dtype=float)  # Linear contextual value per action

    def _softmax_weights(self, q_values: np.ndarray) -> np.ndarray:
        if q_values.size == 0:  # Empty fallback
            return np.array([], dtype=float)
        scaled = q_values / self.temperature  # Temperature scaling
        scaled -= float(np.max(scaled))  # Numerical stability shift
        exp_scores = np.exp(scaled)  # Exponentiate logits
        denom = float(np.sum(exp_scores))  # Softmax normalizer
        if denom <= 0.0:  # Degenerate fallback
            return np.ones(q_values.size, dtype=float) / float(q_values.size)
        return exp_scores / denom  # Probability-like policy weights

    def train(self, base_predictions: np.ndarray, market_features: np.ndarray, actual_returns: np.ndarray) -> None:
        if base_predictions.ndim != 2:  # Enforce matrix input
            raise ValueError("base_predictions must be 2D")

        n_samples, n_bases = base_predictions.shape
        if n_samples == 0 or n_bases == 0:  # Guard empty train set
            raise ValueError("Insufficient training rows for RLAgent")

        returns = np.asarray(actual_returns, dtype=float).reshape(-1)  # Realized returns timeline
        if returns.shape[0] != n_samples:  # Align labels and inputs
            raise ValueError("actual_returns length mismatch")

        market = np.asarray(market_features, dtype=float)  # State feature matrix
        if market.ndim != 2 or market.shape[0] != n_samples:  # Validate contextual dimensions
            raise ValueError("market_features shape mismatch")

        init_state = self._state_vector(market[0])  # Infer state dimensionality
        self._ensure_theta(n_bases, init_state.size)  # Initialize action-value parameters
        prev_signal = np.array([], dtype=float)  # Needed for turnover penalty

        # TD(0): update theta_i for each action i using contextual bootstrap target.
        for t in range(n_samples):  # Iterate chronological transitions
            state_t = self._state_vector(market[t])  # Current context S_t
            self._ensure_theta(n_bases, state_t.size)  # Safety for variable feature width
            signal_t = (2.0 * np.asarray(base_predictions[t], dtype=float).reshape(-1)) - 1.0  # Action signal vector
            if signal_t.size != n_bases:  # Validate action space width
                raise ValueError("base_predictions row width mismatch")

            if prev_signal.size == signal_t.size:  # Compute action-change cost term
                turnover = np.abs(signal_t - prev_signal)
            else:
                turnover = np.zeros(signal_t.size, dtype=float)
            reward_t = (signal_t * float(returns[t])) - (self.reward_cost_rate * turnover)  # Friction-shaped reward

            if t + 1 < n_samples:  # Bootstrap with next-state best action value
                state_next = self._state_vector(market[t + 1])
                bootstrap = float(np.max(self._q_values(state_next)))
            else:
                bootstrap = 0.0  # Terminal transition has no continuation value

            q_t = self._q_values(state_t)  # Current Q(S_t, a)
            td_target = reward_t + (self.gamma * bootstrap)  # Bellman target
            td_error = td_target - q_t  # Temporal difference residual
            self._theta += self.learning_rate * np.outer(td_error, state_t)  # Gradient update per action
            prev_signal = signal_t  # Store for next turnover penalty

        last_state = self._state_vector(market[-1])  # Use latest context to initialize policy
        self._weights = self._softmax_weights(self._q_values(last_state))  # Initial action distribution
        self._last_signal = (2.0 * np.asarray(base_predictions[-1], dtype=float).reshape(-1)) - 1.0  # Last action cache
        self._last_market_state = np.asarray(last_state, dtype=float)  # Last state cache
        self._pending_state = np.array([], dtype=float)  # Clear online TD buffers
        self._pending_signal = np.array([], dtype=float)
        self._pending_reward = np.array([], dtype=float)

    def predict_confidence(self, today_base_predictions: np.ndarray, today_market_features: np.ndarray) -> np.ndarray:
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)  # Flatten base outputs
        if base_vec.size == 0:  # Empty input fallback
            return np.zeros(1, dtype=float)

        signal = (2.0 * base_vec) - 1.0  # Convert probabilities into signed action strengths
        self._last_signal = np.asarray(signal, dtype=float)  # Cache for delayed feedback updates
        state = self._state_vector(today_market_features)  # Encode current market context
        self._last_market_state = np.asarray(state, dtype=float)  # Cache state for fallback

        if self._theta.shape == (signal.size, state.size):  # Use contextual policy when dimensions match
            weights = self._softmax_weights(self._q_values(state))
            self._weights = np.asarray(weights, dtype=float)
        elif self._weights.size != signal.size:  # Uniform fallback for cold-start/mismatch
            weights = np.ones(signal.size, dtype=float) / float(signal.size)
        else:
            weights = self._weights

        confidence = float(np.clip(np.sum(weights * np.abs(signal)), 0.0, 1.0))  # Weighted absolute conviction
        return np.array([confidence], dtype=float)

    def allocate(
        self,
        today_base_predictions: np.ndarray,
        today_market_features: np.ndarray,
        current_allocation: np.ndarray | None = None,
    ) -> np.ndarray:
        base_vec = np.asarray(today_base_predictions, dtype=float).reshape(-1)  # Flatten base outputs
        if base_vec.size == 0:  # Empty input fallback
            return np.zeros(1, dtype=float)

        signal = (2.0 * base_vec) - 1.0  # Convert probabilities into signed action strengths
        self._last_signal = np.asarray(signal, dtype=float)  # Cache last action proposal
        state = self._state_vector(today_market_features)  # Encode current market context
        self._last_market_state = np.asarray(state, dtype=float)  # Cache state for step update

        if self._theta.shape == (signal.size, state.size):  # Contextual policy path
            weights = self._softmax_weights(self._q_values(state))
            self._weights = np.asarray(weights, dtype=float)
        elif self._weights.size != signal.size:  # Uniform fallback for cold-start/mismatch
            weights = np.ones(signal.size, dtype=float) / float(signal.size)
        else:
            weights = self._weights

        target = float(np.clip(np.dot(weights, signal), -1.0, 1.0))  # Raw target allocation

        prev = 0.0  # Default previous allocation
        if current_allocation is not None:  # Read prior position for turnover control
            prev_vec = np.asarray(current_allocation, dtype=float).reshape(-1)
            if prev_vec.size:
                prev = float(np.clip(prev_vec[0], -1.0, 1.0))

        turnover = abs(target - prev)  # Position change magnitude
        trade_fraction = max(0.0, 1.0 - (self.transaction_cost_rate * turnover))  # Friction-aware throttle
        adjusted = prev + trade_fraction * (target - prev)  # Smooth toward target
        return normalize_allocation_vector(np.array([adjusted], dtype=float))  # Return clipped 1D vector

    def step_update(
        self,
        actual_return: np.ndarray,
        realized_base_predictions: np.ndarray | None = None,
        realized_market_features: np.ndarray | None = None,
    ) -> None:
        ret_vec = np.asarray(actual_return, dtype=float).reshape(-1)  # Realized return input
        if ret_vec.size == 0:  # No update without observed outcome
            return

        if realized_base_predictions is not None:  # Prefer realized prediction snapshot when available
            signal = (2.0 * np.asarray(realized_base_predictions, dtype=float).reshape(-1)) - 1.0
        else:
            signal = self._last_signal  # Fallback to cached last signal

        if signal.size == 0:  # Cannot attribute reward without action signal
            return

        if realized_market_features is not None:  # Prefer realized state S_{t+1}
            current_state = self._state_vector(realized_market_features)
            self._last_market_state = np.asarray(current_state, dtype=float)
        elif self._last_market_state.size:  # Use cached state when explicit realized features are absent
            current_state = np.asarray(self._last_market_state, dtype=float)
        else:
            current_state = self._state_vector(np.array([0.0], dtype=float))  # Last-resort neutral state

        self._ensure_theta(signal.size, current_state.size)  # Ensure parameter matrix matches action/state sizes

        # Apply TD update for the previous transition now that S_{t+1} is available.
        if self._pending_state.size and self._pending_reward.size == signal.size:
            q_prev = self._q_values(self._pending_state)  # Q(S_t, .)
            bootstrap = float(np.max(self._q_values(current_state)))  # max_a Q(S_{t+1}, a)
            td_target = self._pending_reward + (self.gamma * bootstrap)  # Bellman target vector
            td_error = td_target - q_prev  # TD residual vector
            self._theta += self.learning_rate * np.outer(td_error, self._pending_state)  # Parameter update

        if self._pending_signal.size == signal.size:  # Turnover versus prior action signal
            turnover = np.abs(signal - self._pending_signal)
        else:
            turnover = np.zeros(signal.size, dtype=float)

        # Reward shaping: penalize turnover directly in the value-learning target.
        reward = (signal * float(ret_vec[0])) - (self.reward_cost_rate * turnover)  # Friction-penalized reward

        # Queue current transition; it will be bootstrapped on the next update.
        self._pending_state = np.asarray(current_state, dtype=float)  # Queue S_t for next bootstrap
        self._pending_signal = np.asarray(signal, dtype=float)  # Queue action signal for turnover term
        self._pending_reward = np.asarray(reward, dtype=float)  # Queue immediate reward vector
        self._weights = self._softmax_weights(self._q_values(current_state))  # Refresh policy from latest Q
        self._last_signal = np.asarray(signal, dtype=float)  # Keep fallback cache aligned
