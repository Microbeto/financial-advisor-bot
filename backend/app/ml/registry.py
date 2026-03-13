from __future__ import annotations

from itertools import product
from typing import Callable, List

from .grand_ensemble import GrandEnsembleCombiner
from .meta_interface import MetaCombiner
from .meta_labeler import MetaLabelerCombiner
from .regime_switcher import RegimeSwitcherCombiner
from .rl_agent import RLAgentCombiner


# --- Named factory helpers (avoid Python lambda late-binding in loops) ---

def _labeler_factory(seed: int, side_threshold: float, label: str) -> Callable[[], MetaCombiner]:
    # MetaLabeler variant factory: captures all params in closure
    def factory() -> MetaCombiner:
        m = MetaLabelerCombiner(random_state=seed, side_threshold=side_threshold)
        m.name = label  # Override class name so tournament stats are uniquely keyed
        return m
    return factory


def _regime_factory(temperature: float, span: int, label: str) -> Callable[[], MetaCombiner]:
    # RegimeSwitcher variant factory: captures temperature and EWMA span
    def factory() -> MetaCombiner:
        m = RegimeSwitcherCombiner(softmax_temperature=temperature, ewma_span=span)
        m.name = label
        return m
    return factory


def _rl_factory(
    learning_rate: float,
    temperature: float,
    gamma: float,
    reward_cost_rate: float,
    label: str,
) -> Callable[[], MetaCombiner]:
    # RLAgent variant factory: captures optimizer speed, exploration, horizon, and friction penalty
    def factory() -> MetaCombiner:
        m = RLAgentCombiner(
            learning_rate=learning_rate,
            temperature=temperature,
            gamma=gamma,
            reward_cost_rate=reward_cost_rate,
        )
        m.name = label
        return m
    return factory


def get_meta_competitor_factories(random_seed: int = 42) -> List[Callable[[], MetaCombiner]]:
    seed = int(random_seed)

    # MetaLabeler grid: vary side_threshold (signal conviction gate)
    # Lower threshold → more active signals; higher → fewer, higher-conviction trades
    meta_labeler_factories = [
        _labeler_factory(seed, 0.01, "meta_labeler_t01"),   # tight gate, high activity
        _labeler_factory(seed, 0.02, "meta_labeler_t02"),   # default
        _labeler_factory(seed, 0.05, "meta_labeler_t05"),   # loose gate, low activity
    ]

    # RegimeSwitcher grid: vary softmax_temperature (exploitation/exploration) and ewma_span (memory)
    # Lower temperature → sharper model selection; longer span → smoother regime memory
    regime_temperatures = [0.5, 1.0, 2.0]
    regime_spans = [10, 20, 40]
    regime_switcher_factories = []
    for temperature, span in product(regime_temperatures, regime_spans):
        temp_tag = str(temperature).replace(".", "p")
        label = f"regime_switcher_t{temp_tag}_s{int(span)}"
        regime_switcher_factories.append(_regime_factory(temperature, span, label))

    # RLAgent grid (9 variants): speed/exploration × horizon/friction profiles
    # This gives tournament selection room to discover regime-specific RL behavior.
    rl_specs = [
        (0.01, 0.5, 0.85, 0.0005, "rl_agent_slow_short_lc"),
        (0.01, 0.5, 0.95, 0.0010, "rl_agent_slow_mid_mc"),
        (0.01, 0.5, 0.99, 0.0020, "rl_agent_slow_long_hc"),
        (0.05, 0.8, 0.85, 0.0005, "rl_agent_base_short_lc"),
        (0.05, 0.8, 0.95, 0.0010, "rl_agent_base_mid_mc"),
        (0.05, 0.8, 0.99, 0.0020, "rl_agent_base_long_hc"),
        (0.10, 1.2, 0.85, 0.0005, "rl_agent_fast_short_lc"),
        (0.10, 1.2, 0.95, 0.0010, "rl_agent_fast_mid_mc"),
        (0.10, 1.2, 0.99, 0.0020, "rl_agent_fast_long_hc"),
    ]
    rl_agent_factories = [
        _rl_factory(lr, temp, gamma, cost, label)
        for lr, temp, gamma, cost, label in rl_specs
    ]

    # GrandEnsemble: blends one default of each combiner, dynamically re-weighted by realized PnL
    # Often wins tournament via diversification and lowest maximum drawdown
    grand_sub_factories: List[Callable[[], MetaCombiner]] = [
        lambda: MetaLabelerCombiner(random_state=seed),
        lambda: RegimeSwitcherCombiner(),
        lambda: RLAgentCombiner(),
    ]
    grand_ensemble_factories: List[Callable[[], MetaCombiner]] = [
        lambda: GrandEnsembleCombiner(sub_factories=grand_sub_factories),
    ]

    return [
        *meta_labeler_factories,
        *regime_switcher_factories,
        *rl_agent_factories,
        *grand_ensemble_factories,
    ]
