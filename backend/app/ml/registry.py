from __future__ import annotations

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


def _rl_factory(learning_rate: float, temperature: float, label: str) -> Callable[[], MetaCombiner]:
    # RLAgent variant factory: captures learning rate and softmax temperature
    def factory() -> MetaCombiner:
        m = RLAgentCombiner(learning_rate=learning_rate, temperature=temperature)
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
    regime_switcher_factories = [
        _regime_factory(0.5, 10,  "regime_switcher_hot"),   # aggressive, short memory
        _regime_factory(1.0, 20,  "regime_switcher_base"),  # default, balanced
        _regime_factory(2.0, 40,  "regime_switcher_cool"),  # smooth, long memory
    ]

    # RLAgent grid: vary learning_rate (Q-update speed) and temperature (policy sharpness)
    # Slow learns cautiously; fast adapts quickly but may overfit recent noise
    rl_agent_factories = [
        _rl_factory(0.01, 0.5, "rl_agent_slow"),   # conservative, focused policy
        _rl_factory(0.05, 0.8, "rl_agent_base"),   # default
        _rl_factory(0.10, 1.5, "rl_agent_fast"),   # aggressive, diverse policy
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
