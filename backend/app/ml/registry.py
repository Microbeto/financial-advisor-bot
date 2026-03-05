from __future__ import annotations

from typing import Callable, List

from .meta_interface import MetaCombiner
from .meta_labeler import MetaLabelerCombiner
from .regime_switcher import RegimeSwitcherCombiner
from .rl_agent import RLAgentCombiner


def get_meta_competitor_factories(random_seed: int = 42) -> List[Callable[[], MetaCombiner]]:
    seed = int(random_seed)
    return [
        lambda: MetaLabelerCombiner(random_state=seed),
        lambda: RegimeSwitcherCombiner(),
        lambda: RLAgentCombiner(),
    ]
