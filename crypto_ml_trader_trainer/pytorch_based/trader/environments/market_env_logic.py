from __future__ import annotations

from collections import namedtuple
from typing import Any, List, NamedTuple, Optional

import numpy as np
from logging import Logger

from pytorch_based.trader.environments.wallet import Wallet
from shared.environments.trading_action import TradingAction


MarketEnvironmentState = namedtuple("MarketEnvironmentState", ["indicators", "wallet", "valid_actions_mask"])
MarketStep = NamedTuple("MarketStep", [("next_state", MarketEnvironmentState), ("reward", Any), ("ended", Any)])


class MarketEnvLogic:

    def __init__(self):
        self.logger: Optional[Logger] = None
        self.wallet: Optional[Wallet] = None

    def next(self, action: TradingAction) -> MarketStep:
        """
        Returns observation, reward, early_termination
        """
        raise NotImplementedError()

    def _state(self) -> np.ndarray:
        raise NotImplementedError()

    def reset(self) -> MarketStep:
        raise NotImplementedError()


    def has_next(self) -> bool:
        raise NotImplementedError()

    def current_price(self):
        raise NotImplementedError()

    def current_date(self):
        raise NotImplementedError()

    def net_worth(self) -> float:
        raise NotImplementedError()

    def valid_moves(self) -> List[TradingAction]:
        raise NotImplementedError()

    def episode_progress(self) -> float:
        raise NotImplementedError()
