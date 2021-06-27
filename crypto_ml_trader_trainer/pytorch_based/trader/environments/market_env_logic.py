from typing import Optional, Tuple

import numpy as np
from logging import Logger

from pytorch_based.trader.environments.wallet import Wallet
from shared.environments.trading_action import TradingAction


class MarketEnvLogic:

    def __init__(self):
        self.logger: Optional[Logger] = None
        self.wallet: Optional[Wallet] = None

    def next(self, action: TradingAction) -> Tuple[np.array, float, bool]:
        """
        Returns observation, reward, early_termination
        """
        raise NotImplementedError()

    def _state(self) -> np.array:
        raise NotImplementedError()

    def reset(self) -> np.array:
        raise NotImplementedError()


    def has_next(self) -> bool:
        raise NotImplementedError()

    def current_price(self):
        raise NotImplementedError()

    def current_date(self):
        raise NotImplementedError()

    def net_worth(self):
        raise NotImplementedError()

    def valid_moves(self) -> [TradingAction]:
        raise NotImplementedError()

    def episode_progress(self) -> float:
        raise NotImplementedError()
