from typing import Dict, Optional, Tuple

import numpy as np
from logging import Logger
from shared.environments.trading_action import TradingAction


class MarketEnvLogic:

    def __init__(self):
        self.logger: Optional[Logger] = None

    def next(self) -> Tuple[np.array, float, bool]:
        """
        Returns observation, reward, early_termination
        """
        raise NotImplementedError()

    def execute_action(self, action: TradingAction):
        raise NotImplementedError()

    def reset(self):
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
