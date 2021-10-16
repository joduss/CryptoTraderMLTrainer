import logging
from collections import namedtuple
from logging import Logger
from typing import Any, List, NamedTuple, Optional

import gym
import numpy as np
from gym import spaces

from pytorch_based.trader.environments.market_plot import MarketPlot
from pytorch_based.trader.environments.wallet_interface import WalletInterface
from shared.environments.trading_action import TradingAction


MarketEnvironmentState = namedtuple("MarketEnvironmentState", ["indicators", "wallet", "valid_actions_mask"])
MarketStep = NamedTuple("MarketStep", [("next_state", MarketEnvironmentState), ("reward", Any), ("ended", Any)])


class MarketEnvironmentAbstract(gym.Env):
    logger: Logger = logging.getLogger(__name__)

    print_market_action_frequency = 500


    def __init__(self, initial_balance: float = 100):
        super().__init__()

        self.logger: Logger = self.logger
        self.logger.level = logging.INFO
        self.logger.disabled = True

        # Check if MarketEnvironmentState fields are matching
        indicator_field_name = "indicators"
        wallet_field_name = "wallet"
        valid_actions_mask_field_name = "valid_actions_mask"

        if indicator_field_name not in list(MarketEnvironmentState._fields) \
            or wallet_field_name not in list(MarketEnvironmentState._fields) \
            or valid_actions_mask_field_name not in list(MarketEnvironmentState._fields):
            raise NameError("Property names of the namedtuple don't match")


        # Initialize common properties.
        self.wallet: Optional[WalletInterface] = None
        self.market_plot: Optional[MarketPlot] = None

        self.initial_balance: float = initial_balance
        self.max_net_worth = 0
        self._episode_ended = False

        # Set the initial state
        reset_state = self.reset()

        # From the state we set the action and observation space.
        self.action_space = spaces.Discrete(3)

        self.observation_space = {
            indicator_field_name: spaces.Box(shape=reset_state.indicators.shape, dtype=np.double, high=np.finfo(np.float32).max, low=np.finfo(np.float32).min),
            valid_actions_mask_field_name: len(reset_state.valid_actions_mask),
            wallet_field_name: spaces.Box(shape=(reset_state.wallet.shape[1],), dtype=np.double, high=np.finfo(np.float32).max, low=np.finfo(np.float32).min)
        }

    # region Environment Interaction


    def step(self, action: int) -> MarketStep:
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            raise Exception("The episode is ended. Reset it before calling step()")

        state = self.next(TradingAction(action))

        if not self.has_next() or state.ended:
            self._episode_ended = True

        if self.market_plot is not None:
            self.market_plot.add(self.current_date(), self.current_price(),
                                 TradingAction(action))

        if self._episode_ended:
            return state.next_state, state.reward, True

        self.max_net_worth = self.wallet.max_worth

        return state.next_state, state.reward, False


    def reset(self) -> MarketEnvironmentState:

        self._episode_ended = False

        if self.market_plot is not None:
            self.market_plot.save_reset_plot()

        state = self._reset_market()
        self.max_net_worth = self.wallet.max_worth
        return state


    def render(self, mode='human'):
        print("-------------")
        print(f"Time: {self.current_date()}")
        print(f"Current price: {self.current_price()}")
        print(f"Net worth: {self.net_worth()} / Max: {self.wallet.max_worth}")
        print(f"Balance: {self.wallet.balance}")
        print(f"Profits: {self.wallet.profits()}")
        print(f"Progress: {self.episode_progress() * 100} %")

        if mode == "plot":
            if self.market_plot is None:
                self.market_plot = MarketPlot()
            self.market_plot.plot()

    # endregion


    # region Environment abstract

    def next(self, action: TradingAction) -> MarketStep:
        """
        Returns a MarketStep
        """
        raise NotImplementedError()

    def _reset_market(self) -> MarketEnvironmentState:
        raise NotImplementedError()


    def _state(self) -> np.ndarray:
        raise NotImplementedError()

    def has_next(self) -> bool:
        raise NotImplementedError()

    def current_price(self):
        raise NotImplementedError()

    def current_date(self):
        raise NotImplementedError()

    def net_worth(self) -> float:
        raise NotImplementedError()

    def allowed_actions(self) -> List[TradingAction]:
        raise NotImplementedError()

    def episode_progress(self) -> float:
        raise NotImplementedError()

    #endregion
