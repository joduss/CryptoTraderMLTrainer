import logging
from logging import Logger
from typing import Optional

import gym
import numpy as np
from gym import spaces

from pytorch_based.trader.environments.market_env_logic import MarketEnvLogic, MarketEnvironmentState, MarketStep
from pytorch_based.trader.environments.market_plot import MarketPlot
from shared.environments.trading_action import TradingAction


class MarketEnvironment(gym.Env):
    logger: Logger = logging.getLogger(__name__)

    max_net_worth = 0
    print_market_action_frequency = 500


    def __init__(self, market_logic: MarketEnvLogic):
        super().__init__()

        self.market_logic: MarketEnvLogic = market_logic
        self.market_logic.logger = self.logger

        self.action_space = spaces.Discrete(3)
        # self.observation_space = spaces.Box(shape=self.market_logic.reset()[0].shape, dtype=np.double, low=-1.0, high=1.0)

        indicator_field_name = "indicators"
        wallet_field_name = "wallet"
        valid_actions_mask_field_name = "valid_actions_mask"

        if indicator_field_name not in list(MarketEnvironmentState._fields) \
            or wallet_field_name not in list(MarketEnvironmentState._fields) \
            or valid_actions_mask_field_name not in list(MarketEnvironmentState._fields):
            raise NameError("Property names of the namedtuple don't match")

        reset_step = self.market_logic.reset()
        reset_state: MarketEnvironmentState = reset_step.next_state

        self.observation_space = {
            indicator_field_name: spaces.Box(shape=(reset_state.indicators.shape[1],), dtype=np.double, high=np.finfo(np.float32).max, low=np.finfo(np.float32).min),
            valid_actions_mask_field_name: len(reset_state.valid_actions_mask),
            wallet_field_name: spaces.Box(shape=(reset_state.wallet.shape[1],), dtype=np.double, high=np.finfo(np.float32).max, low=np.finfo(np.float32).min)
        }
        # self.observation_space = spaces.Box(shape=(10,), dtype=np.double, low=-1.0, high=1.0) # fake...

        self._set_initial_state()
        self.logger.level = logging.INFO
        self.logger.disabled = True
        self.market_plot: Optional[MarketPlot] = None


    def _set_initial_state(self) -> MarketEnvironmentState:
        self.market_logic.reset()
        state, _, _ = self.market_logic.reset()
        self._episode_ended = False
        self.max_net_worth = self.market_logic.net_worth()
        return state


    # region Environment Interaction


    def step(self, action: int) -> MarketStep:
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            raise Exception("The episode is ended. Reset it before calling step()")

        state = self.market_logic.next(TradingAction(action))

        if not self.market_logic.has_next() or state.ended:
            self._episode_ended = True

        if self.market_plot is not None:
            self.market_plot.add(self.market_logic.current_date(), self.market_logic.current_price(),
                                 TradingAction(action))

        if self._episode_ended:
            return state.next_state, state.reward, True

        return state.next_state, state.reward, False


    def reset(self) -> MarketEnvironmentState:
        if self.market_plot is not None:
            self.market_plot.save_reset_plot()
        return self._set_initial_state()


    def render(self, mode='human'):
        print("-------------")
        print(f"Time: {self.market_logic.current_date()}")
        print(f"Current price: {self.market_logic.current_price()}")
        print(f"Net worth: {self.market_logic.net_worth()} / Max: {self.market_logic.wallet.max_worth}")
        print(f"Balance: {self.market_logic.wallet.balance}")
        print(f"Profits: {self.market_logic.wallet.profits()}")
        print(f"Progress: {self.market_logic.episode_progress() * 100} %")

        if mode == "plot":
            if self.market_plot is None:
                self.market_plot = MarketPlot()
            self.market_plot.plot()

    # endregion
