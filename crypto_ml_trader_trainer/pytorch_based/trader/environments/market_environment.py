import logging
from logging import Logger

import gym
import numpy as np
import pandas as pd
from gym import spaces

from pytorch_based.trader.environments.logic.market_env_logic import MarketEnvLogic
from pytorch_based.trader.environments.logic.market_tick_indicators_env_logic import MarketTickIndicatorsEnvLogic
from pytorch_based.trader.environments.market_plot import MarketPlot
from shared.environments.trading_action import TradingAction


class MarketEnvironment(gym.Env):
    logger: Logger = logging.getLogger(__name__)

    max_net_worth = 0
    render_type = "none"
    print_market_action_frequency = 500

    def __init__(self, market_logic: MarketEnvLogic):
        super().__init__()

        self.market_logic = market_logic
        self.market_logic.logger = self.logger

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(shape=self.market_logic.next().shape, dtype=np.double, low=-1.0, high=1.0)

        self._set_initial_state()
        self.logger.level = logging.INFO
        self.logger.disabled = True
        self.plot = MarketPlot()

    def _set_initial_state(self):
        self.market_logic.reset()
        state = self.market_logic.next()
        self._episode_ended = False
        self.max_net_worth = self.market_logic.net_worth()
        return state

    #region Environment Interaction

    def step(self, action: spaces.Discrete):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            raise Exception("The episode is ended. Reset it before calling step()")

        new_state = self.market_logic.next()

        if not self.market_logic.has_next():
            self._episode_ended = True

        # Stop if loosing too much!
        if self.market_logic.net_worth() < 0.75 * self.max_net_worth:
            reward = -100
            self._episode_ended = True

        if self.max_net_worth < self.market_logic.net_worth():
            self.max_net_worth = self.market_logic.net_worth()

        reward = self.market_logic.execute_action(TradingAction(action))

        self.plot.add(self.market_logic.current_date(), self.market_logic.current_price(), TradingAction(action))

        if self._episode_ended:
            return new_state, reward, True, {}

        return new_state, reward, False, {}

    def valid_moves(self) -> [TradingAction]:
        return self.market_logic.valid_moves()

    def reset(self):
        self.plot.save_reset_plot()
        return self._set_initial_state()

    def render(self, mode='human'):
        print("-------------")
        print(f"Time: {self.market_logic.current_date()}")
        print(f"Current price: {self.market_logic.current_price()}")
        print(f"Net worth: {self.market_logic.net_worth()}")
        print(f"Balance: {self.market_logic.wallet.balance}")
        print(f"Profits: {self.market_logic.wallet.profits()}")
        print(f"Progress: {self.market_logic.episode_progress() * 100} %")

        if self.render_type == "plot":
            self.plot.plot()

    #end region