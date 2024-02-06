from __future__ import absolute_import, division, print_function

import logging
import random

import gym
import numpy as np
import pandas as pd
from numpy import float64

from pytorch_based.trader.environments.market_environment_abstract import MarketEnvironmentAbstract, MarketEnvironmentState, \
    MarketStep
from pytorch_based.trader.environments.past_indicators.market_past_indicators_env_data import \
    MarketPastIndicatorsEnvData
from pytorch_based.trader.environments.wallets.single_order_wallet import SingleOrderWallet
from shared.environments.trading_action import TradingAction


class MarketPastIndicatorsEnv(MarketEnvironmentAbstract):

    logger: logging.Logger = logging.getLogger(__name__)
    _fee = 0.1 / 100
    _LEGAL_ACTION_HEAD_START = 100

    def __init__(self, data: pd.DataFrame,
                 initial_balance: float = 100,
                 rules_only: bool = False,
                 cache_dir: str = None,
                 index_jump: int = 15,
                 random_restart_position = True,
                 past_positions=[i for i in range(1,4*24*15,4)]
                 ):
        """

        @param data:
        @param initial_balance:
        @param rules_only:
        @param cache_dir:
        @param index_jump: If each row is data for a minute, then decision_frequency will produce aggregated data over 15 minutes
        @param random_restart_position
        @param past_positions
            If each row is data for a minute, then decision_frequency
            will produce aggregated data over 'index_jump' minutes.
        """

        self.past_positions = np.array(past_positions)
        self.random_restart_position = random_restart_position

        self.wallet: SingleOrderWallet = SingleOrderWallet(initial_balance)
        self.initial_balance = initial_balance

        self.rules_only = rules_only

        # Data and feature engineering (not normalized)
        self.data: MarketPastIndicatorsEnvData = MarketPastIndicatorsEnvData(
            data,
            index_jump,
            cache_dir=cache_dir)

        self._max_idx = len(self.data)
        self._data_idx = max(past_positions)
        self.indicator_count: int = len(self.data[0])
        self.previous_price = 0

        self.illegal_actions = 0
        self.legal_actions = self._LEGAL_ACTION_HEAD_START  # we give a head start
        self.allowed_illegal_action_rate = 0.1

        super().__init__(initial_balance)


    # region Interaction with environment


    def next(self, action: TradingAction) -> MarketStep:

        self._data_idx += 1

        # Handle last.
        if self._data_idx >= len(self.data.processed_data):
            MarketStep(next_state=se, reward=reward, ended=done)
            return

        # Special case for the first state
        if self._data_idx == 1:
            return MarketStep(next_state=self._state(), reward=0, ended=False)

        close = self.data.close_prices[self._data_idx]
        previous_close = self.data.close_prices[self._data_idx - 1]

        reward = self._execute_action(action, previous_close=previous_close, close=close)

        self.wallet.update_coin_price(close)
        done = False

        # Allow to loose max 25% from the max worth
        if self.wallet.net_worth / self.wallet.max_worth < 0.75:
            done = True
            reward = min(-0.25, reward)

        # if self.allowed_illegal_action_rate_min_actions > self.illegal_actions + self.legal_actions:
        if (self.illegal_actions / (self.legal_actions + self.illegal_actions + 1)) > self.allowed_illegal_action_rate:
            done = True
            reward = min(-1.0, reward)

        self.logger.debug("reward " + str(reward))
        return MarketStep(next_state=self._state(), reward=reward, ended=done)


    def _execute_action(self, action: TradingAction, previous_close: float, close: float) -> float:
        """
        Execute the action
        @param action: Action to execute
        @param previous_close: previous close price
        @param close: current close price
        @return: reward
        """
        if action not in self.allowed_actions():
            self.illegal_actions += 1
            return -1
        else:
            self.legal_actions += 1

        # Only learn rules
        if self.rules_only:
            if action == TradingAction.BUY:
                self._buy()
            if action == TradingAction.SELL:
                self._sell()
            return 1

        # Learn all

        if action == TradingAction.BUY:
            self._buy()
            return close / previous_close - 1  # small

        if action == TradingAction.SELL:
            profits = self.current_price() / self.wallet.initial_coin_price - 1
            self._sell()
            return profits

        if action == TradingAction.HOLD:
            return 0


    def _reset_market(self) -> MarketEnvironmentState:
        self._data_idx = max(self.past_positions) if self.random_restart_position is False else random.randint(max(self.past_positions), len(self.data.close_prices) - 60*24*4)
        self.wallet = SingleOrderWallet(self.initial_balance)
        self.illegal_actions = 0
        self.legal_actions = self._LEGAL_ACTION_HEAD_START
        return self.next(action=TradingAction.HOLD).next_state


    # endregion


    # region Environment current state data


    def _wallet_state(self) -> np.array:
        return np.array([
            1 if self.wallet.can_buy() else 0,
            1 if self.wallet.can_sell() else 0,
            self.wallet.initial_coin_price / self.current_price() - 1 if self.wallet.can_sell() else 0
            # self.wallet.balance
        ], dtype=float64).reshape((1,3))


    def _market_indicators(self) -> np.array:

        indices = self._data_idx - self.past_positions

        indicators = self.data[indices].T
        # indicators = indicators.reshape((1,-1))
        return np.expand_dims(indicators, axis=0) # add dimension as batch = 1
        # periods = len(self.data.history_periods)
        # indicator_count = int(self.data.processed_data.shape[1] / periods)
        #
        # return self.data[self._data_idx].reshape((indicator_count, periods))

    def _valid_action_mask(self) -> np.array:
        return np.array([TradingAction.hot_encode(self.allowed_actions())])


    def _state(self) -> MarketEnvironmentState:
        return MarketEnvironmentState(indicators=self._market_indicators(),
                                      wallet=self._wallet_state(),
                                      valid_actions_mask=self._valid_action_mask())

    # endregion


    # region Environment current state values


    def has_next(self) -> bool:
        return self._data_idx < self._max_idx and self.wallet.max_worth * 0.75 < self.wallet.net_worth


    def current_price(self) -> float:
        return self.data.close_prices[self._data_idx]


    def current_date(self):
        return self.data.dates[self._data_idx]


    def net_worth(self) -> float:
        return self.wallet.net_worth

    def allowed_actions(self) -> [TradingAction]:
        if self.wallet.can_sell():
            return [TradingAction.SELL, TradingAction.HOLD]
        elif self.wallet.can_buy():
            return [TradingAction.BUY, TradingAction.HOLD]



    def episode_progress(self) -> float:
        return self._data_idx / self._max_idx


    # endregion


    # region Logic


    def _buy(self) -> float:
        if not self.wallet.can_buy():
            self.logger.warning("Cannot buy")
            return 0

        coin_qty = (self.wallet.balance / self.current_price()) * (1 - self._fee)
        self.wallet.bought_coins(coins=coin_qty, price=self.current_price())

        self.logger.debug(f"{self.current_date()}: Buy {coin_qty} @ {self.current_price()}.")

        return 0


    def _sell(self) -> float:
        if not self.wallet.can_sell():
            self.logger.warning("Cannot sell")
            return 0

        sell_cost = self.wallet.coins_balance * self.current_price() * (1 - self._fee)
        profits = self.wallet.sold_coins(price=self.current_price(), value=sell_cost)

        self.logger.debug(
            f"{self.current_date()}: Selling with profit {profits} @ {self.current_price()}. Total profits: {self.wallet.profits()}, worth: {self.wallet.net_worth}")

        return profits
