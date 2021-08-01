from __future__ import absolute_import, division, print_function

from typing import Tuple

import numpy as np
import pandas as pd
from numpy import float64

from pytorch_based.trader.environments.current_tick_indicators.market_tick_indicator_data import MarketTickIndicatorData
from pytorch_based.trader.environments.market_env_logic import MarketEnvLogic
from pytorch_based.trader.environments.wallets.single_order_wallet import SingleOrderWallet
from shared.environments.trading_action import TradingAction


class MarketTickIndicatorsEnvLogic(MarketEnvLogic):
    _fee = 0.1 / 100
    _LEGAL_ACTION_HEAD_START = 100

    index_jump = 15  # If each row is data for a minute, then index_jump will produce aggregated data over 15 minutes

    def __init__(self, data: pd.DataFrame,
                 initial_balance: float = 100,
                 rules_only: bool = False,
                 cache_dir: str = None):
        super().__init__()

        self.wallet: SingleOrderWallet = SingleOrderWallet(initial_balance)
        self.initial_balance = initial_balance

        self.rules_only = rules_only

        # Data and feature engineering (not normalized)
        self.data: MarketTickIndicatorData = MarketTickIndicatorData(
            data,
            self.index_jump,
            cache_dir=cache_dir)

        self._max_idx = len(self.data)
        self._data_idx = 0
        self.indicator_count: int = len(self.data[0])
        self.previous_price = 0

        self.illegal_actions = 0
        self.legal_actions = self._LEGAL_ACTION_HEAD_START  # we give a head start
        self.allowed_illegal_action_rate = 0.1

        self.reset()


    # region Interaction with environment


    def next(self, action: TradingAction) -> Tuple[np.array, float, bool]:

        self._data_idx += 1

        # Special case for the first state
        if self._data_idx == 1:
            return self._state(), 0, False

        close = self.data.close_prices[self._data_idx]
        previous_close = self.data.close_prices[self._data_idx - 1]

        reward = self._execute_action(action, previous_close=previous_close, close=close)

        self.wallet.update_coin_price(close)
        done = False

        # Allow to loose max 25% from the max worth
        if self.wallet.net_worth / self.wallet.max_worth < 0.75:
            done = True
            reward = min(-1.0, reward)

        # if self.allowed_illegal_action_rate_min_actions > self.illegal_actions + self.legal_actions:
        if (self.illegal_actions / (self.legal_actions + self.illegal_actions + 1)) > self.allowed_illegal_action_rate:
            done = True
            reward = min(-1.0, reward)

        self.logger.debug("reward " + str(reward))
        return self._state(), reward, done


    def _execute_action(self, action: TradingAction, previous_close: float, close: float) -> float:
        """
        Execute the action
        @param action: Action to execute
        @param previous_close: previous close price
        @param close: current close price
        @return: reward
        """
        if action not in self.valid_moves():
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


    def reset(self) -> np.array:
        self._data_idx = 0
        self._original_data_idx = 0
        self.wallet = SingleOrderWallet(self.initial_balance)
        self.illegal_actions = 0
        self.legal_actions = self._LEGAL_ACTION_HEAD_START
        return self.next(action=TradingAction.HOLD)


    # endregion


    # region Environment current state data


    def _wallet_state(self) -> np.array:
        return np.array([
            1 if self.wallet.can_buy() else 0,
            1 if self.wallet.can_sell() else 0,
            self.wallet.initial_coin_price / self.current_price() - 1 if self.wallet.can_sell() else 0
            # self.wallet.balance
        ], dtype=float64)


    def _market_state(self) -> np.array:
        return self.data[self._data_idx]


    def _state(self) -> np.array:
        wallet_data = self._wallet_state()
        market_state = self._market_state()

        return np.concatenate((wallet_data, market_state)).reshape(
            (1, len(wallet_data) + self.indicator_count))


    # endregion


    # region Environment current state values


    def has_next(self) -> bool:
        return self._data_idx < self._max_idx and self.wallet.max_worth * 0.75 < self.wallet.net_worth


    def current_price(self):
        return self.data.close_prices[self._data_idx]


    def current_date(self):
        return self.data.dates[self._data_idx]


    def net_worth(self):
        return self.wallet.net_worth


    def valid_moves(self) -> [TradingAction]:
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
