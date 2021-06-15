from __future__ import absolute_import, division, print_function

from typing import Tuple

import numpy as np
import pandas as pd

from pytorch_based.trader.environments.current_tick_indicators.market_tick_indicator_data import MarketTickIndicatorData
from pytorch_based.trader.environments.market_env_logic import MarketEnvLogic
from pytorch_based.trader.environments.wallets.single_order_wallet import SingleOrderWallet
from shared.environments.trading_action import TradingAction


class MarketTickIndicatorsEnvLogic(MarketEnvLogic):

    _fee = 0.1 / 100
    _index_jump = 15


    def __init__(self, data: pd.DataFrame, initial_balance: float = 100):
        super().__init__()

        self.wallet: SingleOrderWallet = SingleOrderWallet(initial_balance)
        self.initial_balance = initial_balance

        # Data and feature engineering (not normalized)
        self.original_data: pd.DataFrame = data
        self.data = MarketTickIndicatorData.prepare_data(data, self._index_jump)

        self._max_idx = len(self.data)
        self._original_data_idx = 0
        self._data_idx = 0
        self.indicator_count: int = len(self.data[0])
        self.previous_net_worth = 0
        self.current_reward = 0

        self.reset()


    # region Interaction with environment


    def next(self) -> Tuple[np.array, float, bool]:

        # TODO: stop loss

        done = False

        self._original_data_idx += self._index_jump
        self._data_idx += 1
        close = self.original_data.iloc[self._original_data_idx]["close"]

        wallet_data = [
            1 if self.wallet.can_buy() else 0,
            1 if self.wallet.can_sell() else 0,
            # self.wallet.balance
        ]

        self.wallet.update_coin_price(close)

        if 1 - self.wallet.net_worth / self.wallet.max_worth > 0.25:
            done = True


        return np.concatenate((wallet_data, self.data[self._data_idx])).reshape(
            (1, len(wallet_data) + self.indicator_count)), self.current_reward, done


    def execute_action(self, action: TradingAction):

        if action not in self.valid_moves():
            return -1

        if action == TradingAction.BUY:
            self._buy()
            self.current_reward = self.wallet.net_worth - self.previous_net_worth

        if action == TradingAction.SELL:
            self.current_reward = self._sell()

        if action == TradingAction.HOLD:
            if self.wallet.can_sell():
                self.current_reward = self.wallet.net_worth - self.previous_net_worth
            else:
                self.current_reward = 0


    def reset(self):
        self._data_idx = 0
        self._original_data_idx = 0
        self.wallet = SingleOrderWallet(self.initial_balance)


    # endregion


    # region Environment current state


    def has_next(self) -> bool:
        return self._data_idx < self._max_idx and self.wallet.max_worth * 0.75 < self.wallet.net_worth

    def current_price(self):
        return self.original_data["close"].iloc[self._original_data_idx]

    def current_date(self):
        return self.original_data.index[self._data_idx]

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


    #region Logic


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
