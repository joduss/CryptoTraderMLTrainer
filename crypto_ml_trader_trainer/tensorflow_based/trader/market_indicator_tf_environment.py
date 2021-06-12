from __future__ import absolute_import, division, print_function

from enum import Enum
from typing import Any

import numpy as np

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.typing import types
import pandas as pd
import pandas_ta as pdt

from shared.environments.trading_action import TradingAction


class MarketIndicatorTfEnvironment(PyEnvironment):

    data_file_path: str
    wallet_state_size: int = 4

    # balance, nb_trade_open, nb_trade_available, price, macd, rsi,
    observation_length: int = 6

    def __init__(self, data: pd.DataFrame):
        super().__init__()

        self.state_gen = MarketEnvironmentStateGenerator(data)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec =  array_spec.BoundedArraySpec(shape=(self.state_gen.indicator_count,), dtype=np.double, minimum=0, name='observation')

        self.setInitialState()


    def setInitialState(self):
        self.state_gen.reset()
        self._state = self.state_gen.next()
        self._episode_ended = False

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec


    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec


    def get_info(self) -> Any:
        pass


    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        new_state = self.state_gen.next()

        if self.state_gen.hasNext() == False:
            self._episode_ended = True

        reward = self.state_gen.action(TradingAction(action.item()))

        if reward == None:
            return ts.termination(new_state, -10)

        if self._episode_ended and self.state_gen.total_created_order < self.state_gen.max_order_count:
            return ts.termination(new_state, -1)

        if self._episode_ended:
            return ts.termination(new_state, self.state_gen.total_profits)

        # if self._balance < 10:
        #     return ts.termination(new_state, 0)
        # else:
            #print("transition")
        return ts.transition(new_state, reward=reward, discount=0.9)


    def _reset(self) -> ts.TimeStep:
        self.setInitialState()
        return ts.restart(self._state)



# ========================================================
# MarketEnvironmentStateGenerator
# ========================================================


# FOR NOW, order = 100%
class MarketEnvironmentStateGenerator:

    total_created_order: int = 0

    _history_size: int = 225
    _idx: int = 0
    _max_idx: int = 0

    _fee = 0.1 / 100

    balance: int = 159
    coin_qty: int = 0
    max_order_count = 10
    order_count = 0
    order_size = balance / max_order_count

    order_price: float = 0

    indicator_count: int = 8

    index_jump = 16

    total_profits: float = 0


    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data
        self.prepareData()

        self.data = self.data.dropna()
        self._max_idx = self.data.shape[0]
        self._idx = self._history_size - 1

    def prepareData(self):
        self.data["ma-5min"] = self.data["close"].rolling(window=5).mean()
        # self.data["ma-30min"] = self.data["close"].rolling(window=30).mean()
        # self.data["ma-180min"] = self.data["close"].rolling(window=180).mean()
        # self.data["ma-360min"] = self.data["close"].rolling(window=360).mean()
        self.data["rsi-14days"] = self.data.ta.rsi(length=14*60*24)
        self.data["rsi-14h"] = self.data.ta.rsi(length=14*60)
        # self.data["rsi-3h"] = self.data.ta.rsi(length=3*60)
        self.data["ema-12"] = self.data.ta.ema(length=12*5)
        self.data["ema-26"] = self.data.ta.ema(length=26*5)
        # self.data["bbands"] = self.data.ta.bbands(length=15)
        # self.data["stoch"] = self.data.ta.stoch(length=60)
        # self.data = self.data.join(self.data.ta.macd(length=15))


    def discount(self) -> float:
        if self.coin_qty == 0:
            return 0

        sell_cost = self.coin_qty * self._current_price() * (1 - self._fee)
        profits = sell_cost - self.order_size

        return profits

    def next(self):
        self._idx += self.index_jump
        # state_data: pd.DataFrame = self.data["close"].iloc[self._idx - self._history_size : self._idx]
        # return {
        #     "market" : state_data.to_numpy(),
        #     "wallet" : np.array([self.balance, self.order_count, self.max_order_count, self.order_size], dtype=float)
        # }
        market_data = self.data[["ma-5min", "rsi-14days", "rsi-14h", "ema-12", "ema-26"]].iloc[self._idx].to_numpy()
        close = self.data["close"].iloc[self._idx]
        market_data[0] = self.normalize(market_data[0], close)
        market_data[3] = self.normalize(market_data[3], close)
        market_data[4] = self.normalize(market_data[4], close)

        wallet_data = np.array([self.max_order_count - self.order_count, self.max_order_count, self.order_count])

        return np.concatenate((wallet_data, market_data))
        # return self.data[["time", "ma-30min","rsi-14days","rsi-14h","rsi-3h","macd","ema-12", "ema-26","bbands","stoch"]].to_numpy()

    def normalize(self, column: float, based_on: float) -> float:
       return (column - based_on) / based_on

    def hasNext(self) -> bool:
        return self._idx + self.index_jump < self._max_idx

    def _current_price(self):
        return self.data["close"].iloc[self._idx]

    def action(self, action: TradingAction) -> float:
        if action == TradingAction.BUY:
            return self.buy()

        if action == TradingAction.SELL:
            return self.sell()

        if action == TradingAction.HOLD:
            return 0


    def can_buy(self) -> float:
        return self.balance >= self.order_size and self.order_count < self.max_order_count

    def can_sell(self) -> bool:
        return self.order_count > 0


    def buy(self) -> float:
        if not self.can_buy():
            return 0

        self.coin_qty = (self.order_size / self._current_price()) * (1 - self._fee)
        self.order_count = 1
        self.balance -= self.order_size

        self.order_price = self._current_price()

        self.total_created_order += 1
        return 0

    def sell(self) -> float:
        if not self.can_sell():
            return 0

        sell_cost = self.coin_qty * self._current_price() * (1 - self._fee)
        profits = sell_cost - self.order_size

        self.order_price = 0

        self.balance += sell_cost
        self.order_count = 0
        self.order_size = self.balance / (self.max_order_count - self.order_count)
        self.coin_qty = 0

        self.total_created_order += 1

        self.total_profits += profits

        return profits


    def reset(self):
        self._idx = self._history_size - 1
        self.balance: int = 152
        self.coin_qty: int = 0
        self.order_count = 0
        self.order_size = self.balance / self.max_order_count

        self.order_price: float = 0
