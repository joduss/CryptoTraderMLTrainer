from __future__ import absolute_import, division, print_function

from enum import Enum

import gym
import numpy as np
import pandas as pd
import pandas_ta
from gym import Space, spaces


class CryptoMarketIndicatorsEnvironment(gym.Env):

    initial_balance = 100
    wallet_state_size: int = 4

    # balance, nb_trade_open, nb_trade_available, price, macd, rsi,
    observation_length: int = 6

    def __init__(self, data: pd.DataFrame):
        super().__init__()

        self.state_gen = MarketEnvironmentStateGenerator(data, self.initial_balance)

        self._action_spec = spaces.Discrete(3)
        self._observation_spec =  spaces.Box(shape=[self.state_gen.indicator_count], dtype=np.double, low=-1.0, high=1)

        self.setInitialState()

    def observation_spec(self) -> Space:
        return self._observation_spec

    def action_spec(self) -> Space:
        return self._action_spec

    def setInitialState(self):
        self.state_gen.reset()
        state = self.state_gen.next()
        self._episode_ended = False
        return state

    def step(self, action: spaces.Discrete):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.s
            # return self.reset()
            raise Exception("The episod is ended. Reset it before calling step()")

        new_state = self.state_gen.next()

        if self.state_gen.hasNext() == False:
            self._episode_ended = True

        reward = self.state_gen.action(Action(action))

        if reward == None:
            return new_state, -10, True, {}

        if self._episode_ended and self.state_gen.total_created_order < self.state_gen.max_order_count:
            return new_state, -1, False, {}

        if self._episode_ended:
            return new_state, reward, True, {}

        # if self._balance < 10:
        #     return ts.termination(new_state, 0)
        # else:
            #print("transition")
        return new_state, reward,  False, {}

    def reset(self):
        return self.setInitialState()

    def render(self, mode='human'):
        print(f"Net worth: {self.state_gen.net_worth}")
        print(f"Balance: {self.state_gen.balance}")
        print(f"Profits: {self.state_gen.total_profits}")
        print(f"Current price: {self.state_gen.current_price}")


class Action(Enum):
    BUY = 0
    HOLD = 1
    SELL = 2



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

    balance: float
    coin_qty: int = 0

    net_worth: float
    total_profits: float = 0

    max_order_count: int = 10
    order_count = 0
    order_value: float

    indicator_count: int = 8

    index_jump = 16
    current_price: float = 0


    def __init__(self, data: pd.DataFrame, initial_balance: float):
        self.INITIAL_BALANCE: float = initial_balance

        self.data: pd.DataFrame = data
        self.prepareData()

        self.data = self.data.dropna()
        self._max_idx = self.data.shape[0]
        self._idx = self._history_size - 1

        self.order_value = initial_balance / self.max_order_count
        self.balance = initial_balance
        self.net_worth = initial_balance

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
        profits = sell_cost - self.order_value

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

        self.net_worth = self.balance + close * self.coin_qty
        self.current_price = close

        return np.concatenate((wallet_data, market_data))
        # return self.data[["time", "ma-30min","rsi-14days","rsi-14h","rsi-3h","macd","ema-12", "ema-26","bbands","stoch"]].to_numpy()

    def normalize(self, column: float, based_on: float) -> float:
       return (column - based_on) / based_on

    def hasNext(self) -> bool:
        return self._idx + self.index_jump < self._max_idx

    def _current_price(self):
        return self.data["close"].iloc[self._idx]

    def action(self, action: Action) -> float:
        if action == Action.BUY:
            return self.buy()

        if action == Action.SELL:
            return self.sell()

        if action == Action.HOLD:
            return 0


    def can_buy(self) -> float:
        return self.balance >= self.order_value and self.order_count < self.max_order_count

    def can_sell(self) -> bool:
        return self.order_count > 0


    def buy(self) -> float:
        if not self.can_buy():
            return 0

        self.coin_qty = (self.order_value / self._current_price()) * (1 - self._fee)
        self.order_count = 1
        self.balance -= self.order_value

        self.order_price = self._current_price()

        self.total_created_order += 1
        return 0

    def sell(self) -> float:
        if not self.can_sell():
            return 0

        sell_cost = self.coin_qty * self._current_price() * (1 - self._fee)
        profits = sell_cost - self.order_value

        self.order_price = 0

        self.balance += sell_cost
        self.order_count = 0
        self.order_value = self.balance / (self.max_order_count - self.order_count)
        self.coin_qty = 0

        self.total_created_order += 1

        self.total_profits += profits

        return profits


    def reset(self):
        self._idx = self._history_size - 1
        self.balance: int = 152
        self.coin_qty: int = 0
        self.order_count = 0
        self.order_value = self.balance / self.max_order_count

        self.order_price: float = 0
