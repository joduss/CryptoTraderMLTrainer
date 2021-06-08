from __future__ import absolute_import, division, print_function

import logging
from logging import Logger
from typing import Dict

import gym
import numpy as np
import pandas as pd
import pandas_ta
from gym import Space, spaces

from .market_plot import MarketPlot
from .trading_action import Action


class CryptoMarketIndicatorsEnvironment(gym.Env):

    logger: Logger = logging.getLogger(__name__)



    def __init__(self, data: pd.DataFrame):
        super().__init__()

        self.state_gen = MarketEnvironmentStateGenerator(data, logger=self.logger)

        self.action_space = spaces.Discrete(3)
        self.observation_space =  spaces.Box(shape=self.state_gen.next().shape, dtype=np.double, low=-1.0, high=1.0)

        self.setInitialState()
        self.logger.level = logging.INFO
        self.logger.disabled = True
        self.plot = MarketPlot()


    def valid_moves(self) -> [Action]:
        return self.state_gen.valid_moves()

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

        self.plot.add(self.state_gen.time(), self.state_gen.current_price(), Action(action))

        reward = self.state_gen.execute_action(Action(action))

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
        self.plot.save_reset_plot()
        return self.setInitialState()

    def render(self, mode='human'):
        print("-------------")
        print(f"Time: {self.state_gen.time()}")
        print(f"Current price: {self.state_gen.current_price}")
        print(f"Net worth: {self.state_gen.net_worth()}")
        print(f"Balance: {self.state_gen.balance}")
        print(f"Profits: {self.state_gen.total_profits}")
        print(f"Progress: {self.state_gen.progress() * 100} %")

        self.plot.plot()








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

    total_profits: float = 0

    max_order_count: int = 1
    order_count = 0
    order_value: float = 0
    order_price: float = 0

    indicator_count: int = 8

    index_jump = 16

    market_data_cache: Dict = dict()


    def __init__(self, data: pd.DataFrame, logger: Logger, initial_balance: float = 100):
        self.INITIAL_BALANCE: float = initial_balance

        self.data: pd.DataFrame = data
        self.prepareData()

        self.data: pd.DataFrame = self.data.dropna()
        self._max_idx = self.data.shape[0]
        self._idx = self._history_size - 1

        self.order_value = initial_balance / self.max_order_count
        self.balance = initial_balance

        self.indicator_count = len(self.next())
        self.reset()
        self.logger = logger



    def prepareData(self):
        # min = self.data["low"].min()
        # max = self.data["high"].max()
        current = self.data["close"].iloc[-1]

        self.data["ma-5min"] = self.data["close"].rolling(window=5).mean()
        # self.data["ma-30min"] = self.data["close"].rolling(window=30).mean()
        # self.data["ma-180min"] = self.data["close"].rolling(window=180).mean()
        # self.data["ma-360min"] = self.data["close"].rolling(window=360).mean()
        self.data["rsi-14days"] = self.data.ta.rsi(length=14*60*24) / 100
        self.data["rsi-14h"] = self.data.ta.rsi(length=14*60) / 100
        # self.data["rsi-3h"] = self.data.ta.rsi(length=3*60)
        self.data["ema-12"] = self.data.ta.ema(length=12*5)
        self.data["ema-26"] = self.data.ta.ema(length=26*5)
        # self.data["bbands"] = self.data.ta.bbands(length=15)
        # self.data["stoch"] = self.data.ta.stoch(length=60)
        # self.data = self.data.join(self.data.ta.macd(length=15))


    def normalize(self, column: float, center: float, divisor: float) -> float:
        """
         Normalize the data in respect to a given factor.
        :param column:
        :param factor:
        :return:
        """
        return (column - center) / divisor
    #
    #
    # def discount(self) -> float:
    #     if self.coin_qty == 0:
    #         return 0
    #
    #     sell_cost = self.coin_qty * self.current_price() * (1 - self._fee)
    #     profits = sell_cost - self.order_value
    #
    #     return profits


    def next(self):
        self._idx += self.index_jump
        # state_data: pd.DataFrame = self.data["close"].iloc[self._idx - self._history_size : self._idx]
        # return {
        #     "market" : state_data.to_numpy(),
        #     "wallet" : np.array([self.balance, self.order_count, self.max_order_count, self.order_size], dtype=float)
        # }
        close = self.data["close"].iloc[self._idx]

        if self._idx in self.market_data_cache:
            market_data = self.market_data_cache[self._idx]
        else:
            market_data = self.data[["close", "ma-5min", "rsi-14days", "rsi-14h", "ema-12", "ema-26"]].iloc[self._idx].to_numpy()

            max_value = max(market_data[0], market_data[1], market_data[4], market_data[5])
            min_value =  min(market_data[0], market_data[1], market_data[4], market_data[5])
            max_diff = max(abs(max_value - close), abs(close - min_value))

            market_data[0] = self.normalize(market_data[0], close, max_diff)
            market_data[1] = self.normalize(market_data[1], close, max_diff)
            market_data[4] = self.normalize(market_data[4], close, max_diff)
            market_data[5] = self.normalize(market_data[5], close, max_diff)



            self.market_data_cache[self._idx] = market_data.tolist()

        wallet_data = [
            (self.max_order_count - self.order_count) / self.max_order_count,
            self.order_price / close - 1
        ]

        return np.concatenate((wallet_data, market_data)).reshape((1,len(wallet_data) + len(market_data)))
        # return self.data[["time", "ma-30min","rsi-14days","rsi-14h","rsi-3h","macd","ema-12", "ema-26","bbands","stoch"]].to_numpy()

    def valid_moves(self) -> [Action]:
        if self.can_sell():
            return [Action.SELL, Action.HOLD]
        elif self.can_buy():
            return [Action.BUY, Action.HOLD]

    def hasNext(self) -> bool:
        return self._idx + self.index_jump < self._max_idx


    def current_price(self):
        return self.data["close"].iloc[self._idx]


    def time(self):
        return self.data.index[self._idx]


    def net_worth(self):
        return self.balance + self.current_price() * self.coin_qty

    def execute_action(self, action: Action) -> float:
        if action == Action.BUY:
            return self.buy()

        if action == Action.SELL:
            return self.sell()

        if action == Action.HOLD:
            if self.order_price != 0:
                return (self.order_price - self.current_price()) * self.coin_qty
            else:
                return 0


    def can_buy(self) -> float:
        return self.balance >= self.order_value and self.order_count < self.max_order_count

    def can_sell(self) -> bool:
        return self.order_count > 0


    def buy(self) -> float:
        if not self.can_buy():
            self.logger.warning("Cannot buy")
            return 0

        self.coin_qty = (self.order_value / self.current_price()) * (1 - self._fee)
        self.order_count += 1
        self.balance -= self.order_value

        self.order_price = self.current_price()

        self.total_created_order += 1
        self.logger.debug(f"{self.time()}: Buy {self.coin_qty} @ {self.current_price()}.")

        return 0

    def sell(self) -> float:
        if not self.can_sell():
            self.logger.warning("Cannot sell")
            return 0

        sell_cost = self.coin_qty * self.current_price() * (1 - self._fee)
        profits = sell_cost - self.order_value

        self.order_price = 0

        self.balance += sell_cost
        self.order_count = 0
        self.order_value = self.balance / (self.max_order_count - self.order_count)
        self.coin_qty = 0

        self.total_created_order += 1

        self.total_profits += profits

        self.logger.debug(f"{self.time()}: Selling with profit {profits} @ {self.current_price()}. Total profits: {self.total_profits}, worth: {self.balance}")

        return profits

    def progress(self) -> float:
        return self._idx / self._max_idx


    def reset(self):
        self._idx = self._history_size - 1
        self.balance = self.INITIAL_BALANCE
        self.coin_qty: int = 0
        self.order_count = 0
        self.order_value = self.balance / self.max_order_count
        self.total_profits = 0
        self.order_price = 0

