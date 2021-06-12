from datetime import datetime
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.figure as figure

from shared.environments.trading_action import TradingAction



class MarketPlot:

    _dates: List[datetime] = []
    _prices: List[float] = []

    _sell_dates: List[datetime] = []
    _sell_prices: List[float] = []

    _buy_dates: List[datetime] = []
    _buy_prices: List[float] = []

    _round = 0


    def __init__(self):
        plt.ion()
        self.ax = None

    def add(self, date: float, price: float, action: TradingAction):
        self._dates.append(pd.to_datetime(date, unit='s'))
        self._prices.append(price)

        if (action == TradingAction.BUY):
            self._buy_prices.append(price)
            self._buy_dates.append(pd.to_datetime(date, unit='s'))

        elif (action == TradingAction.SELL):
            self._sell_prices.append(price)
            self._sell_dates.append(pd.to_datetime(date, unit='s'))


    def plot(self):

        data = pd.DataFrame(
            {
                "price" : pd.Series(data = self._prices, index=self._dates),
                "sell": pd.Series(data=self._sell_prices, index=self._sell_dates),
                "buy": pd.Series(data=self._buy_prices, index=self._buy_dates),
            }
        )


        self.ax = data["price"].plot(ax=self.ax, color="b")
        data["buy"].plot(marker="^", color="g")
        data["sell"].plot(marker="v", color="r")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        fig = plt.gcf() # type:figure.Figure
        fig.set_size_inches(25,13)
        fig.tight_layout()
        plt.draw()
        plt.show(block=False)
        plt.grid(color='gray', linestyle='--', linewidth=0.2)
        plt.pause(0.01)


    def save_reset_plot(self):
        plt.savefig(f'output/plots/round_{self._round}.svg', bbox_inches='tight', dpi=600)
        plt.close('all')
        self.ax = None
        self._round += 1

        self._dates = []
        self._prices = []
        self._buy_dates = []
        self._buy_prices = []
        self._sell_prices = []
        self._sell_dates = []