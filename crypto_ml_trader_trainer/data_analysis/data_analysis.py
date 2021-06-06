from crypto_ml_trader_trainer.utilities.DateUtility import dateparse
import pandas as pd
import pandas_ta as ta

#%%
data = pd.read_csv('input/ohlc_btc-usd_1min_2021.csv',
                   delimiter=',',
                   names=["time", "open", "high", "low", "close", "volume", "trades"],
                   parse_dates=True,
                   date_parser=dateparse,
                   index_col='time')

interval = 1

data["close"].plot()
data.ta.ema(length=interval * 12).plot()
data.ta.ema(length=interval * 26).plot()
# data.ta.macd(length=15).plot()