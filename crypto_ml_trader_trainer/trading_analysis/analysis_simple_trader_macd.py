import pandas as pd
from crypto_ml_trader_trainer.utilities.DateUtility import dateparse





data = pd.read_csv('/Users/jonathanduss/Desktop/macd-buy-analysis.csv',
                   delimiter=',',
                   parse_dates=True,
                   date_parser=dateparse,
                   index_col='date')
data = data.dropna()

ax1 = data[["bid", "buy", "sell"]].plot()
ax2 = ax1.twinx()

data["macd"].plot(ax=ax2, color="lime")
data["signal"].plot(ax=ax2,color="red")

print(data)