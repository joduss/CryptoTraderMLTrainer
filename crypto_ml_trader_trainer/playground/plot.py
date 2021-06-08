import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

#%
from utilities.DateUtility import dateparse

prices = np.array([50, 52, 55, 53, 60, 55, 52, 51, 48, 47])
price_dates = np.array(range(1623088797, 1623088797 + 599, 60))


buys = np.array([55,52,47])
buys_dates = np.array([1623088797 + 120, 1623088797 + 360, 1623088797 + 540])
sells = np.array([60, 48])
sells_dates = np.array([1623088797 + 240, 1623088797 + 480])

#%
# noinspection PyTypeChecker
data = pd.DataFrame(
    {
    "buy" : pd.Series(buys, index=pd.to_datetime(buys_dates, unit='s')),
    "sell" : pd.Series(sells, index=pd.to_datetime(sells_dates, unit='s')),
    "price": pd.Series(prices, index=pd.to_datetime(price_dates, unit='s')),
    }
)



#%%

# ax = data.iloc[0].plot()

plt.ion()
ax = None


# formatter = matplotlib.dates.DateFormatter('%H:%M:%S')
# ax.xaxis.set_major_formatter(formatter)

for i in range(1,10):
    i_data = pd.DataFrame(
        {
            "price": pd.Series(prices[:i], index=pd.to_datetime(price_dates[:i], unit='s')),
        }
    )

    ax = i_data.plot(ax=ax, color="b")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.draw()
    plt.pause(0.2)
plt.show()



data["buy"].plot(marker="^", color="g")
data["sell"].plot(marker="v", color="r")