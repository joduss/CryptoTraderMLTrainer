import gym
import pandas as pd


from utilities.DateUtility import dateparse

# delete if it's registered
env_name = 'crypto-market-indicators-v1'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]




data_file_path = "./input/trades_binance_eth-usd-14-05-2021_1min.csv"
train_ratio = 0.7

data = pd.read_csv(data_file_path,
                        delimiter=',',
                        names=["time", "open", "high", "low", "close", "volume", "trades"],
                        parse_dates=True,
                        date_parser=dateparse,
                        index_col='time')
# data_size = len(data)
# data_train_size = round(data_size * train_ratio)
#
# data_train = data.iloc[:data_train_size]
# data_val = data.iloc[data_train_size:]


environment = gym.make('crypto_market:crypto-market-indicators-v1', data=data)

print(environment.reset())
print("DONE")


# environment_train = CryptoMarketIndicatorsEnvironment(data_train, 100)
# environment_val = MarketIndicatorTfEnvironment(data_val)



