import logging

import gym
import pandas as pd
import torch.optim
from stable_baselines3.common.env_checker import check_env

from pytorch_based.core.dqn_trainer import DQNTrainer, DQNTrainerParameters
from pytorch_based.trader.models.MarketIndicatorNN import MarketIndicatorNN
from pytorch_based.trader.trader_policy import TraderPolicy
from pytorch_based.utils.environment_tensor_wrapper import EnvironmentTensorWrapper
from utilities.DateUtility import dateparse
# import crypto_market_envs.crypto_market.envs
from crypto_market_envs.crypto_market.envs import CryptoMarketIndicatorsEnvironment


# delete if it's registered
env_name = 'crypto-market-indicators-v1'
# if env_name in gym.envs.registry.env_specs:
    # del gym.envs.registry.env_specs[env_name]



data_file_path = "./input/trades_binance_eth-usd-14-05-2021_1min_subset.csv"
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
logging.basicConfig()


env: CryptoMarketIndicatorsEnvironment = gym.make('crypto_market:crypto-market-indicators-v1', data=data).unwrapped
check_env(env)
wrapped_env = EnvironmentTensorWrapper(env)
print(wrapped_env.reset())

env.logger.disabled = False
env.logger.level = logging.DEBUG


# environment_train = CryptoMarketIndicatorsEnvironment(data_train, 100)
# environment_val = MarketIndicatorTfEnvironment(data_val)

model = MarketIndicatorNN(input_length=wrapped_env.observation_space.shape[1])
optimizer = torch.optim.RMSprop(model.network.parameters())
policy = TraderPolicy(model)



dqn_trainer = DQNTrainer(model=model,
                         environment=wrapped_env,
                         parameters=DQNTrainerParameters(),
                         optimizer=optimizer,
                         policy=policy)
print("Start training")
dqn_trainer.train(5)


print("DONE")