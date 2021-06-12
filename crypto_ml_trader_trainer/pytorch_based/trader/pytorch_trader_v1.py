import os
import logging

import pandas as pd
import torch.optim
from stable_baselines3.common.env_checker import check_env

from ..core.dqn_trainer import DQNTrainer, DQNTrainerParameters
from ..core.pytorch_global_config import Device
from ..trader.environments.crypto_market_indicators_environment import CryptoMarketIndicatorsEnvironment
from ..trader.models.MarketIndicatorNN import MarketIndicatorNN
from ..trader.trader_policy import TraderPolicy
from ..utils.environment_tensor_wrapper import EnvironmentTensorWrapper
from crypto_ml_trader_trainer.utilities.DateUtility import dateparse

def run(data_file_path = None):
    print("run")

    # delete if it's registered
    env_name = 'crypto-market-indicators-v1'
    # if env_name in gym.envs.registry.env_specs:
        # del gym.envs.registry.env_specs[env_name]

    if data_file_path is None:
        data_file_path = "./input/trades_binance_eth-usd-14-05-2021_1min_subset.csv"

    if os.path.isfile(data_file_path)  == False:
        raise Exception(f"Path {data_file_path} doesn't exist.")


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

    Device.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    env: CryptoMarketIndicatorsEnvironment = CryptoMarketIndicatorsEnvironment(data).unwrapped
    #check_env(env)
    wrapped_env = EnvironmentTensorWrapper(env)

    env.logger.disabled = False
    env.logger.level = logging.DEBUG


    # environment_train = CryptoMarketIndicatorsEnvironment(data_train, 100)
    # environment_val = MarketIndicatorTfEnvironment(data_val)

    model = MarketIndicatorNN(input_length=wrapped_env.observation_space.shape[1])
    optimizer = torch.optim.RMSprop(model.network.parameters())
    policy = TraderPolicy(env, model)

    dqn_trainer = DQNTrainer(model=model,
                             environment=wrapped_env,
                             parameters=DQNTrainerParameters(),
                             optimizer=optimizer,
                             policy=policy)

    # Fill the buffer with a few random transitions.
    env.logger.disabled = True
    dqn_trainer.initialize_replay_buffer(RandomTraderPolicy(env), 20)
    env.logger.disabled = False

    print("Start training")
    dqn_trainer.train(200)


    print("DONE")


if __name__ == "__main__":
    run()