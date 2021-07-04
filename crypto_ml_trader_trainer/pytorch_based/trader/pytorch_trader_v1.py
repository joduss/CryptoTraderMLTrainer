import os
import logging

import pandas as pd
import torch.optim

from pytorch_based.trader.environments.current_tick_indicators.market_tick_indicators_env_logic import MarketTickIndicatorsEnvLogic
from pytorch_based.trader.environments.market_environment import MarketEnvironment
from pytorch_based.trader.greedy_policy import GreedyPolicy
from pytorch_based.trader.policies.random_policy import RandomPolicy
from pytorch_based.trader.random_trader_policy import RandomTraderPolicy
from ..core.dqn_trainer import DQNTrainer, DQNTrainerParameters
from ..core.pytorch_global_config import Device
from pytorch_based.trader.environments.current_tick_indicators.market_indicator_nn import MarketIndicatorNN
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


    env: MarketEnvironment = MarketEnvironment(MarketTickIndicatorsEnvLogic(data)).unwrapped
    #check_env(env)
    wrapped_env = EnvironmentTensorWrapper(env)




    # environment_train = CryptoMarketIndicatorsEnvironment(data_train, 100)
    # environment_val = MarketIndicatorTfEnvironment(data_val)

    model = MarketIndicatorNN(input_length=wrapped_env.observation_space.shape[1])
    optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.01)
    # policy = TraderPolicy(env, model)
    policy = GreedyPolicy(num_actions=3, policy_net=model, eps_decay=30000, decay_per_episode=False)

    parameters = DQNTrainerParameters()
    parameters.target_update = 1000
    parameters.gamma = 0.99

    dqn_trainer = DQNTrainer(model=model,
                             environment=wrapped_env,
                             parameters=DQNTrainerParameters(),
                             optimizer=optimizer,
                             policy=policy)


    # Logging levels
    env.logger.level = logging.DEBUG
    dqn_trainer.logger.level = logging.INFO

    # Fill the buffer with a few random transitions.
    env.logger.disabled = True
    dqn_trainer.initialize_replay_buffer(RandomPolicy(3), 1000)
    env.logger.disabled = True

    print("Start training")

    dqn_trainer.train(30000)

    torch.save({
        'epoch': dqn_trainer.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"output/{model.__class__.__name__}.pt_model")


    print("DONE")


if __name__ == "__main__":
    run()