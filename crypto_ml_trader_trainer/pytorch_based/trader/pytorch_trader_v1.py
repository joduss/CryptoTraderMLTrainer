import os
import logging

import pandas as pd
import torch.optim

from pytorch_based.trader.environments.market_dqn_trainer import MarketDQNTrainer, MarketDQNTrainerParameters
from pytorch_based.core.pytorch_global_config import Device
from pytorch_based.trader.environments.current_tick_indicators.market_current_indicators_env import MarketCurrentIndicatorsEnv
from pytorch_based.trader.environments.market_environment_abstract import MarketEnvironmentAbstract
from pytorch_based.trader.environments.past_indicators.market_past_indicators_env import MarketPastIndicatorsEnv
from pytorch_based.trader.environments.past_indicators.market_past_indicators_nn import MarketPastIndicatorsNN
from pytorch_based.trader.policies.random_policy import RandomPolicy

from pytorch_based.trader.environments.current_tick_indicators.market_current_indicators_nn import MarketIndicatorNN
from crypto_ml_trader_trainer.utilities.DateUtility import dateparse
from pytorch_based.trader.environments.environment_tensor_wrapper import MarketEnvironmentTensorWrapper
from pytorch_based.trader.policies.trader_greedy_policy import TraderGreedyPolicy
from torchinfo import summary

def run(data_file_path: str, cache_dir: str):
    print("run")

    # delete if it's registered
    env_name = 'crypto-market-indicators-v1'
    # if env_name in gym.envs.registry.env_specs:
        # del gym.envs.registry.env_specs[env_name]

    if os.path.isfile(data_file_path)  == False:
        raise Exception(f"Path {data_file_path} doesn't exist.")

    os.makedirs(cache_dir, exist_ok=True)


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

    # Creation and configuration of the environment
    # env: MarketEnvironmentAbstract = MarketCurrentIndicatorsEnv(data, cache_dir=cache_dir, index_jump=15).unwrapped
    env: MarketEnvironmentAbstract = MarketPastIndicatorsEnv(data, cache_dir=cache_dir, index_jump=15).unwrapped
    #check_env(env)
    wrapped_env = MarketEnvironmentTensorWrapper(env)

    # environment_train = CryptoMarketIndicatorsEnvironment(data_train, 100)
    # environment_val = MarketIndicatorTfEnvironment(data_val)

    # Setup the model that will train using the previously created environment.
    # model = MarketIndicatorNN(input_length=wrapped_env.observation_space["indicators"].shape[0],
    #                           wallet_input_length=wrapped_env.observation_space["wallet"].shape[0])

    model = MarketPastIndicatorsNN(indicator_count=wrapped_env.observation_space["indicators"].shape[1],
                                   history_length=wrapped_env.observation_space["indicators"].shape[2],
                                   wallet_input_length=wrapped_env.observation_space["wallet"].shape[0])





    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    policy = TraderGreedyPolicy(policy_net=model, eps_decay=200, decay_per_episode=True, env=env, eps_end=0.02)

    parameters = MarketDQNTrainerParameters()
    parameters.batch_size = 64
    parameters.target_update = 25
    parameters.memory_size = 15000
    parameters.gamma = 0.999

    state = wrapped_env.reset()
    summary(model, input_size=[state.indicators.shape, state.wallet.shape, state.valid_actions_mask.shape],
            dtypes=[torch.double, torch.double, torch.double])

    dqn_trainer = MarketDQNTrainer(model=model,
                                   environment=wrapped_env,
                                   parameters=MarketDQNTrainerParameters(),
                                   optimizer=optimizer,
                                   policy=policy)


    # Logging levels
    env.logger.level = logging.DEBUG
    dqn_trainer.logger.level = logging.INFO

    # Fill the buffer with a few random transitions.
    env.logger.disabled = True
    dqn_trainer.initialize_replay_buffer(RandomPolicy(3), 10)
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
    run("./input/btc-ucd_ohlc_1min_binance_08-08-2021.csv", "./cache/")