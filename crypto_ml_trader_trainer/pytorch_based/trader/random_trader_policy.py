import random

import torch

from pytorch_based.core.policy import Policy
from pytorch_based.core.pytorch_global_config import Device
from pytorch_based.trader.environments.market_environment import MarketEnvironment


class RandomTraderPolicy(Policy):

    _steps_done: int = 0
    _N_ACTIONS = 3


    def __init__(self, env: MarketEnvironment):
        self.env: MarketEnvironment = env


    def decide(self, state):
        action = random.sample(self.env.valid_moves(), 1)[0]
        return torch.tensor([[action.value]],  dtype=torch.long).to(Device.device)

    def next_episode(self):
        return
