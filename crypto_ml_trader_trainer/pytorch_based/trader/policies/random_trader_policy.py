import random

import torch

from pytorch_based.core.policy import Policy
from pytorch_based.core.pytorch_global_config import Device
from pytorch_based.trader.environments.market_environment_abstract import MarketEnvironmentAbstract


class RandomTraderPolicy(Policy):
    """
    Predicts randomly valid actions.
    """

    _steps_done: int = 0
    _N_ACTIONS = 3


    def __init__(self, env: MarketEnvironmentAbstract):
        self.env: MarketEnvironmentAbstract = env


    def decide(self, state):
        action = random.sample(self.env.valid_moves(), 1)[0]
        return torch.tensor([[action.value]],  dtype=torch.long).to(Device.device)

    def next_episode(self):
        return
