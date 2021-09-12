import math
import random

import torch

from pytorch_based.core.policy import Policy
from pytorch_based.core.pytorch_global_config import Device
from pytorch_based.trader.environments.current_tick_indicators.market_indicator_nn import MarketIndicatorNN
from pytorch_based.trader.environments.market_environment import MarketEnvironment

from shared.environments.trading_action import TradingAction


class TraderGreedyPolicy(Policy):
    """
    Predicts actions, valid or not.
    """

    def __init__(self, env: MarketEnvironment, policy_net: MarketIndicatorNN, eps_start: float = 0.95, eps_end: float = 0.05, eps_decay: float = 200, decay_per_episode: bool = True):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.policy_net = policy_net

        self._episodes_done: int = 0
        self._actions_taken: int = 0
        self._num_actions = env.action_space.n
        self.decay_per_episode = decay_per_episode
        self.env = env

    def next_episode(self):
        self._episodes_done += 1

    def decide(self, *state) -> (torch.Tensor, bool):

        self._actions_taken += 1
        self.policy_net.action_mask = torch.Tensor(TradingAction.hot_encode(self.env.market_logic.valid_moves()))

        if self.decay_per_episode:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1. * self._episodes_done / self.eps_decay)
        else:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1. * self._actions_taken / self.eps_decay)

        sample = random.random()

        if sample > eps_threshold:
            return self.policy_net(*state).max(1)[1].view(1,1)
        else:
            action = random.sample(self.env.market_logic.valid_moves(), 1)[0]
            return torch.tensor([[action.value]], dtype=torch.long).to(Device.device)
