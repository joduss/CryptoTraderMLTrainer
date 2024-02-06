import math
import random

import torch

from pytorch_based.core.policy import Policy
from pytorch_based.core.pytorch_global_config import Device
from pytorch_based.trader.environments.current_tick_indicators.market_current_indicators_nn import MarketIndicatorNN
from pytorch_based.trader.environments.market_environment_abstract import MarketEnvironmentAbstract

from shared.environments.trading_action import TradingAction


class TraderGreedyPolicy(Policy):
    """
    Predicts actions, valid or not.
    """

    def __init__(self, env: MarketEnvironmentAbstract,
                 policy_net: MarketIndicatorNN,
                 eps_start: float = 0.95,
                 eps_end: float = 0.05,
                 eps_decay: float = 200,
                 decay_per_episode: bool = True):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.policy_net = policy_net

        self._episodes_done: int = 0
        self._actions_taken: int = 0
        self._num_actions = env.action_space.n
        self.decay_per_episode = decay_per_episode
        self.env = env

        # in case of a buy or sell, we might force all the next decision up to the next sell/buy to rely on the network.
        self.random_paused = False

    def next_episode(self):
        self._episodes_done += 1
        self.random_paused = False

    def decide(self, *state) -> (torch.Tensor, bool):

        self._actions_taken += 1
        self.policy_net.action_mask = torch.Tensor(TradingAction.hot_encode(self.env.allowed_actions()))

        if self.decay_per_episode:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1. * self._episodes_done / self.eps_decay)
        else:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1. * self._actions_taken / self.eps_decay)

        sample = random.random()

        if sample > eps_threshold or self.random_paused:
            action = self.policy_net(*state).max(1)[1].view(1,1)
        else:
            action = random.sample(self.env.allowed_actions(), 1)[0]
            action = torch.tensor([[action.value]], dtype=torch.long).to(Device.device)

        trading_action = TradingAction(action.item())
        if trading_action is TradingAction.BUY:
            sample = random.random()
            self.random_paused = True if sample > eps_threshold else False
        if trading_action is TradingAction.SELL:
            sample = random.random()
            self.random_paused = True if sample > eps_threshold else False

        return action



