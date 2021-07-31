import math
import random

import torch
from torch import nn

from pytorch_based.core.policy import Policy
from pytorch_based.core.pytorch_global_config import Device


class GreedyPolicy(Policy):
    """
    Predicts actions, valid or not.
    """

    def __init__(self, num_actions: int, policy_net: nn.Module, eps_start: float = 0.95, eps_end: float = 0.05, eps_decay: float = 200, decay_per_episode: bool = True):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.policy_net = policy_net

        self._episodes_done: int = 0
        self._actions_taken: int = 0
        self._num_actions = num_actions
        self.decay_per_episode = decay_per_episode

    def next_episode(self):
        self._episodes_done += 1

    def decide(self, state: torch.Tensor) -> (torch.Tensor, bool):

        self._actions_taken += 1

        if self.decay_per_episode:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1. * self._episodes_done / self.eps_decay)
        else:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1. * self._actions_taken / self.eps_decay)

        sample = random.random()

        if sample > eps_threshold:
            return self.policy_net(state).max(1)[1].view(1,1)
        else:
            action = random.randint(0, self._num_actions - 1)
            return torch.tensor([[action]], dtype=torch.long).to(Device.device)
