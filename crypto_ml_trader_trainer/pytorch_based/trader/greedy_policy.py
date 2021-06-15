import math
import random

import torch
from torch import nn

from pytorch_based.core.pytorch_global_config import Device


class GreedyPolicy:
    """
    Predicts actions, valid or not.
    """

    def __init__(self, num_actions: int, policy_net: nn.Module, eps_start: float = 0.95, eps_end: float = 0.05, eps_decay: float = 200):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.policy_net = policy_net

        self._episodes_done: int = 0
        self._num_actions = num_actions

    def next_episode(self):
        self._episodes_done += 1

    def decide(self, state: torch.Tensor) -> (torch.Tensor, bool):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self._episodes_done / self.eps_decay)
        sample = random.random()

        if sample > eps_threshold:
            return self.policy_net(state)
        else:
            action = random.randint(0, self._num_actions - 1)
            return torch.tensor([[action.value]], dtype=torch.long).to(Device.device)
