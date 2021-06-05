import math
import random

import torch
from torch import nn

from pytorch_based.core.policy import Policy
from pytorch_based.core.pytorch_global_config import PytorchGlobalConfig


class TraderPolicy(Policy):

    _steps_done: int = 0
    _N_ACTIONS = 3


    def __init__(self, policy_net: nn.Module, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: float = 200):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.policy_net = policy_net


    def decide(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self._steps_done / self.eps_decay)
        sample = random.random()

        self._steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return
                # largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self._N_ACTIONS)]], device=PytorchGlobalConfig.device,
                                dtype=torch.long)


