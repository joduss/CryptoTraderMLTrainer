import random

import torch

from pytorch_based.core.policy import Policy


class RandomPolicy(Policy):

    def __init__(self, action_count: int):
        self.action_count = action_count

    def decide(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[random.randint(0, self.action_count - 1)]])