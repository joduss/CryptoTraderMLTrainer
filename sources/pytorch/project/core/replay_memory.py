from collections import deque

import random

from pytorch.project.core.Transition import Transition


class ReplayMemory:

    def __init__(self, size: int):
        self.size: int = size
        self._next_index = 0
        self.memory = deque([], maxlen=size)

    def push(self, *args):
        self.memory.append(Transition(*args))
        self._next_index = (1 + self._next_index) % self.size


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)