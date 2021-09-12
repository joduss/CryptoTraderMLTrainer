import torch


class Policy:

    def decide(self, *observation) -> torch.Tensor:
        raise NotImplementedError()

    def next_episode(self):
        """
        Notify the next episode starts
        """
        raise NotImplementedError()
