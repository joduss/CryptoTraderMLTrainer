from torch import nn

from ...core.pytorch_global_config import PytorchGlobalConfig


class MarketIndicatorNN(nn.Module):

    def __init__(self, input_length: int):
        super(MarketIndicatorNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_length, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
            nn.ReLU()
        ).double()

        self.network = self.network.to(device=PytorchGlobalConfig.device)

    def forward(self, x):
        logits = self.network(x)
        return logits