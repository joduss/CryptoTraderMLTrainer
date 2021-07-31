from torch import nn

from pytorch_based.core.pytorch_global_config import Device


class MarketIndicatorNN(nn.Module):

    def __init__(self, input_length: int):
        super(MarketIndicatorNN, self).__init__()
        self.network = nn.Sequential(
            # nn.Linear(input_length, 64),
            # nn.ReLU(),
            # nn.Linear(64,32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 3),
            nn.Linear(input_length, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        ).double()

        self.network = self.network.to(device=Device.device)

    def forward(self, x):
        logits = self.network(x)
        return logits