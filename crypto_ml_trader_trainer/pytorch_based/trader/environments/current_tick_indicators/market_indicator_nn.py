import torch
from torch import nn

from pytorch_based.core.pytorch_global_config import Device


class MarketIndicatorNN(nn.Module):

    def __init__(self, input_length: int, wallet_input_length: int):

        super(MarketIndicatorNN, self).__init__()
        self.network = nn.Sequential(
            # nn.Conv1d(in_channels=input_length, out_channels=32, kernel_size=3),
            # nn.Flatten(),
            nn.Linear(input_length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        ).double()

        self.wallet_net = nn.Sequential(
            nn.Linear(wallet_input_length, 8),
            nn.ReLU(),
        ).double()

        self.decision_net = nn.Sequential(
            nn.Linear(64 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        ).double()
        # logits = self.softmax(logits) NO: WE CANNOT USE SOFTMAX. The value returned is the reward!

        self.network = self.network.to(device=Device.device)

    def forward(self, state, wallet, action_mask):
        market_state_output = self.network(state)
        wallet_output = self.wallet_net(wallet)

        logits = self.decision_net(torch.cat((market_state_output, wallet_output), dim=1))

        if action_mask is not None:
            logits = action_mask * logits

        return logits