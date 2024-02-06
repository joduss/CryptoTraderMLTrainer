import torch
from torch import nn

from pytorch_based.core.pytorch_global_config import Device


class MarketPastIndicatorsNN(nn.Module):

    def __init__(self, indicator_count: int, history_length: int, wallet_input_length: int):

        super(MarketPastIndicatorsNN, self).__init__()

        kernel_size = 9
        padding = 0
        stride = 3

        conv1_size = ((history_length - kernel_size + 2 * padding) / stride) + 1
        conv2_size = int(((conv1_size - kernel_size + 2 * padding)/ stride) + 1)

        self.market_net = nn.Sequential(
            nn.BatchNorm1d(num_features=indicator_count),
            nn.Conv1d(in_channels=indicator_count, out_channels=16, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv2_size * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        ).double().to(device=Device.device)

        self.wallet_net = nn.Sequential(
            nn.Linear(wallet_input_length, 8),
            nn.ReLU(),
        ).double().to(device=Device.device)

        self.decision_net = nn.Sequential(
            nn.Linear(64 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        ).double().to(device=Device.device)
        # logits = self.softmax(logits) NO: WE CANNOT USE SOFTMAX. The value returned is the reward!


    def forward(self, state, wallet, action_mask):

        # got [32, 360, 132]
        market_state_output = self.market_net(state)
        wallet_output = self.wallet_net(wallet)

        logits = self.decision_net(torch.cat((market_state_output, wallet_output), dim=1))

        if action_mask is not None:
            logits = action_mask * logits

        return logits