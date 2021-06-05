from typing import Callable

import torch
from torch.nn import Module
from torch.nn.modules import loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_based.core.pytorch_global_config import PytorchGlobalConfig


class Trainer:

    def __init__(self, model: Module, loss_fn, optimizer: Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        model.to(self.device)
        self.device = PytorchGlobalConfig.device


    def _train(self, dataloader):
        size = len(dataloader.dataset)
        previous_len = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            batch = batch + 1

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss, current = loss.item(), batch * len(X)

            if previous_len > len(X):
                current = previous_len * (batch - 1) + len(X)
            previous_len = len(X)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def _test(self, dataloader):
        size = len(dataloader.dataset)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
        test_loss /= size
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")


    def train(self, dataloader_test: DataLoader, dataloader_train: DataLoader, epochs: int):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self._train(dataloader_train)
            self._test(dataloader_test)
