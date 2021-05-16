import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from pytorch.utils.Trainer import Trainer


X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 3.3, 1.1, 2.4, 5.3, 1, 0, 0.1, 17, 3.345, 8.443, 3.9, 6.554, 2.2, 10]
Y  = torch.Tensor([[10 * x] for x in X])
X = torch.Tensor([[x] for x in X])


dataset = TensorDataset(X, Y)

dataset_test_len = round(len(X)*0.3)
dataset_train_len = len(X) - dataset_test_len

[dataset_train, dataset_test] = torch.utils.data.random_split(dataset, [dataset_train_len, dataset_test_len])

dataloader_train = DataLoader(dataset_train, batch_size=3)
dataloader_test = DataLoader(dataset_test)




#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Creating the model
# ====================

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        logits = self.network(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


#%%
# Optimization
# ====================

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    previous_len = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        batch = batch + 1

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)

        if previous_len > len(X):
            current = previous_len * (batch - 1) + len(X)
        previous_len = len(X)

        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader_train, model, loss_fn, optimizer)
    test(dataloader_test, model)

pred = model(torch.tensor([[1.0]]))

print("Done!")


# trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
# trainer.train()