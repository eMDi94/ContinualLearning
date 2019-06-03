from __future__ import print_function

from networks import MLP
from datasets import get_mnist_training_data_loader, get_mnist_validation_data_loader
from utils import device

import torch
import torch.nn as nn
import torch.optim as optim


def train_mlp():
    mlp = MLP(784, 10)
    mlp = mlp.to(device)
    optimizer = optim.SGD(mlp.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print('Start training...')
    for idx in range(50):
        mnist = get_mnist_training_data_loader('./data/', 20)
        losses = []
        for batch in mnist:
            data, labels = batch
            data = data.to(device)
            data = data.view(data.size()[0], -1)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = mlp(data)
            loss = criterion(output, labels)
            loss.backward()
            losses.append(loss)
            optimizer.step()
        mean_loss = torch.tensor(losses, device=device).mean()
        print('Iteration ', idx, '. Mean loss: ', mean_loss)

    print('End training...')

    print('Saving state dictionary...')
    torch.save(mlp.state_dict(), 'mlp_weights')
    print('Model saved.')
    print('Done!!')


if __name__ == '__main__':
    train_mlp()
