from __future__ import print_function
import torch

import utils


def train_mlp(model, data_loader, optimizer, criterion, iterations, device=utils.device):
    model = model.to(device)
    for idx in range(iterations):
        losses = []
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            losses.append(loss)
        mean_loss = torch.tensor(losses, device=device).mean()
        print('Optimization iteration number ' + str(idx + 1) + '. Mean Loss: ' + mean_loss)
    return model
