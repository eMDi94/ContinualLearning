from __future__ import print_function

from networks import MLP
from datasets import get_mnist_training_data_loader, get_mnist_validation_data_loader
from utils import device, create_weights_init_fn
from distillation import ClassificationDistillationTrainer

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


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


def distillation():
    mlp = MLP(784, 10)
    init_fn = create_weights_init_fn(torch.nn.init.normal_, mean=0, std=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = ClassificationDistillationTrainer(mlp, 100, (28, 28), init_fn, 0.0001, 10, loss_fn, 0.0001, device)

    mnist = get_mnist_training_data_loader('./data/', 20)

    distilled_images, distilled_labels, _ = trainer.distill_(mnist, 10, 1)
    distilled_images = list(torch.split(distilled_images, 1, dim=0))
    distilled_labels = list(torch.split(distilled_labels, 1))
    to_pil_img = transforms.ToPILImage()

    for img, label in zip(distilled_images, distilled_labels):
        img = to_pil_img(img)
        img.save('./output/' + str(label) + '.jpg')


if __name__ == '__main__':
    distillation()
