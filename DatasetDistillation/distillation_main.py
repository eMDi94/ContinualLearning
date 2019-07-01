from __future__ import print_function
import os

from networks import MLP, LeNet
from datasets import get_mnist_training_data_loader, get_mnist_validation_data_loader
from utils import device, create_weights_init_fn, FlatTransform
from distillation import ClassificationDistillationTrainer

import torch
import torchvision.transforms as T


def weights_init(device):
    def closure(numel):
        t = torch.empty(numel, device=device, dtype=torch.float, requires_grad=True)
        torch.nn.init.normal_(t, 0, 1)
        return t
    return closure


def save_mlp_data(distilled_data, distilled_targets):
    to_pil_img = T.ToPILImage()
    for img, label in zip(distilled_data, distilled_targets):
        img = img.squeeze()
        img = img.view(-1, 1, 28, 28)
        img = to_pil_img(img)
        img.save('./output/' + label + '.jpg')


def save_lenet_data(distilled_data, distilled_targets):
    to_pil_img = T.ToPILImage()
    for img, label in zip(distilled_data, distilled_targets):
        img = img.squeeze()
        img = to_pil_img(img)
        img.save('./output/' + label + '.jpg')


def distillation():
    mlp = MLP(784, 10)
    net = LeNet()
    init_fn = create_weights_init_fn(torch.nn.init.normal_, mean=0, std=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = ClassificationDistillationTrainer(mlp, 1000, (784,), init_fn, 0.001, 100, loss_fn, 0.0001, device)

    mnist = get_mnist_training_data_loader('./data/', 20)

    mnist.dataset.transform = T.Compose([
        mnist.dataset.transform,
        FlatTransform()
    ])

    trainer.distill(mnist, 10, 1)
    if not os.path.isdir('./output'):
        os.mkdir('./output')

    print('Saving distilled images...')
    trainer.save_data(save_mlp_data)
    print('Images saved.')


if __name__ == '__main__':
    distillation()
