import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

import torchvision
import torchvision.transforms as transforms

from distillation import distill_dataset
from cnn.nn import get_nn


def main_cnn():
    model = get_nn()
    tfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
    mnist = torchvision.datasets.MNIST(root='./data/mnist/train', train=True, transform=tfs, download=True)
    sampler = RandomSampler(TensorDataset(torch.arange(len(mnist))), True)
    data_loader = DataLoader(mnist, 10, sampler=sampler, num_workers=4)
    data_size = mnist[0][0].size()
    # TODO (Save the distilled dataset and the learning rate)
