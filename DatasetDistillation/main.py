import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

import torchvision
import torchvision.transforms as transforms

from distillation import distill_dataset
from nn import get_nn


def main():
    model = get_nn()
    tfs = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=tfs)
    img_size = mnist[0][0].size()
    sampler = RandomSampler(TensorDataset(torch.arange(len(mnist), dtype=torch.int)))
    data_loader = DataLoader(mnist, batch_size=100, sampler=sampler, num_workers=2)
    distilled_dataset = distill_dataset(model, nn.init.uniform_, 50, 20, 0.001, 1000, 0.001, data_loader, img_size,
                                        nn.Softmax(10))


if __name__ == '__main__':
    main()
