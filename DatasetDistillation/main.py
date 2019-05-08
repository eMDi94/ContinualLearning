import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

import torchvision
import torchvision.transforms as transforms

from distillation import distill_dataset
from nn import get_nn
from parameters import *


def main():
    model = get_nn()
    tfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
    mnist = torchvision.datasets.MNIST(root='./data/mnist/train', train=True, transform=tfs, download=True)
    sampler = RandomSampler(TensorDataset(torch.arange(len(mnist))), True)
    data_loader = DataLoader(mnist, batch_size, sampler=sampler, num_workers=num_workers)
    data_size = mnist[0][0].size()
    distilled_dataset, distilled_lr = distill_dataset(model, weights_distribution, weights_batch_size,
                                                      distilled_dataset_size, step_size, optimization_iterations,
                                                      lr0, data_loader, data_size, criterion)
    # TODO (Save the distilled dataset and the learning rate)


if __name__ == '__main__':
    main()
