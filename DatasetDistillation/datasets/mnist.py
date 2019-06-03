from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_mnist_training_data_loader(root, batch_size):
    trfs = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist = datasets.MNIST(root=root, train=True, transform=trfs, download=True)
    loader = DataLoader(mnist, batch_size, shuffle=True)
    return loader


def get_mnist_validation_data_loader(root, batch_size):
    trfs = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist = datasets.MNIST(root=root, train=False, transform=trfs, download=True)
    loader = DataLoader(mnist, batch_size, shuffle=True)
    return loader
