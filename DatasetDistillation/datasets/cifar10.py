from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as T


def get_cifar10_data_loader(root, batch_size, training=True):
    trfs = T.Compose([
        T.ToTensor()
    ])
    cifar10 = datasets.cifar.CIFAR10(root=root, train=training, transform=trfs, download=True)
    loader = DataLoader(cifar10, batch_size, shuffle=True)
    return loader
