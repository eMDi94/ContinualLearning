from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as T


def get_mnist_data_loader(root, batch_size, training=True):
    trfs = T.Compose([
        T.ToTensor()
    ])
    mnist = datasets.MNIST(root=root, train=training, transform=trfs, download=True)
    loader = DataLoader(mnist, batch_size, shuffle=True)
    return loader
