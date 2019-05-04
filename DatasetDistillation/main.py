import torchvision

from distillation import dataset_distillation
from nn import get_nn


def main():
    nn = get_nn()
    mnist = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True)



if __name__ == '__main__':
    main()
