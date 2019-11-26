from .mnist import get_mnist_data_loader
from .cifar10 import get_cifar10_data_loader


AVAILABLE_DATASET = ('mnist', 'cifar10',)


def select_dataset(dataset_name, root, batch_size, training):
    if dataset_name == AVAILABLE_DATASET[0]:
        return get_mnist_data_loader(root, batch_size, training), 1, 28
    elif dataset_name == AVAILABLE_DATASET[1]:
        return get_cifar10_data_loader(root, batch_size, training), 3, 32
    else:
        raise ValueError('database_name not recognized')
