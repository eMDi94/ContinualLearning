import torch


def normalize_0_1(tensor):
    with torch.no_grad():
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max()
    return tensor
