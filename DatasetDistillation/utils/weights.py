import torch.nn as nn


def create_weights_init_fn(init_fn, **kwargs):
    def i_init_fn(module):
        if hasattr(module, 'bias') is True:
            nn.init.constant_(module.bias, 0.)
        if hasattr(module, 'weight') is True:
            init_fn(module.weight, **kwargs)

    return i_init_fn
