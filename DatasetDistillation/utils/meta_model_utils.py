from functools import reduce
from operator import mul

import torch
import torch.nn as nn


class MetaModelUtils:

    AVAIL_MODULES = [nn.Linear, nn.Conv2d, nn.BatchNorm2d]

    @staticmethod
    def reset(model):
        """ Breaks history of params. """
        for module in model.modules():
            if module.__class__ in MetaModelUtils.AVAIL_MODULES:
                module.weight = nn.Parameter(module.weight.data)
                if module.bias is not None:
                    module.bias = nn.Parameter(module.bias.data)

    @staticmethod
    def get_flat_params(model):
        params = []
        for module in model.modules():
            if module.__class__ in MetaModelUtils.AVAIL_MODULES:
                params.append(module.weight.view(-1))
                if module.bias is not None:
                    params.append(module.bias.view(-1))
        return torch.cat(params)

    @staticmethod
    def set_flat_params(model, flat_params):
        offs = 0
        for i, module in enumerate(model.modules()):
            if module.__class__ in MetaModelUtils.AVAIL_MODULES:
                for k in module._parameters.keys():
                    assert k in ['weight', 'bias']
                    p = module._parameters[k]
                    if p is not None:
                        p_shape = module._parameters[k].size()
                        p_flat_shape = reduce(mul, p_shape, 1)
                        module._parameters[k] = \
                            flat_params[offs:offs+p_flat_shape].view(*p_shape)
                        offs += p_flat_shape

    @staticmethod
    def get_flat_grads(model_with_grads):
        grads = []
        for module in model_with_grads.modules():
            if module.__class__ in MetaModelUtils.AVAIL_MODULES:
                grads.append(module.weight.grad.view(-1, 1))
                if module.bias is not None:
                    grads.append(module.bias.grad.view(-1, 1))
        return torch.cat(grads)
