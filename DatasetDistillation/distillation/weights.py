import torch.nn as nn


def create_weights_init(init_fn, **kwargs):
    """
    Create a function to initialize the weights
    :param init_fn: One of torch.nn.init functions
    :param kwargs: Extra arguments that are needed by the init_fn function
    :return: A function that accepts a tensor and modify it in place
    """
    def init(model):
        class_name = model.__class__.__name__
        if class_name.startswith('Conv') or class_name == 'Linear':
            if getattr(model, 'bias', None) is not None:
                nn.init.normal_(model.bias)
            if getattr(model, 'weight', None) is not None:
                init_fn(model.weight, **kwargs)
        elif 'Norm' in class_name:
            if getattr(model, 'weight', None) is not None:
                model.weight.data.fill_(1.)
            if getattr(model, 'bias', None) is not None:
                model.bias.data.zero_()
    return init


def init_model(model, init_fn, **kwargs):
    """
    Initialize the model parameters
    :param model:
    :param init_fn:
    :param kwargs:
    :return:
    """
    init = create_weights_init(init_fn, **kwargs)
    model.apply(init)
    return model
