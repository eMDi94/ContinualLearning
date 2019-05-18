import torch
import torch.nn as nn
from collections import namedtuple
from contextlib import contextmanager


ModelInformations = namedtuple('ModelInformations', 'weight_module_names weights_numel weights_shapes')


def patch_modules(model):
    # Save the modules of the model together with their names
    weight_module_names = []

    for m in model.modules():
        for n, p in model.named_parameters(recurse=False):
            if p is not None:
                weight_module_names.append((m, n))

    # Save the previous informations
    weight_module_names = tuple(weight_module_names)
    weights = tuple(m._parameters[n] for m, n in weight_module_names)
    weights_numel = tuple(w.numel() for w in weights)
    weights_shapes = tuple(w.shape for w in weights)

    # Flatten to a unique parameter
    with torch.no_grad():
        flat_weights = torch.cat([w.reshape(-1) for w in weights])

    # Remove all the old parameters and assign the names as buffers' name
    for m, n in weight_module_names:
        delattr(m, n)
        m.register_buffer(n, None)

    # Now register a new parameter composed by the flat weights
    model.register_parameter('flat_weights', nn.Parameter(flat_weights, requires_grad=True))

    return model, ModelInformations(weight_module_names=weight_module_names, weights_numel=weights_numel,
                                    weights_shapes=weights_shapes)


@contextmanager
def unflatten_weights(flat_weights, model_informations):
    weights_modules_names, weights_numel, weights_shapes = model_informations.weights_modules_names, model_informations.weights_numel, model_informations.weights_shapes
    weights = (t.view(s) for (t, s) in zip(flat_weights.split(weights_numel), weights_shapes))
    for (m, n), w in zip(weights_modules_names, weights):
        setattr(m, n, w)
    yield
    for m, n in weights_modules_names:
        setattr(m, n, None)
