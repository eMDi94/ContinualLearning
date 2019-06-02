import torch
import torch.nn as nn
from contextlib import contextmanager


class FlattenReparametrizationModule(nn.Module):

    def __init__(self):
        super(FlattenReparametrizationModule, self).__init__()
        self.weights_modules_names = None
        self.weights_numels = None
        self.weights_shapes = None
        self.flat_w = None

    def flat_all(self):
        # Collect modules and relative names and save them in the structure
        w_modules_names = []
        for m in self.modules():
            for n, p in m.named_parameters():
                if p is not None:
                    w_modules_names.append((m, n))
        self.weights_modules_names = w_modules_names

        ws = tuple(m._parameters[n].detach() for m, n in self.weights_modules_names)

        # Reparam to a single flat parameter
        self.weights_numels = tuple(w.numel() for w in ws)
        self.weights_shapes = tuple(w.size() for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        # Remove old parameters and add the names as buffers
        for m, n in self.weights_modules_names:
            delattr(m, n)
            self.register_buffer(n, None)

        # Register the flat weights
        self.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

    @contextmanager
    def unflatten_weights(self, flat_w):
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self.weights_numels), self.weights_shapes))
        for (m, n), w in zip(self.weights_modules_names, ws):
            setattr(m, n, w)
        yield
        for m, n in self.weights_modules_names:
            setattr(m, n, None)

    def forward_with_params(self, x, new_flat_w):
        with self.unflatten_weights(new_flat_w):
            return nn.Module.__call__(self, x)

    def __call__(self, x):
        return self.forward_with_params(x, self.flat_w)
