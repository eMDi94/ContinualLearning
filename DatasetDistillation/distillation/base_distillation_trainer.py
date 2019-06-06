import torch
import torch.optim as optim


class BaseDistillationTrainer(object):

    def __init__(self, model, optimization_iterations, data_size, weights_init_fn, learning_rate, weights_batch_size,
                 loss_fn, alpha, device):
        self.model = model.to(device)
        self.T = optimization_iterations
        self.device = device
        self.weights_init_fn = weights_init_fn
        self.eta = learning_rate
        self.weights_batch_size = weights_batch_size
        self.data_size = data_size
        self.loss_fn = loss_fn
        self.alpha = alpha
        self._distilled_data = None
        self._distilled_learning_rate = None
        self._distilled_targets = None
        self.numels = torch.tensor([w.numel() for w in self.model.parameters()], device=self.device).sum().item()

    @property
    def distilled_data(self):
        if self._distilled_data is None:
            raise ValueError('No distilled data are available.')
        return self._distilled_data

    @property
    def distilled_learning_rate(self):
        if self._distilled_learning_rate is None:
            raise ValueError('No distilled learning rate is available.')
        return self._distilled_learning_rate

    @property
    def distilled_targets(self):
        if self._distilled_targets is None:
            raise ValueError('No distilled targets are available.')
        return self._distilled_targets

    def _get_next_batch(self, it, loader):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        return b, it

    def distill(self, *args):
        raise NotImplementedError()

    def train(self, optimizer=optim.SGD, optimizer_args=None):
        if any(var is None for var in [self._distilled_data, self._distilled_targets, self._distilled_learning_rate]):
            raise RuntimeError('You cannot perform a training without having distilled any data.')

        # As the paper says, only one step of SGD is necessary. Anyway i allow the customization of the optimizer.
        # The default is the Stochastic Gradient Descent. optimizer_args allows to pass any extra arguments that
        # the optimizer requires. The learning rate is not allowed since the used lr will be the distilled one.
        if optim.Optimizer not in optimizer.__bases__:
            raise ValueError('Optimizer must be a subclass torch.optim.optimizer.Optimizer')

        optimizer_args = optimizer_args if optimizer_args is not None else dict()
        if not isinstance(optimizer_args, dict):
            raise ValueError('Only python dictionary are allowed as optimizer_args.')
        if 'lr' in optimizer_args.keys():
            raise ValueError("The learning rate is the one distilled. It's not possible to pass it as an argument.")

        self.distilled_data.to(self.device)
        self.distilled_targets.to(self.device)

        self.model.train()
        op = optimizer(self.model.parameters(), lr=self.distilled_learning_rate, **optimizer_args)

        op.zero_grad()
        out = self.model(self.distilled_data)
        loss = self.loss_fn(out, self.distilled_targets)
        loss.backward()
        op.step()
