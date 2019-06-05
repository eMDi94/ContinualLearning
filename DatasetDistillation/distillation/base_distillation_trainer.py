

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

    def train(self, *args):
        raise NotImplementedError()
