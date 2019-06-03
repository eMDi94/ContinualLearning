import torch
import torch.nn as nn

from utils import MetaModelUtils


class ClassificationDistillationTrainer(object):

    def __init__(self, model, optimization_iterations, data_size, weights_init_fn, learning_rate, device):
        self.model = model
        self.T = optimization_iterations
        self.device = device
        self.weights_init_fn = weights_init_fn
        self.eta = learning_rate
        self.data_size = data_size

    def __get_next_batch(self, it, loader):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        return b, it

    def distill_and_train(self, training_data_loader, n_labels, exemples_per_label):
        distilled_labels = torch.arange(n_labels).repeat(exemples_per_label).reshape(exemples_per_label, 1)\
            .transpose(1, 0).reshape(-1)
        distilled_data_size = torch.Size([distilled_labels.size(0)]) + torch.Size(self.data_size)
        distilled_data = torch.randn(distilled_data_size, device=self.device, requires_grad=True, dtype=torch.float)
        eta = torch.tensor([self.eta], requires_grad=True, device=self.device)

        it = iter(training_data_loader)
        for iteration in range(self.T):
            data, labels = self.__get_next_batch(it, training_data_loader)
            data = data.to(self.device)
            labels = labels.to(self.device)
