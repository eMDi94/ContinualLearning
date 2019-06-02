import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from .base import FlattenReparametrizationModule
from .utils import create_weights_init_fn


class ClassificationDistillationModule(FlattenReparametrizationModule):

    def __init__(self, input_shape, optimization_iterations, training_dataset, training_batch_size, weights_batch_size,
                 weights_init_fn, distillation_lr, n_labels, examples_per_label, weights_init_fn_params=None):
        super(ClassificationDistillationModule, self).__init__()
        self.input_shape = input_shape
        self.labels = torch.arange(n_labels).repeat(examples_per_label).reshape(examples_per_label, -1).transpose(1, 0).reshape(-1)
        self.weights_init_fn = create_weights_init_fn(weights_init_fn, **weights_init_fn_params)
        self.T = optimization_iterations
        self.dataset = training_dataset
        self.training_batch_size = training_batch_size
        self.weights_batch_size = weights_batch_size
        self.eta = distillation_lr

    @property
    def M(self):
        return self.labels.size()

    def perform_distillation_and_train(self):
        pass
