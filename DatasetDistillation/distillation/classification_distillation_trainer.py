import torch
import torch.nn as nn

from utils import MetaModelUtils


class ClassificationDistillationTrainer(object):

    def __init__(self, optimization_iterations, n_labels, exemples_per_label, data_size,
                 weights_init_fn, learning_rate, training_data_loader, device):
        self.T = optimization_iterations
        self.device = device
        self.labels = torch.arange(n_labels, device=device).repeat(exemples_per_label).reshape(exemples_per_label, -1).transpose(1, 0).reshape(-1)
        self.weights_init_fn = weights_init_fn
        self.eta = learning_rate
        self.data_loader = training_data_loader

