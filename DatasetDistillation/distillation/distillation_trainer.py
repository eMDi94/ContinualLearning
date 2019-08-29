from itertools import cycle
import os

import numpy as np
import torch
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as T
from torchvision.utils import make_grid

from utils.meta_model_utils import MetaModelUtils
from utils.folder import create_folder_if_not_exists, append_separator_if_needed

from .distillation_dataset import DistillationDataset


class DistillationTrainer(object):

    CLASSIFICATION_DISTILLATION_TYPE = 'Classification'
    REGRESSION_DISTILLATION_TYPE = 'Regression'
    DISTILLATION_TYPES = (CLASSIFICATION_DISTILLATION_TYPE, REGRESSION_DISTILLATION_TYPE,)

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
        self.__distillation_type = None
        self.__summary_writer = SummaryWriter()

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

    def __in_range_0_1(self, tensor):
        with torch.no_grad():
            tensor = tensor - tensor.min()
            tensor = tensor / tensor.max()
        return tensor

    def __create_batches(self, distilled_data, distilled_labels, distilled_data_batch_size):
        distillation_data_loader = DataLoader(DistillationDataset(distilled_data, distilled_labels),
                                              batch_size=distilled_data_batch_size, shuffle=True)
        distillation_batches = []
        for data_batch, labels_batch in distillation_data_loader:
            data_batch.requires_grad_(True)
            distillation_batches.append((data_batch, labels_batch, torch.zeros_like(data_batch, device=self.device)))

        return distillation_batches

    def classification_distillation(self, training_data_loader, n_labels, examples_per_label,
                                    mean=0.0, std=1.0, distilled_data_batch_size=64,
                                    save_image_after=50, log_loss_after=10):
        if self.__distillation_type is not None and self.__distillation_type == self.CLASSIFICATION_DISTILLATION_TYPE:
            raise ValueError("This distillator has already been used for regression. "
                             "It's not possible to use it for classification now.")
        else:
            self.__distillation_type = self.CLASSIFICATION_DISTILLATION_TYPE
        # Create the distilled data variables
        distilled_labels = torch.arange(n_labels, device=self.device).repeat(examples_per_label)
        distilled_labels = distilled_labels.reshape((examples_per_label, -1)).transpose(1, 0).reshape((-1,))
        distilled_data_size = distilled_labels.size() + torch.Size(tuple(self.data_size))
        distilled_data = torch.empty(distilled_data_size, device=self.device, dtype=torch.float)
        # nn.init.normal_(distilled_data, mean, std)
        distilled_data.normal_(mean=mean, std=std)
        distilled_data = self.__in_range_0_1(distilled_data)
        # Divide the distilled data and labels in batches
        distillation_batches = self.__create_batches(distilled_data, distilled_labels, distilled_data_batch_size)
        eta = torch.tensor([self.eta], device=self.device, requires_grad=True)

        # Set the model in training mode
        self.model.train()

        # Now start the computation
        it = cycle(training_data_loader)
        for iteration in range(self.T):
            print('Optimization iteration ' + str(iteration) + ' started...')

            data, labels = next(it)

            # Send the data to the selected device
            data, labels = data.to(self.device), labels.to(self.device)

            eta_grad_accumulator = torch.zeros_like(eta, device=self.device, dtype=torch.float)
            for weights_batch_index in range(self.weights_batch_size):
                flat_weights = torch.empty(self.numels, dtype=torch.float, device=self.device, requires_grad=True)
                MetaModelUtils.set_flat_params(self.model, flat_weights)
                self.model.apply(self.weights_init_fn)

                for data_batch, labels_batch, _ in distillation_batches:
                    self.model.zero_grad()
                    out = self.model(data_batch)
                    loss = self.loss_fn(out, labels_batch)

                    (flat_grads, ) = autograd.grad(loss, flat_weights, retain_graph=True, create_graph=True)
                    flat_weights = flat_weights - eta * flat_grads
                    MetaModelUtils.set_flat_params(self.model, flat_weights)

                out = self.model(data)
                loss = self.loss_fn(out, labels)
                grads = autograd.grad(loss, [batches[0] for batches in distillation_batches] + [eta])
                loss.detach()

                if (iteration * self.weights_batch_size + weights_batch_index) % log_loss_after == 0:
                    self.__summary_writer.add_scalar('Final Loss', loss.item(),
                                                     iteration * self.weights_batch_size + weights_batch_index)

                eta_grad_accumulator.add_(grads[-1])
                for i in range(len(grads) - 1):
                    g = grads[i]
                    grad_accumulator = distillation_batches[i][-1]
                    grad_accumulator.add_(g)

                if (weights_batch_index + 1) % 50 == 0:
                    print('Working...')

            eta.data.add_(-self.alpha * eta_grad_accumulator)
            new_batches = []
            for data_batch, labels_batch, grad in distillation_batches:
                data_batch.data.add_(-self.alpha * grad)
                # d = self.__in_range_0_1(data_batch)
                # d.requires_grad_(True)
                new_batches.append((data_batch, labels_batch, torch.zeros_like(data_batch),))
            distillation_batches = new_batches

            if iteration % save_image_after == 0:
                with torch.no_grad():
                    datas = [d[0] for d in distillation_batches]
                    datas = torch.cat(datas)
                    grid = make_grid(datas, 10)
                    self.__summary_writer.add_image('Distilled Images', grid, iteration)

        print('Distillation Ended')
        self.model.eval()
        distilled_data = [d[0].detach() for d in distillation_batches]
        self._distilled_data = torch.cat(distilled_data)
        self._distilled_targets = distilled_labels.detach()
        self._distilled_learning_rate = eta.detach().item()

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

    def save_distilled_data(self, output_name, save_single_item_fn, transformations=tuple()):
        if self.__distillation_type is None:
            raise ValueError("No data has been distilled. You cannot save anything")
        elif self.__distillation_type == self.REGRESSION_DISTILLATION_TYPE:
            np.savez_compressed(output_name, data=self.distilled_data.cpu().numpy(),
                                labels=self.distilled_targets.cpu().numpy(),
                                lr=self.distilled_learning_rate)
        else:
            create_folder_if_not_exists(output_name)
            t = T.Compose(transformations)
            unique_labels = torch.unique(self.distilled_targets)
            for label in unique_labels:
                label_directory = append_separator_if_needed(output_name) + str(label) + os.sep
                create_folder_if_not_exists(label_directory)
                for i, distilled_value in enumerate(self.distilled_data[self.distilled_targets == label]):
                    transformed_data = t(distilled_value)
                    save_single_item_fn(label_directory + str(i), transformed_data)
