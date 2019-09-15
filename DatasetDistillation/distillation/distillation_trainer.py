from itertools import cycle

import torch
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils.meta_model_utils import MetaModelUtils
from utils.numeric import normalize_0_1

import torchvision.transforms as T


class DistillationTrainer(object):

    CONSTANT_WEIGHTS_INIT = 0
    RANDOM_WEIGHTS_INIT = 1
    WEIGHTS_INIT_MODALITIES = (CONSTANT_WEIGHTS_INIT, RANDOM_WEIGHTS_INIT, )

    def __init__(self, model, device, loss_fn):
        self.model = model
        self.model = self.model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.numels = torch.tensor([w.numel() for w in self.model.parameters()], device=self.device).sum().item()
        self.summary_writer = SummaryWriter()

    def init_weights(self, mode, weights_init_fn):
        if mode == self.CONSTANT_WEIGHTS_INIT:
            if hasattr(self, 'initial_weights'):
                flat_weights = getattr(self, 'initial_weights').clone()
                flat_weights.requires_grad_(True)
                MetaModelUtils.set_flat_params(self.model, flat_weights)
                self.model.apply(weights_init_fn)
                return flat_weights
            else:
                flat_weights = torch.empty(self.numels, device=self.device, dtype=torch.float)
                setattr(self, 'initial_weights', flat_weights)
                flat_weights = flat_weights.clone()
                flat_weights.requires_grad_(True)
                MetaModelUtils.set_flat_params(self.model, flat_weights)
                self.model.apply(weights_init_fn)
                return flat_weights
        elif mode == self.RANDOM_WEIGHTS_INIT:
            flat_weights = torch.empty(self.numels, device=self.device, dtype=torch.float,
                                       requires_grad=True)
            MetaModelUtils.set_flat_params(self.model, flat_weights)
            self.model.apply(weights_init_fn)
            return flat_weights
        else:
            raise Exception('Init weights modality not supported')

    def log_distilled_data(self, distilled_data, iteration):
        img = make_grid(distilled_data, 10)
        self.summary_writer.add_image('distilled-data', img, global_step=iteration)

    def create_distilled_data(self, lr, data_size, n_labels, examples_per_label, mean, std):
        # Distilled data in the form [0, ..., n_labels - 1] repeated for examples_per_label times
        distilled_labels = torch.arange(n_labels, device=self.device).repeat(examples_per_label) \
            .reshape((examples_per_label, -1)).transpose(1, 0).reshape(-1)
        # Create the distilled data, find the size, initialize with normal and then require grad
        distilled_data_size = distilled_labels.size() + torch.Size(data_size)
        distilled_data = torch.empty(distilled_data_size, device=self.device, dtype=torch.float)
        distilled_data.normal_(mean, std)
        distilled_data = normalize_0_1(distilled_data)
        eta = torch.tensor([lr], dtype=torch.float, device=self.device)
        return distilled_data, distilled_labels, eta

    def training_batch_distillation(self, optimization_iterations, distilled_data, eta, distilled_labels,
                                    alpha, training_data_loader, n_input_batches, weights_batch_size,
                                    weights_init_type, weights_init_fn, save_data_after, log_loss_after):
        for iteration in range(optimization_iterations):
            print('Optimization iteration ' + str(iteration) + ' started...')

            distilled_data.requires_grad_(True)
            eta.requires_grad_(True)

            eta_grad_accumulator = torch.zeros_like(eta, device=self.device)
            distilled_data_grad_accumulator = torch.zeros_like(distilled_data, device=self.device)

            for weights_batch_number in range(weights_batch_size):
                flat_weights = self.init_weights(weights_init_type, weights_init_fn)

                out = self.model(distilled_data)
                loss = self.loss_fn(out, distilled_labels)
                print('Distilled data loss: ', loss.item())

                if weights_batch_number % log_loss_after == 0:
                    self.summary_writer.add_scalar('Distilled data loss', loss,
                                                   iteration * weights_batch_size + weights_batch_number)

                (weights_grad,) = autograd.grad(loss, flat_weights, retain_graph=True, create_graph=True)
                flat_weights = flat_weights - eta * weights_grad
                MetaModelUtils.set_flat_params(self.model, flat_weights)

                it = cycle(training_data_loader)
                losses = torch.empty(n_input_batches, device=self.device)
                for n in range(n_input_batches):
                    data, labels = next(it)
                    data, labels = data.to(self.device), labels.to(self.device)

                    out = self.model(data)
                    loss = self.loss_fn(out, labels)
                    print('Real data loss. Iteration ' + str(iteration) + '. Weight batch '
                          + str(weights_batch_number) + ' Data batch ' + str(n) + ': ', loss.item())
                    losses[n] = loss.item()

                    distilled_grad, eta_grad = autograd.grad(loss, (distilled_data, eta), retain_graph=True)
                    distilled_data_grad_accumulator += distilled_grad
                    eta_grad_accumulator += eta_grad
                loss.detach()

                if weights_batch_number % log_loss_after == 0:
                    self.summary_writer.add_scalar('Real data mean loss', losses.mean().item(),
                                                   iteration * weights_batch_size + weights_batch_number)

            # Compute the mean on the gradients accumulators and update the distilled data
            distilled_data = distilled_data.detach().add_(-alpha * distilled_data_grad_accumulator)
            distilled_data = normalize_0_1(distilled_data)
            eta = eta.detach().add_(-alpha * eta_grad_accumulator)

            if (iteration + 1) % save_data_after == 0:
                self.log_distilled_data(distilled_data, iteration + 1)
            self.summary_writer.add_scalar('eta', eta.item(), iteration)

        return distilled_data, distilled_labels, eta

    def simple_paper_distillation(self, optimization_iterations, distilled_data, eta, distilled_labels,
                                  alpha, training_data_loader, weights_batch_size, weights_init_type,
                                  weights_init_fn, save_data_after, log_loss_after):
        it = cycle(training_data_loader)
        for iteration in range(optimization_iterations):
            print('Optimization iteration ' + str(iteration) + ' started...')

            data, labels = next(it)
            data, labels = data.to(self.device), labels.to(self.device)

            distilled_data.requires_grad_(True)
            eta.requires_grad_(True)
            distilled_data_grad_accumulator = torch.zeros_like(distilled_data, device=self.device)
            eta_grad_accumulator = torch.zeros_like(eta, device=self.device)

            for weights_batch_number in range(weights_batch_size):
                flat_weights = self.init_weights(weights_init_type, weights_init_fn)

                out = self.model(distilled_data)
                loss = self.loss_fn(out, distilled_labels)
                print('Distilled data loss: ', loss.item())

                if weights_batch_number % log_loss_after == 0:
                    self.summary_writer.add_scalar('Distilled data loss', loss,
                                                   iteration * weights_batch_size + weights_batch_number)

                (weights_grad,) = autograd.grad(loss, flat_weights, retain_graph=True, create_graph=True)
                flat_weights = flat_weights - eta * weights_grad
                MetaModelUtils.set_flat_params(self.model, flat_weights)

                out = self.model(data)
                loss = self.loss_fn(out, labels)
                (distilled_grad, eta_grad) = autograd.grad(loss, (distilled_data, eta))
                print('Real data loss: ', loss.item())

                if weights_batch_number % log_loss_after == 0:
                    self.summary_writer.add_scalar('Real data loss', loss,
                                                   iteration * weights_batch_size + weights_batch_number)

                loss.detach()

                distilled_data_grad_accumulator += distilled_grad
                eta_grad_accumulator += eta_grad

            distilled_data = distilled_data.detach().add_(-alpha * distilled_data_grad_accumulator)
            eta = eta.detach().add_(-alpha * eta_grad_accumulator)

            if (iteration + 1) % save_data_after == 0:
                self.log_distilled_data(distilled_data, iteration + 1)
            self.summary_writer.add_scalar('eta', eta.item(), iteration)

        return distilled_data, distilled_labels, eta

    def classification_distillation(self, lr, alpha, training_data_loader, optimization_iterations,
                                    data_size, n_labels, examples_per_label, weights_init_type,
                                    weights_init_fn, weights_batch_size, n_input_batches=32,
                                    mean=0.0, std=1.0, save_data_after=50, log_loss_after=10):
        distilled_data, distilled_labels, eta = self.create_distilled_data(lr, data_size, n_labels,
                                                                           examples_per_label, mean, std)

        grid = make_grid(distilled_data.cpu(), 10)
        T.ToPILImage()(grid).save('./initial.jpg')

        # Create the learning rate tensor
        self.log_distilled_data(distilled_data, 0)

        # Set the model in train mode
        self.model.train()

        distilled_data, distilled_labels, eta = \
            self.training_batch_distillation(optimization_iterations, distilled_data, eta, distilled_labels,
                                             alpha, training_data_loader, n_input_batches, weights_batch_size,
                                             weights_init_type, weights_init_fn, save_data_after,
                                             log_loss_after)

        self.model.eval()
        grid = make_grid(distilled_data.detach().cpu(), 10)
        T.ToPILImage()(grid).save('./final.jpg')
