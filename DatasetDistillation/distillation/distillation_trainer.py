from itertools import cycle

import torch
import torch.autograd as autograd
import torch.optim as optim
from torchvision.utils import make_grid

from utils.meta_model_utils import MetaModelUtils

import torchvision.transforms as T


class DistillationTrainer(object):

    def __init__(self, model, device):
        self.model = model
        self.model = self.model.to(device)
        self.device = device
        self.numels = torch.tensor([w.numel() for w in self.model.parameters()], device=self.device).sum().item()

    def save_grid_img(self, tensor, nrow, img_name):
        img = make_grid(tensor.detach().cpu(), nrow)
        T.ToPILImage()(img).save(img_name)

    def distill(self, optimization_iterations, weights_batch_size, weights_init_fn, training_data_loader,
                n_labels, examples_per_label, alpha, initial_eta, loss_fn):
        # Create distilled data
        distilled_labels = torch.arange(n_labels, device=self.device).repeat(examples_per_label)\
            .reshape(examples_per_label, -1).transpose(1, 0).reshape(-1)
        training_data = torch.cat([data for (data, _) in training_data_loader])
        training_labels = torch.cat([labels for (_, labels) in training_data_loader])
        training_data, training_labels = training_data.to(self.device), training_labels.to(self.device)
        distilled_data = torch.empty(distilled_labels.size(0), *training_data.size()[1:], device=self.device,
                                     dtype=torch.float)
        for label in range(n_labels):
            m = training_data[training_labels == label].mean(dim=0)
            distilled_labels_map = distilled_labels == label
            distilled_data[distilled_labels_map] = m
        # For debug
        self.save_grid_img(distilled_data, 10, 'initial.jpg')
        distilled_data.requires_grad_(True)
        eta = torch.tensor([initial_eta], device=self.device, requires_grad=True, dtype=torch.float)

        # Define the optimizer for the distilled_data
        optimizer = optim.Adagrad((distilled_data, eta), lr=alpha)

        # Main cycle
        self.model.train()
        it = cycle(training_data_loader)
        for iteration in range(optimization_iterations):
            print('Optimization iteration ' + str(iteration + 1) + ' started...')

            data, labels = next(it)
            data, labels = data.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            for weight_batch_number in range(weights_batch_size):
                flat_weights = torch.empty(self.numels, device=self.device, requires_grad=True)
                MetaModelUtils.set_flat_params(self.model, flat_weights)
                self.model.apply(weights_init_fn)

                # Compute the output with the distilled data
                out = self.model(distilled_data)
                loss = loss_fn(out, distilled_labels)

                # Update the weights
                (weights_grad, ) = autograd.grad(loss, flat_weights, retain_graph=True, create_graph=True)
                flat_weights = flat_weights - eta * weights_grad
                MetaModelUtils.set_flat_params(self.model, flat_weights)

                # Compute the output on the training data
                out = self.model(data)
                loss = loss_fn(out, labels)
                print('Loss on distilled data: ', loss.item())

                # Accumulate the gradient on distilled data
                loss.backward()
                print('Loss on real data: ', loss.item())

            # Perform the step on the distilled data
            optimizer.step()

        self.save_grid_img(distilled_data, 10, 'final.jpg')
