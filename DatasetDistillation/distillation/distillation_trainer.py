from itertools import cycle

import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as T
from sklearn.cluster import KMeans

from utils.meta_model_utils import MetaModelUtils


class DistillationTrainer(object):

    __OPTIMIZERS = {
        'sgd': optim.SGD,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adadelta': optim.Adadelta
    }
    OPTIMIZERS = __OPTIMIZERS.keys()
    DISTILLED_DATA_INITIALIZATION = ('random-real', 'k-means', 'average-real',)

    def __init__(self, model, device):
        self.model = model
        self.model = self.model.to(device)
        self.device = device
        self.numels = torch.tensor([w.numel() for w in self.model.parameters()], device=self.device).sum().item()

    def save_grid_img(self, tensor, nrow, img_name):
        img = make_grid(tensor.detach().cpu(), nrow)
        T.ToPILImage()(img).save(img_name)

    @classmethod
    def create_distilled_optimizer(cls, params, lr, opt_name, opt_args):
        if opt_name not in DistillationTrainer.OPTIMIZERS:
            raise ValueError('The given optimizer is not recognized')
        else:
            optimizer_class = DistillationTrainer.__OPTIMIZERS[opt_name]
            optimizer = optimizer_class(params, lr, **opt_args)
            return optimizer

    def create_distilled_data(self, distilled_init, initial_eta, training_data_loader, n_labels, examples_per_label):
        if distilled_init not in DistillationTrainer.DISTILLED_DATA_INITIALIZATION:
            raise ValueError('Initialization of distillation data not recognized')

        distilled_labels = torch.arange(n_labels, device=self.device).repeat(examples_per_label)
        training_data = []
        training_labels = []
        for data, labels in training_data_loader:
            training_data.append(data)
            training_labels.append(labels)
        training_data = torch.cat(training_data)
        training_labels = torch.cat(training_labels)
        training_data, training_labels = training_data.to(self.device), training_labels.to(self.device)

        distilled_data = torch.empty(distilled_labels.size(0), *training_data.size()[1:], device=self.device,
                                     dtype=torch.float)
        eta = torch.tensor([initial_eta], device=self.device, requires_grad=True, dtype=torch.float)
        if distilled_init == 'average-real':
            for label in range(n_labels):
                m = training_data[training_labels == label].mean(dim=0)
                distilled_data[distilled_labels == label] = m
        elif distilled_init == 'random-real':
            for label in range(n_labels):
                loader = DataLoader(TensorDataset(training_data[training_labels == label]),
                                    batch_size=examples_per_label, shuffle=True)
                distilled_data[distilled_labels == label] = next(iter(loader))[0]
        else:
            kmeans = KMeans(n_clusters=n_labels)
            t_data = training_data.view(training_data.size(0), -1).cpu().numpy()
            kmeans.fit_transform(t_data, training_labels.cpu().numpy())
            for label, center in enumerate(kmeans.cluster_centers_):
                distilled_data[distilled_labels == label] = torch.from_numpy(center).view(*training_data.size()[1:])\
                    .to(self.device)
        distilled_data.requires_grad_(True)
        return distilled_labels, distilled_data, eta

    def distill(self, optimization_iterations, weights_batch_size, weights_init_fn, training_data_loader,
                n_labels, examples_per_label, alpha, initial_eta, loss_fn, distilled_init, distilled_optimizer_name,
                distilled_optimizer_args=None, log_img_after=10, log_directory='./log/'):
        d_opt_args = distilled_optimizer_args if distilled_optimizer_args is not None else {}
        # Create distilled data
        distilled_labels, distilled_data, eta = self.create_distilled_data(distilled_init, initial_eta,
                                                                           training_data_loader, n_labels,
                                                                           examples_per_label)
        # For debug
        self.save_grid_img(distilled_data, 10, 'initial.jpg')
        distilled_data.requires_grad_(True)
        eta = torch.tensor([initial_eta], device=self.device, requires_grad=True, dtype=torch.float)

        # Define the optimizer for the distilled_data
        optimizer = DistillationTrainer.create_distilled_optimizer((distilled_data, eta), alpha,
                                                                   distilled_optimizer_name,
                                                                   d_opt_args)

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
                print('Loss on distilled data: ', loss.item())

                # Update the weights
                (weights_grad, ) = autograd.grad(loss, flat_weights, retain_graph=True, create_graph=True)
                flat_weights = flat_weights - eta * weights_grad
                MetaModelUtils.set_flat_params(self.model, flat_weights)

                # Compute the output on the training data
                out = self.model(data)
                loss = loss_fn(out, labels)
                print('Loss on real data: ', loss.item())

                # Accumulate the gradient on distilled data
                loss.backward()

            # Perform the step on the distilled data
            optimizer.step()
            if (iteration + 1) % log_img_after == 0:
                self.save_grid_img(distilled_data, 10, log_directory + 'img' + str(iteration) + '.jpg')

        self.save_grid_img(distilled_data, 10, 'final.jpg')
