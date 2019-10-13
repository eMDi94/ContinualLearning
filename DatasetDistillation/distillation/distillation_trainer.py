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
    DISTILLED_DATA_INITIALIZATION = ('random-real', 'k-means', 'average-real', 'random',)

    def __init__(self, model, device, loss_fn):
        self.model = model
        self.model = self.model.to(device)
        self.device = device
        self.numels = torch.tensor([w.numel() for w in self.model.parameters()], device=self.device).sum().item()
        self.loss_fn = loss_fn

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
        elif distilled_init == 'random':
            distilled_data = torch.randn_like(distilled_data, device=self.device, requires_grad=True, dtype=torch.float)
        else:
            kmeans = KMeans(n_clusters=n_labels)
            t_data = training_data.view(training_data.size(0), -1).cpu().numpy()
            kmeans.fit_transform(t_data, training_labels.cpu().numpy())
            for label, center in enumerate(kmeans.cluster_centers_):
                distilled_data[distilled_labels == label] = torch.from_numpy(center).view(*training_data.size()[1:])\
                    .to(self.device)
        distilled_data.requires_grad_(True)
        return distilled_labels, distilled_data, eta

    def distill(self, optimization_iterations, gd_steps, weights_init_fn, training_data_loader,
                n_labels, examples_per_label, n_batch, alpha, initial_eta, distilled_init, distilled_optimizer_name,
                distilled_optimizer_args=None, log_img_after=10, log_directory='./log/'):
        d_opt_args = distilled_optimizer_args if distilled_optimizer_args is not None else {}
        # Create distilled data
        data_steps = []
        params = []
        for _ in range(n_batch):
            distilled_labels, distilled_data, eta = self.create_distilled_data(distilled_init, initial_eta,
                                                                           training_data_loader, n_labels,
                                                                           examples_per_label)
            data_steps.append((distilled_labels, distilled_data.requires_grad_(True), eta.requires_grad_(True)))
            params.append(distilled_data)
            params.append(eta)

        # For debug
        self.save_grid_img(torch.cat([data for (_, data, _) in data_steps]), 10, 'initial.jpg')

        # Define the optimizer for the distilled_data
        optimizer = DistillationTrainer.create_distilled_optimizer(params, alpha, distilled_optimizer_name, d_opt_args)

        # Main cycle
        self.model.train()
        for iteration in range(optimization_iterations):
            flat_weights = torch.empty(self.numels, device=self.device, requires_grad=True, dtype=torch.float)
            MetaModelUtils.set_flat_params(self.model, flat_weights)
            self.model.apply(weights_init_fn)
            for it , (data, labels) in enumerate(training_data_loader):
                if it == 2:
                    break
                data, labels = data.to(self.device), labels.to(self.device)
                for gd_step in range(gd_steps):
                    for d_labels, d_data, eta in data_steps:
                        out = self.model(d_data)
                        loss = self.loss_fn(out, d_labels)
                        print('Loss on distilled data: ', loss.item())

                        (gw,) = autograd.grad(loss, flat_weights, retain_graph=True, create_graph=True)
                        flat_weights = flat_weights - eta * gw
                        MetaModelUtils.set_flat_params(self.model, flat_weights)

                out = self.model(data)
                loss = self.loss_fn(out, labels)
                print('Loss on real data: ', loss.item())
                for _, d_data, eta in data_steps:
                    d_grad, eta_grad = autograd.grad(loss, (d_data, eta), retain_graph=True)
                    d_data.grad = d_grad
                    eta.grad = eta_grad
                loss.detach()

                optimizer.step()

        distilled_data = torch.cat([x for (_, x, _) in data_steps])
        self.save_grid_img(distilled_data, 10, 'final.jpg')
        distilled_labels = torch.cat([labels for (labels, _, _) in data_steps])
        distilled_data = distilled_data.detach()
        return distilled_data, distilled_labels
