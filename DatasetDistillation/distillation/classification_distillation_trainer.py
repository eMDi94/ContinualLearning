import torch

from utils import MetaModelUtils


class ClassificationDistillationTrainer(object):

    def __init__(self, model, optimization_iterations, data_size, weights_init_fn, learning_rate, weights_batch_size,
                 loss_fn, alpha, device):
        self.model = model
        self.T = optimization_iterations
        self.device = device
        self.weights_init_fn = weights_init_fn
        self.eta = learning_rate
        self.weights_batch_size = weights_batch_size
        self.data_size = data_size
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.distilled_labels = None
        self.distilled_data = None
        self.distilled_learning_rate = None

    def __get_next_batch(self, it, loader):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        return b, it

    def distill_(self, training_data_loader, n_labels, exemples_per_label):
        distilled_labels = torch.arange(n_labels).repeat(exemples_per_label).reshape(exemples_per_label, 1)\
            .transpose(1, 0).reshape(-1)
        distilled_data_size = torch.Size([distilled_labels.size(0)]) + torch.Size(self.data_size)
        distilled_data = torch.randn(distilled_data_size, device=self.device, dtype=torch.float)
        eta = torch.tensor([self.eta], device=self.device)

        self.model.train()

        it = iter(training_data_loader)
        for iteration in range(self.T):
            data, labels = self.__get_next_batch(it, training_data_loader)
            data = data.to(self.device)
            labels = labels.to(self.device)

            # Make distilled_data and eta able to obtain gradient
            distilled_data.requires_grad_(True)
            eta.requires_grad_(True)

            distilled_data_grads = []
            eta_grads = []
            for weights_batch_index in range(self.weights_batch_size):
                # Weights batch init
                self.model.apply(self.weights_init_fn)

                self.model.zero_grad()

                out = self.model(distilled_data)
                loss = self.loss_fn(out, distilled_labels)

                # Backward on loss
                # Flag retain_graph=True allows to maintain the computational graph
                # Flag create_graph=True allows to create the graph on the computed derivatives
                loss.backward(retain_graph=True, create_graph=True)

                # Get flat weights and flat grad to perform the update on the weights
                flat_weights = MetaModelUtils.get_flat_params(self.model)
                flat_grads = MetaModelUtils.get_flat_grads(self.model)
                flat_grads = self.eta.item() * flat_grads
                flat_weights = flat_weights - flat_grads
                MetaModelUtils.set_flat_params(self.model, flat_weights)

                # Compute the loss on the training_data
                self.model.zero_grad()
                out = self.model(data)
                final_loss = self.loss_fn(out, labels)

                # Performing now the backward allows to obtain the gradients of the distilled data
                # and distilled learning rate. This time is not necessary to set retain_graph=True and
                # create_graph=True
                final_loss.backward()

                # Save gradients of distilled data and distilled learning rate
                distilled_data_grads.append(distilled_data.grad)
                eta_grads.append(eta.grad)

            # Set both distilled_data and eta to requires_grad=False
            distilled_data.requires_grad_(False)
            eta.requires_grad_(False)
            distilled_data -= self.alpha * torch.tensor(distilled_data_grads, device=self.device).sum(dim=0)
            eta -= self.alpha * torch.tensor(eta_grads, device=self.device).sum()

        self.model.eval()
        self.distilled_data = distilled_data.detach()
        self.distilled_labels = distilled_labels
        self.distilled_learning_rate = eta.detach()

        return self.distilled_data, self.distilled_labels, self.distilled_learning_rate
