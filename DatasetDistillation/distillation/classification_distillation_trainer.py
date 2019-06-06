import torch
import torch.autograd as autograd

from utils import MetaModelUtils
from .base_distillation_trainer import BaseDistillationTrainer


class ClassificationDistillationTrainer(BaseDistillationTrainer):

    def __init__(self, model, optimization_iterations, data_size, weights_init_fn, learning_rate, weights_batch_size,
                 loss_fn, alpha, device):
        super(ClassificationDistillationTrainer, self).__init__(model, optimization_iterations, data_size,
                                                                weights_init_fn, learning_rate, weights_batch_size,
                                                                loss_fn, alpha, device)

    def distill(self, training_data_loader, n_labels, examples_per_label):
        distilled_labels = torch.arange(n_labels, device=self.device).repeat(examples_per_label)
        distilled_labels = distilled_labels.reshape(examples_per_label, -1).transpose(1, 0).reshape(-1)
        distilled_data_size = distilled_labels.size() + torch.Size(tuple(self.data_size))
        distilled_data = torch.randn(distilled_data_size, device=self.device, requires_grad=True, dtype=torch.float)
        eta = torch.tensor([self.eta], device=self.device, requires_grad=True)

        self.model.train()

        it = iter(training_data_loader)
        for iteration in range(self.T):
            print('Optimization iteration ', iteration + 1, ' started.')
            (data, labels), it = self._get_next_batch(it, training_data_loader)
            data = data.to(self.device)
            labels = labels.to(self.device)

            loss_grad_wrt_distilled_data = torch.zeros_like(distilled_data, device=self.device, dtype=torch.float)
            loss_grad_wrt_eta = torch.zeros_like(eta, device=self.device, dtype=torch.float)
            for weights_batch_index in range(self.weights_batch_size):
                # Weights batch init
                self.model.apply(self.weights_init_fn)
                MetaModelUtils.reset(self.model)

                with torch.enable_grad():
                    self.model.zero_grad()

                    out = self.model(distilled_data)
                    loss = self.loss_fn(out, distilled_labels)

                    # Backward on loss
                    # Flag retain_graph=True allows to maintain the computational graph
                    # Flag create_graph=True allows to create the graph on the computed derivatives
                    loss.backward(retain_graph=True, create_graph=True)

                    # Get flat weights and flat grad to perform the update on the weights
                    flat_weights = MetaModelUtils.get_flat_params(self.model)
                    flat_grads = MetaModelUtils.get_flat_grads(self.model).view(-1)
                    flat_grads.mul_(eta)
                    flat_weights.sub_(flat_grads)
                    MetaModelUtils.set_flat_params(self.model, flat_weights)
                    del flat_weights, flat_grads

                    # Compute the loss on the training_data
                    out = self.model(data)
                    final_loss = self.loss_fn(out, labels)

                    # Performing now the backward allows to obtain the gradients of the distilled data
                    # and distilled learning rate. This time is not necessary to set retain_graph=True and
                    # create_graph=True. allow_unused is necessary to compute the gradient with respect to x and eta
                    # since they are not directly used during the computation of final loss
                    x_grad, eta_grad = autograd.grad(final_loss, (distilled_data, eta), allow_unused=True,
                                                     retain_graph=False)

                    # Update the gradients sum
                    loss_grad_wrt_distilled_data.sub_(x_grad)
                    loss_grad_wrt_eta.sub_(eta_grad)

                    del x_grad, eta_grad

                    if weights_batch_index % 50 == 0:
                        print('Working...')

            with torch.no_grad():
                distilled_data.sub_(self.alpha * loss_grad_wrt_distilled_data)
                eta.sub_(self.alpha * loss_grad_wrt_eta)

        print('Distillation ended.')
        self.model.eval()
        self._distilled_data = distilled_data.detach().cpu()
        self._distilled_targets = distilled_labels.cpu()
        self._distilled_learning_rate = eta.detach().cpu().item()
