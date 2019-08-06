import torch
import torch.autograd as autograd

from utils import MetaModelUtils
from .base_distillation_trainer import BaseDistillationTrainer

# angpo github

class ClassificationDistillationTrainer(BaseDistillationTrainer):

    def __init__(self, model, optimization_iterations, data_size, weights_init_fn,
                 learning_rate, weights_batch_size, loss_fn, alpha, device):
        super(ClassificationDistillationTrainer, self).__init__(model, optimization_iterations, data_size,
                                                                weights_init_fn, learning_rate, weights_batch_size,
                                                                loss_fn, alpha, device)

    def _single_computation(self, data, labels, distilled_data, distilled_labels, eta):

        self.model.apply(self.weights_init_fn)
        MetaModelUtils.reset(self.model)
        self.model.zero_grad()

        out = self.model(distilled_data)
        loss = self.loss_fn(out, distilled_labels)

        flat_weights = MetaModelUtils.get_flat_params(self.model)
        #flat_grads = autograd.grad(loss, flat_weights, retain_graph=True, create_graph=True)[0]
        flat_grads = torch.cat([ x.view(-1) for x in autograd.grad(loss, self.model.parameters(),
                                                                   retain_graph=True, create_graph=True)])
        MetaModelUtils.set_flat_params(self.model, flat_weights - eta * flat_grads)

        out = self.model(data)
        final_loss = self.loss_fn(out, labels)

        x_grad, eta_grad = autograd.grad(final_loss, (distilled_data, eta), retain_graph=False, create_graph=False)

        print('Final Loss: ', final_loss.item(), ' eta: ', eta.item())

        return x_grad, eta_grad

    def distill(self, training_data_loader, n_labels, examples_per_label):

        # Create the distilled data variables
        distilled_labels = torch.arange(n_labels, device=self.device).repeat(examples_per_label)
        distilled_labels = distilled_labels.reshape((examples_per_label, -1)).transpose(1, 0).reshape((-1,))
        distilled_data_size = distilled_labels.size() + torch.Size(tuple(self.data_size))
        # todo: Initialize distilled data with the mean and variance of mnist
        distilled_data = torch.rand(distilled_data_size, device=self.device, dtype=torch.float, requires_grad=True)
        eta = torch.tensor([self.eta], device=self.device, requires_grad=True)

        # Set the model in training mode
        self.model.train()

        # Now start the computation
        it = iter(training_data_loader)
        for iteration in range(self.T):
            print('Optimization iteration ' + str(iteration) + ' started...')

            (data, labels), it = self._get_next_batch(it, training_data_loader)

            # Send the data to the selected device
            data, labels = data.to(self.device), labels.to(self.device)

            loss_grad_wrt_distilled_data = torch.zeros_like(distilled_data, device=self.device, dtype=torch.float)
            loss_grad_wrt_eta = torch.zeros_like(eta, device=self.device, dtype=torch.float)

            for weights_batch_index in range(self.weights_batch_size):

                x_grad, eta_grad = self._single_computation(data, labels, distilled_data, distilled_labels, eta)
                loss_grad_wrt_distilled_data.add_(x_grad)
                loss_grad_wrt_eta.add_(eta_grad)

                if weights_batch_index % 50 == 0:
                    print('Working...')

            distilled_data.data.add_(-self.alpha * loss_grad_wrt_distilled_data)
            eta.data.add_(-self.alpha * loss_grad_wrt_eta)

        print('Distillation Ended')
        self.model.eval()
        self._distilled_data = distilled_data.detach().cpu()
        self._distilled_targets = distilled_labels.cpu()
        self._distilled_learning_rate = eta.detach().cpu().item()
