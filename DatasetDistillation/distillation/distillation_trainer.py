import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

from utils.meta_model_utils import MetaModelUtils


class DistillationTrainer(object):

    __OPTIMIZERS = {
        'sgd': optim.SGD,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adadelta': optim.Adadelta
    }
    OPTIMIZERS = __OPTIMIZERS.keys()

    def __init__(self, models, device, distill_epochs, epochs, n_distilled_batches, n_classes, examples_per_class,
                 data_size, initial_eta, optimizer_name, alpha, loss_fn, weights_init_fn, train_loader,
                 optimizer_kwargs=None, save_log_fn=None, log_img_after=30, log_epoch=5):
        self.models = models
        for model in self.models:
            model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.distill_epochs = distill_epochs
        self.epochs = epochs
        self.weights_init_fn = weights_init_fn
        self.train_loader = train_loader
        self.save_log_fn = save_log_fn
        self.log_img_after = log_img_after
        self.log_epoch = log_epoch
        self.distilled_data_batches = None
        self.distilled_labels_batches = None
        self.eta = None
        self.optimizer = None
        self.initialize_data_and_optimizer(n_distilled_batches, n_classes, examples_per_class,
                                           data_size, initial_eta, optimizer_name, alpha,
                                           optimizer_kwargs if optimizer_kwargs is not None else {})

    def create_distilled_optimizer(self, params, optimizer_name, alpha, optimizer_kwargs):
        if optimizer_name not in DistillationTrainer.OPTIMIZERS:
            raise ValueError('The given optimizer is not recognized')
        else:
            optimizer_class = DistillationTrainer.__OPTIMIZERS[optimizer_name]
            optimizer = optimizer_class(params, alpha, **optimizer_kwargs)
            return optimizer

    def initialize_data_and_optimizer(self, n_distilled_batches, n_classes, examples_per_class, data_size,
                                      initial_eta, optimizer_name, alpha, optimizer_kwargs):
        optimizer_params = []

        labels = []
        data = []
        distilled_labels = torch.arange(n_classes, dtype=torch.long, device=self.device)\
            .repeat(examples_per_class, 1)
        distilled_labels = distilled_labels.t().reshape(-1)
        for _ in range(n_distilled_batches):
            labels.append(distilled_labels)
            distilled_data = torch.randn(n_classes * examples_per_class, *data_size,
                                         device=self.device, dtype=torch.float, requires_grad=True)
            distilled_data.grad = torch.zeros_like(distilled_data, device=self.device, dtype=distilled_data.dtype)
            data.append(distilled_data)
            optimizer_params.append(distilled_data)
        self.distilled_data_batches = data
        self.distilled_labels_batches = labels

        eta = torch.tensor(initial_eta, device=self.device)
        eta = eta.repeat(n_distilled_batches * self.epochs, 1)
        eta = eta.expm1_().log_().requires_grad_(True)
        self.eta = eta
        optimizer_params.append(eta)

        self.optimizer = self.create_distilled_optimizer(optimizer_params, optimizer_name, alpha, optimizer_kwargs)

    def get_data_steps(self):
        data_label = [dl for _ in range(self.distill_epochs)
                      for dl in zip(self.distilled_data_batches, self.distilled_labels_batches)]
        etas = F.softplus(self.eta).unbind()

        data_steps = []
        for (distilled_data, distilled_labels), eta in zip(data_label, etas):
            data_steps.append((distilled_data, distilled_labels, eta))

        return data_steps

    def forward(self, model, real_data, real_labels, data_steps):
        flat_weights = MetaModelUtils.get_flat_params(model)
        flat_weights.requires_grad_(True)
        weights_params = [flat_weights]
        gradient_weights_params = []

        model.train()
        for iteration, (distilled_data, distilled_labels, eta) in enumerate(data_steps):
            MetaModelUtils.set_flat_params(model, flat_weights)
            out = model(distilled_data)
            loss = self.loss_fn(out, distilled_labels)
            print('Loss on distilled data: ', loss.item())
            (grad_weights,) = autograd.grad(loss, flat_weights, eta, create_graph=True)

            with torch.no_grad():
                new_flat_weights = flat_weights.sub(grad_weights).requires_grad_(True)
                weights_params.append(new_flat_weights)
                gradient_weights_params.append(grad_weights)
                flat_weights = new_flat_weights

        MetaModelUtils.set_flat_params(model, weights_params[-1])
        model.eval()
        out = model(real_data)
        final_loss = self.loss_fn(out, real_labels)
        print('Loss on real data: ', final_loss.item())
        return final_loss, weights_params, gradient_weights_params

    def backward(self, model, data_steps, infos_for_backward):
        # Notation:
        # - L is the loss w.r.t. the real data
        # - l is the loss w.r.t. the distilled data

        loss, weights, grad_weights = infos_for_backward

        data = []
        gradient_data = []
        etas = []
        gradient_etas = []

        # Gradient of the loss on real data w.r.t. the last weights
        (dLdw,) = autograd.grad(loss, (weights[-1],))

        # Set the model in train mode for the backward
        model.train()
        for (distilled_data, _, eta), weight, grad_weight in reversed(list(zip(data_steps, weights, grad_weights))):
            hessian_in = list()
            hessian_in.append(weight)
            hessian_in.append(distilled_data)
            hessian_in.append(eta)
            dgw = dLdw.neg()
            hessian_out = autograd.grad(outputs=(grad_weight,),
                                        inputs=hessian_in,
                                        grad_outputs=(dgw,))
            with torch.no_grad():
                data.append(distilled_data)
                gradient_data.append(hessian_out[1])
                etas.append(eta)
                gradient_etas.append(hessian_out[2])

                dLdw.add_(hessian_out[0])

        return data, gradient_data, etas, gradient_etas

    def accumulate_gradients(self, grad_infos):
        bwd_out = list()
        bwd_in = list()
        for datas, grad_datas, etas, grad_etas in grad_infos:
            bwd_out += list(etas)
            bwd_in += list(grad_etas)
            for data, grad_data in zip(datas, grad_datas):
                data.grad.add_(grad_data)
        if len(bwd_out) > 0:
            autograd.backward(bwd_out, bwd_in)

    def distill(self):
        for epoch in range(self.distill_epochs):
            for iteration, (real_data, real_labels) in enumerate(iter(self.train_loader)):
                print('Epoch: ', epoch, ' Iteration: ', iteration, ' started')
                if (epoch % self.log_epoch == 0) and (iteration % self.log_img_after == 0):
                    self.save_log_distilled_data(epoch, iteration, 'start', self.save_log_fn,
                                                 self.distilled_data_batches, self.distilled_labels_batches)
                self.optimizer.zero_grad()
                real_data, real_labels = real_data.to(self.device), real_labels.to(self.device)

                data_steps = self.get_data_steps()

                grad_infos = []
                for model in self.models:
                    model.apply(self.weights_init_fn)
                    saved_infos = self.forward(model, real_data, real_labels, data_steps)
                    info = self.backward(model, data_steps, saved_infos)
                    grad_infos.append(info)
                    del saved_infos

                self.accumulate_gradients(grad_infos)

                self.optimizer.step()
                if (epoch % self.log_epoch == 0) and (iteration % self.log_img_after == 0):
                    self.save_log_distilled_data(epoch, iteration, 'end', self.save_log_fn,
                                                 self.distilled_data_batches, self.distilled_labels_batches)
                    del data_steps, grad_infos

        self.save_log_distilled_data('end', 'end', 'end', self.save_log_fn,
                                     self.distilled_data_batches, self.distilled_labels_batches)
        data_steps = self.get_data_steps()
        data_steps = [(d.detach(), l.detach(), eta.detach()) for (d, l, eta) in data_steps]
        return data_steps

    @property
    def distilled_data(self):
        return torch.cat(self.distilled_data_batches)

    @property
    def distilled_labels(self):
        return torch.cat(self.distilled_labels_batches)

    def save_log_distilled_data(self, epoch, iteration, sub, save_function, distilled_data, distilled_labels):
        distilled_data = torch.cat([d.detach().cpu() for d in distilled_data])
        distilled_labels = torch.cat([l.detach().cpu() for l in distilled_labels])
        save_function(distilled_data, distilled_labels,
                      './log/epoch' + str(epoch) + '/iteration' + str(iteration) + '/' + str(sub) + '/')
