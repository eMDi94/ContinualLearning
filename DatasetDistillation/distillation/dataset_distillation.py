import torch
import torch.nn as nn
import torch.optim as optim

from .weights import create_weights_distribution

from globals import device

import copy


def update_weights(model, batch_data, batch_target, lr, optimizer_type, criterion):
    optimizer = optimizer_type(model.parameters(), lr=lr)
    optimizer.zero_grad()
    out = model(batch_data)
    loss = criterion(out, batch_target)
    loss.backward()
    optimizer.step()


def record_gradients(model):
    grads = []
    for p in model.parameters():
        grad = p.grad.view(-1)
        grads.append(grad)
        p.zero_grad_()
    return grads


def compute_loss_on_real_data(model, batch_data, batch_target, criterion):
    out = model(batch_data)
    loss = criterion(out, batch_target)
    # No backward because all losses will be summed at the end
    return loss


def distill_dataset(model, number_distilled_examples, input_data_size, step_size, optimization_iterations, lr0,
                    training_data_loader, criterion, weights_batch_size, optimizer_type=optim.SGD,
                    weights_distribution=nn.init.xavier_normal_, weights_distribution_params=None):
    """
    Function responsible of dataset distillation
    :param model: nn.Module subclass which is used to predict the output values
    :param number_distilled_examples: The number of distilled examples that we want to get
    :param input_data_size: The size of a single input
    :param step_size: Step size used to update the distilled data and the learning rate
    :param optimization_iterations: Number of iterations required
    :param lr0: The initial value for the learning rate
    :param training_data_loader: DataLoader which gives a batch of training data
    :param criterion: The loss function used
    :param weights_batch_size: The number of batch weights used to perform the training and the distillation in a single iteration
    :param optimizer_type: The optimizer type, one the torch.optim optimizer
    :param weights_distribution: Distribution of the weights, one of the torch.nn.init functions
    :param weights_distribution_params: Dictionary which contains the extra parameters to pass the weights init function
    :return: torch.Tensor of distilled data, torch.Tensor of distilled targets, learned learning rate
    """
    # Normalize the weights_distribution_params dictionary
    weights_distribution_params = weights_distribution_params if weights_distribution_params is not None else {}

    # Send the model to the selected device
    model = model.to(device)

    # Create the distilled data and targets and initialize them randomly, and then the learning rate
    distilled_data = torch.randn((number_distilled_examples, ) + input_data_size, dtype=torch.float,
                                 device=device)
    distilled_target = torch.randn(number_distilled_examples, dtype=torch.float, device=device)
    lr = torch.tensor(lr0, dtype=torch.float, device=device)

    params = model.parameters()

    # Create the weights distribution
    weights_init = create_weights_distribution(weights_distribution, **weights_distribution_params)

    it = None
    it_copy = copy.deepcopy(iter(training_data_loader))
    data_loader_len = len(training_data_loader)
    recording = []
    for iteration in range(optimization_iterations):
        if iteration % data_loader_len == 0:
            it = copy.deepcopy(iter(it_copy))
        batch_data, batch_target = next(it)
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        loss_sum = torch.tensor(0., dtype=torch.float, device=device)
        gradients_products = torch.zeros(weights_batch_size, len(params))
        for weights_batch_number in range(weights_batch_size):
            for p in params:
                weights_init(tensor=p)
            # Weights update step plus gradients recording
            model = model.train()
            distilled_data.requires_grad_(False)
            distilled_target.requires_grad_(False)
            lr.requires_grad_(False)
            update_weights(model, distilled_data, distilled_target, lr, optimizer_type, criterion)
            if optimization_iterations % 50 == 0:
                grads1 = record_gradients(model)

            # Loss recording step plus gradients recording
            model = model.eval()
            distilled_data.requires_grad_(True)
            distilled_target.requires_grad_(True)
            lr.requires_grad_(True)
            loss = compute_loss_on_real_data(model, batch_data, batch_target, criterion)
            loss_sum += loss
            if optimization_iterations % 50 == 0:
                grads2 = record_gradients(model)

            # Dot product between the gradients for recording
            if optimization_iterations % 50 == 0:
                for idx, (g1, g2) in enumerate(zip(grads1, grads2)):
                    gradients_products[weights_batch_number, idx] = g1.dot(g2)
                mean = torch.mean(gradients_products, dim=0)
                recording.append(mean)

        loss_sum.backward()
        distilled_data -= step_size * distilled_data.grad
        distilled_target -= step_size * distilled_target.grad
        lr -= step_size * lr.grad

    return distilled_data, distilled_target, lr, recording
