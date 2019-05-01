import torch
import torch.optim as optim

from .weights import create_weights_distribution

from globals import device


def get_random_permutation(n_samples, batch_size):
    permutation = torch.randperm(n_samples)
    return permutation[:batch_size]


def update_weights(model, lr, criterion, batch, target):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    out = model(batch)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()


def compute_loss_on_real_targets(model, criterion, batch, target):
    out = model(batch)
    loss = criterion(out, target)
    loss.backward()
    return loss


def distill_dataset(model, weights_distribution, weights_batch_size, distilled_dataset_size, input_data_batch_size,
                    step_size, optimization_iterations, lr0, train_set_data, train_set_target, criterion):
    model.to(device)
    distilled_dataset_targets = torch.randn(train_set_target.size(), dtype=torch.int, device=device, requires_grad=True)
    distilled_dataset_size = (distilled_dataset_size,) + train_set_data.size()[1:]
    distilled_dataset_data = torch.randn(distilled_dataset_size, dtype=torch.float, device=device, requires_grad=True)

    weights_init = create_weights_distribution(weights_distribution)
    params = model.parameters()

    lr = torch.tensor(lr0, dtype=torch.float, device=device)

    for _ in range(optimization_iterations):
        rand_idxs = get_random_permutation(train_set_data.size()[0], input_data_batch_size)
        batch_data = train_set_data[rand_idxs]
        batch_target = train_set_target[rand_idxs]

        losses = []
        for _ in range(weights_batch_size):
            for p in params:
                weights_init(tensor=p)
            update_weights(model, lr, criterion, batch_data, batch_target)
            loss = compute_loss_on_real_targets(model, criterion, batch_data, batch_target)
            losses.append(loss)
        losses_sum = torch.sum(torch.tensor(losses))
        losses_sum.backward()
        losses_sum_by_x = distilled_dataset_data.grad
        losses_sum_by_lr = lr.grad
        distilled_dataset_data -= step_size*losses_sum_by_x
        distilled_dataset_targets -= step_size*losses_sum_by_x
        lr -= step_size*losses_sum_by_lr

    return distilled_dataset_data, distilled_dataset_targets
