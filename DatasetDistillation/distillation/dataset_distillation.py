import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

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
                    step_size, optimization_iterations, lr0, dataset, criterion):
    model.to(device)
    distilled_dataset_size = (distilled_dataset_size,) + dataset[0].size()[1:]
    distilled_dataset_data = torch.randn(distilled_dataset_size, dtype=torch.float, device=device, requires_grad=True)

    weights_init = create_weights_distribution(weights_distribution)
    params = model.parameters()

    lr = torch.tensor(lr0, dtype=torch.float, device=device, requirer_grad=True)

    for _ in range(optimization_iterations):
        sampler = BatchSampler(RandomSampler(dataset), input_data_batch_size)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=input_data_batch_size)
        batch = next(iter(data_loader))
        batch_data = batch[:][0]
        batch_target = batch[:][1]

        losses = []
        for _ in range(weights_batch_size):
            for p in params:
                weights_init(tensor=p)
            model.train()
            update_weights(model, lr, criterion, batch_data, batch_target)
            model.eval()
            loss = compute_loss_on_real_targets(model, criterion, batch_data, batch_target)
            losses.append(loss.item())
        losses_sum = torch.tensor(losses).sum()
        losses_sum.backward()
        losses_sum_by_x = distilled_dataset_data.grad
        losses_sum_by_lr = lr.grad
        distilled_dataset_data -= step_size*losses_sum_by_x
        lr -= step_size*losses_sum_by_lr

    return distilled_dataset_data
