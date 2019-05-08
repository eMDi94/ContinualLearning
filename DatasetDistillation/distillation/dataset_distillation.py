import torch
import torch.optim as optim

from .weights import create_weights_distribution

from globals import device


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


def distill_dataset(model, weights_distribution, weights_batch_size, distilled_dataset_size, step_size,
                    optimization_iterations, lr0, data_loader, data_size, criterion):
    model = model.to(device)
    distilled_dataset_size = (distilled_dataset_size,) + data_size
    distilled_dataset_data = torch.randn(distilled_dataset_size, dtype=torch.float, device=device)

    weights_init = create_weights_distribution(weights_distribution)
    params = model.parameters()

    lr = torch.tensor(lr0, dtype=torch.float, device=device)
    data_loader_len = len(data_loader)
    it = None
    for idx in range(optimization_iterations):
        if idx % data_loader_len == 0:
            it = iter(data_loader)
        batch_data, batch_target = next(it)
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        loss_sum = torch.tensor(0., dtype=torch.float, device=device)
        for _ in range(weights_batch_size):
            for p in params:
                weights_init(tensor=p)
            model.train()
            lr.requires_grad_(False)
            distilled_dataset_data.requires_grad_(False)
            update_weights(model, lr, criterion, distilled_dataset_data, batch_target)
            model.eval()
            lr.requires_grad_(True)
            distilled_dataset_data.requires_grad_(True)
            loss = compute_loss_on_real_targets(model, criterion, batch_data, batch_target)
            loss_sum.add_(loss)
        loss_sum.backward()
        distilled_dataset_data -= step_size*batch_data.grad
        lr -= step_size*lr.grad

    return distilled_dataset_data, lr
