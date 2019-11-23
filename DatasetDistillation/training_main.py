import torch
import torch.nn as nn
import torch.autograd as autograd

from networks.le_net import LeNet
from datasets.mnist import get_mnist_data_loader
from parsers import training_parser
from utils.meta_model_utils import MetaModelUtils
from utils.weights import create_weights_init_fn


def main():
    parser = training_parser()
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = LeNet()
    net.to(device)

    training_data = torch.load(args.training_set)

    init_weights_fn = create_weights_init_fn(nn.init.xavier_normal_, gain=1.0)
    net.apply(init_weights_fn)
    weights = MetaModelUtils.get_flat_params(net)
    loss_fn = nn.CrossEntropyLoss()

    net.train()
    for data, label, lr in training_data:
        MetaModelUtils.set_flat_params(net, weights)
        data, label, lr = data.to(device), label.to(device), lr.to(device)
        out = net(data)
        loss = loss_fn(out, label)
        print('Loss: ', loss.item())

        (flat_grad,) = autograd.grad(loss, weights, grad_outputs=(lr,))
        weights = weights - flat_grad

    test_loader = get_mnist_data_loader('./data/', 1024, False)
    net.eval()
    accuracy_sum = 0.
    it = iter(test_loader)
    n = len(test_loader)

    with torch.no_grad():
        for data, label in it:
            data, label = data.to(device), label.to(device)
            out = net(data)
            _, predictions = torch.max(out, dim=1)
            c = (predictions == label)
            c = c.sum()
            c = c.item() / out.size(0)
            accuracy_sum += c

    print('Mean accuracy is: ', (accuracy_sum / n))


if __name__ == '__main__':
    main()
