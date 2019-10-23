import argparse
from json_configuration import JsonConfiguration

import torch
import torch.nn as nn

from distillation.distillation_trainer import DistillationTrainer
from networks.le_net import LeNet
from datasets.mnist import get_mnist_data_loader
from utils.weights import create_weights_init_fn
from utils.savings import save_distilled_log_data


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data-batch-size', type=int, default=1024)
    parser.add_argument('--distill-epochs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--distilled-batches', type=int, default=3)
    parser.add_argument('--n-classes', type=int)
    parser.add_argument('--examples-per-class', type=int, default=10)
    parser.add_argument('--distill-lr', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--optimizer-name', type=str, default='adam', choices=DistillationTrainer.OPTIMIZERS)
    parser.add_argument('--log-img-after', type=int, default=30)
    parser.add_argument('--log-epoch', type=int, default=1)
    return parser


def main():
    args = JsonConfiguration().parse(parser())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mnist_loader = get_mnist_data_loader('./data/', args.training_data_batch_size, training=True)
    network = LeNet()
    loss_fn = nn.CrossEntropyLoss()
    weights_init_fn = create_weights_init_fn(nn.init.xavier_normal_, gain=1.0)

    distillation_trainer = DistillationTrainer(network, device, loss_fn)
    distillation_trainer.distill(args.distill_epochs, args.epochs, args.distilled_batches, args.n_classes,
                                 args.examples_per_class, (1, 28, 28), args.distill_lr, args.optimizer_name, args.alpha,
                                 mnist_loader, weights_init_fn, save_log_fn=save_distilled_log_data,
                                 log_img_after=args.log_img_after, log_epoch=args.log_epoch)


if __name__ == '__main__':
    main()
